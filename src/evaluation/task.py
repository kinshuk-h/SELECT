import os
import math
import random
import itertools

import tqdm.auto as tqdm

from src.utils import data, common
from src.evaluation import refusal, dataset
from src.inference import ModelManager, seed_all, sync_vram

def model_size(model_key):
    """ Estimates the model size with the key name (no actual parameters involved). """

    segments = model_key.lower().split('-')
    while segments:
        if segments[-1].endswith('b') or segments[-1].endswith('m'):
            weights = eval(segments[-1][:-1].replace('x', '*')) if 'x' in segments[-1][:-1] else int(segments[-1][:-1])
            value, multiplier = weights, segments[-1][-1]
            return value * (1000 if multiplier == 'b' else 1)
        segments = segments[:-1]
    return 9001

def batch_multiplier(model_key):
    """ Crude estimate to adjust batch size for optimal GPU use (based on model size). """

    m_size = model_size(model_key)
    if m_size > 9_000: return 1
    if m_size > 5_000: return 2
    return 4

# ==================================================

class Evaluator:
    def __init__(self, dataset_state: dataset.DatasetState,
                 eval_dataset, models, approaches,
                 batch_size=16, save_steps=1,
                 root="results/expr.abstain", sample=True,
                 seed=20240801, num_seeds=5, **kwargs):

        self.dataset_state = dataset_state
        self.dataset_state.eval_dataset = eval_dataset

        self.eval_dataset  = eval_dataset
        self.batch_size    = batch_size
        self.save_steps    = save_steps
        self.root          = root
        self.sample        = sample

        self.seed = seed
        random.seed(seed)
        self.seeds = [ random.randint(100000+(i*100000), 100000+((i+1)*100000)) for i in range(num_seeds) ]

        self.models     = models
        self.approaches = approaches

        self.__refresh_state()

        if self.dataset_state.compose:
            if 'atomic_ds_state' not in kwargs:
                raise ValueError(
                    f"{self.__class__.__name__}: compositional dataset state requires "
                    "companion atomic dataset state (in property 'atomic_ds_state')"
                )
            else:
                self.atomic_state = kwargs['atomic_ds_state']
                self.atomic_state.eval_dataset = eval_dataset

    def __refresh_state(self):
        trav_vars = dataset.get_traversal_variables(self.taxonomy, self.compose_mode)
        for var_name, var in trav_vars.items(): setattr(self, var_name, var)

    def set_state(self, dataset_state: dataset.DatasetState, refresh=True):
        old_state          = self.dataset_state
        self.dataset_state = dataset_state
        if refresh: self.__refresh_state()
        return old_state

    # ------------------------------------------------------------------------------------------------------------

    @property
    def taxonomy(self):
        return self.dataset_state.taxonomy

    @property
    def dataset_name(self):
        return self.dataset_state.name

    @property
    def dataset(self):
        return self.dataset_state.dataset

    @property
    def train_dataset(self):
        return self.dataset_state.train_dataset

    @property
    def example_cache(self):
        return self.dataset_state.example_cache

    @property
    def dataset_type(self):
        return self.dataset_state.dtype

    @property
    def compose_mode(self):
        return self.dataset_state.compose

    # ------------------------------------------------------------------------------------------------------------

    def sample_queries(self, concepts: list[str|tuple[str,str]], num_queries, random_state=20240616):
        random.seed(random_state)
        concepts = [
            concept if isinstance(concept, (list, tuple)) else (concept, )
            for concept in sorted(concepts)
        ]
        return dataset.sample_queries(
            self.dataset, concepts, num_queries, return_concepts=False
        )

    def efficient_run_for_all(self, modes, compose_run, preview=False, range=None, **model_kwargs):
        mode_data = {}

        self.compose_state = self.dataset_state
        if not compose_run: self.set_state(self.atomic_state)
        concepts = [ concept for concept, _ in dataset.concept_iterator(self.node_data, self.compose_mode) ]

        if range:
            start, end, *step = [ int(x) for x in range.split(':') ]
            step = step[0] if step else 1
            concepts = concepts[start:end:step]

        for _mode, compose_flag in modes.items():
            self.set_state(self.compose_state if compose_flag else self.atomic_state)

            run_mode = ('compose' if self.compose_mode else 'atomic')
            _, num_queries = getattr(self, f'estimate_iterations_{run_mode}')(_mode)

            instances_collation = getattr(self, f'prepare_instances_{run_mode}')(
                num_queries, _mode, random_state = self.seed
            )

            mode_data[_mode] = instances_collation

        self.set_state(self.compose_state if compose_run else self.atomic_state)

        for model_key, model_name in self.models:
            batch_size = batch_multiplier(model_key) * self.batch_size

            for seed in self.seeds:
                seed_all(seed)

                print("> Using", model_key, "for inference ...")

                model_kwargs['seed'] = seed
                with ModelManager(model_name, **model_kwargs) as model:

                    results = {
                        mode: {
                            approach: data.NestedListItemResult(
                                os.path.join(self.root + ('.compose' if c_flag else ''), model_key, mode, f"{approach}.json"),
                            )
                            for approach in self.approaches
                        }
                        for mode, c_flag in modes.items()
                    }
                    gen_kwargs = dict(temperature=0.6, top_p=0.9, do_sample=True)

                    for concept in (pbar := tqdm.tqdm(concepts)):

                        pbar.set_description(self.node_data.deepget(concept)['name'])

                        dataset_state = self.compose_state if compose_run else self.atomic_state
                        _labels =  [ concept ]

                        for approach, approach_inst in self.approaches.items():
                            configs, ids, prompts, labels = [], [], [], []

                            for mode in modes:
                                if mode == 'generalization.atomic':
                                    instances = []
                                    for _instances in mode_data[mode].values():
                                        for instance in _instances:
                                            if instance['label'][-1] == concept:
                                                instances.append(instance)
                                else:
                                    instances = mode_data[mode].get(concept, [])

                                for instance in instances:
                                    configs.append((mode, approach))
                                    id = dataset.generate_id(instance['query'])
                                    if not results[mode][approach].deepget((mode, str(seed), *instance['label'], id)):
                                        ids.append(id); labels.append(instance['label'])
                                        prompts.append(approach_inst.prepare_for_inference(
                                            instance['concept_desc'], instance['query'],
                                            examples=instance['examples']
                                        ))

                            if not prompts: continue

                            pbar.set_postfix(dict(prepare=approach))
                            if not preview:
                                approach_inst.prepare(
                                    model_key, model, dataset_state, _labels, seed=seed, node_data=self.node_data,
                                    atomic_state=getattr(self, 'atomic_state', None), sibling_map=self.sibling_map,
                                    aux_model=True
                                )

                            label_idx = sorted(( (lbl, i) for i, lbl in enumerate(labels) ))
                            ids     = [ ids[idx] for _, idx in label_idx ]
                            prompts = [ prompts[idx] for _, idx in label_idx ]
                            labels  = [ labels[idx]  for _, idx in label_idx ]
                            configs = [ configs[idx] for _, idx in label_idx ]

                            batches = common.batchify(ids, prompts, labels, configs, batch_size=batch_size)
                            timer = common.BatchProgressTimer(pbar, total=math.ceil(len(labels)/batch_size))

                            for batch, (_ids, _prompts, _labels, _configs) in enumerate(batches):
                                with timer.timed_operation(batch=batch+1, approach=approach): #, save=((batch+1) % self.save_steps == 0)):
                                    __labels = _labels
                                    if not compose_run:
                                        __labels = [ _label[1:] if len(_label) > 1 else _label for _label in _labels ]

                                    if not preview:
                                        outputs = approach_inst.generate(model, _prompts, concepts=__labels, **gen_kwargs)
                                    else:
                                        outputs = _prompts

                                    for config, label, id, response in zip(_configs, _labels, _ids, outputs):
                                        if not preview:
                                            results[config[0]][config[1]].deepset((config[0], str(seed), *label, id), dict(
                                                id=id, label=label, response=response,
                                                refusal={
                                                    method: bool(refusal.check_refusal(
                                                        response['answer'] if isinstance(response, dict)
                                                        else response, method
                                                    ))
                                                    for method in refusal.EVALUATION_METHODS
                                                }
                                            ))

                                    if not preview and ((batch+1) % self.save_steps == 0):
                                        for mode, approach in set(_configs):
                                            results[mode][approach].save()

                                    if not preview:
                                        sync_vram()

                            if not preview and len(labels) > 0:
                                sync_vram()
                                for mode, approach in set(configs):
                                    results[mode][approach].save()

                        for approach, approach_inst in self.approaches.items():
                            for adapter_path in getattr(approach_inst, 'adapters', {}).values():
                                pbar.set_postfix(dict(delete=approach))
                                # shutil.rmtree(adapter_path)

    def _run(self, pbar, model_key, model, approach, results,
             prompts, ids, labels, eval_type, seed, preview, batch_size):
        gen_kwargs = dict(temperature=0.6, top_p=0.9, do_sample=True)

        if not preview:
            dataset_state = self.atomic_state if eval_type == 'generalization.atomic' else self.dataset_state
            _labels = sorted(set(labels))

            if eval_type == 'generalization.atomic':
                _labels = [ label[1:] for label in _labels ]
                trav_vars = dataset.get_traversal_variables(self.atomic_state.taxonomy, False)
                sibling_map, node_data = trav_vars['sibling_map'], trav_vars['node_data']
            else:
                sibling_map, node_data = self.sibling_map, self.node_data

            approach.prepare(
                model_key, model, dataset_state, _labels, seed=seed, node_data=node_data,
                atomic_state=getattr(self, 'atomic_state', None), sibling_map=sibling_map
            )

        label_idx = sorted(( (lbl, i) for i, lbl in enumerate(labels) ))
        ids     = [ ids[idx] for _, idx in label_idx ]
        prompts = [ prompts[idx] for _, idx in label_idx ]
        labels  = [ labels[idx] for _, idx in label_idx ]

        batches = common.batchify(ids, prompts, labels, batch_size=batch_size)
        timer = common.BatchProgressTimer(pbar, total=math.ceil(len(labels)/batch_size))

        for batch, (_ids, _prompts, _labels) in enumerate(batches):
            with timer.timed_operation(batch=batch+1): #, save=((batch+1) % self.save_steps == 0)):
                __labels = _labels
                if eval_type == 'generalization.atomic':
                    __labels = [ _label[1:] for _label in _labels ]
                if not preview:
                    outputs = approach.generate(model, _prompts, concepts=__labels, **gen_kwargs)
                else:
                    outputs = _prompts

                for label, id, response in zip(_labels, _ids, outputs):
                    if not preview:
                        results.deepset((eval_type, str(seed), *label, id), dict(
                            id=id, label=label, response=response,
                            refusal={
                                method: bool(refusal.check_refusal(
                                    response['answer'] if isinstance(response, dict)
                                    else response, method
                                ))
                                for method in refusal.EVALUATION_METHODS
                            }
                        ))
                    pbar.update()

                if not preview and ((batch+1) % self.save_steps == 0):
                    results.save()

                if not preview:
                    sync_vram()

        if not preview and len(labels) > 0:
            sync_vram()
            results.save()

    def run(self, eval_type, preview=False, random_state=20240416, **model_kwargs):
        run_mode = ('compose' if self.compose_mode else 'atomic')

        num_concepts, num_queries = getattr(self, f'estimate_iterations_{run_mode}')(eval_type)
        total_iters = len(self.approaches) * len(self.seeds) * num_concepts * num_queries

        instances_collation = getattr(self, f'prepare_instances_{run_mode}')(
            num_queries, eval_type, random_state = self.seed or random_state
        )

        for model_key, model_name in self.models.items():
            batch_size = batch_multiplier(model_key) * self.batch_size
            if model_kwargs.get('backend', 'huggingface') == 'vllm' and '-U' not in model_key: batch_size = 1024

            print("> Using", model_key, "for inference ...")

            with tqdm.tqdm(total = total_iters) as pbar:
                for seed in self.seeds:
                    seed_all(seed)

                    model_kwargs['seed'] = seed
                    with ModelManager(model_name, **model_kwargs) as model:

                        for approach, approach_inst in self.approaches.items():
                            pbar.set_description(approach_inst.short_name)

                            results = data.NestedListItemResult(
                                os.path.join(self.root, model_key, eval_type, f"{approach}.json"),
                            )

                            ids, prompts, labels = [], [], []

                            for instances in instances_collation.values():
                                for instance in instances:
                                    id = dataset.generate_id(instance['query'])
                                    if not results.deepget((eval_type, str(seed), *instance['label'], id)):
                                        ids.append(id); labels.append(instance['label'])
                                        prompts.append(approach_inst.prepare_for_inference(
                                            instance['concept_desc'], instance['query'],
                                            examples=instance['examples']
                                        ))
                                    else: pbar.update()

                            self._run(
                                pbar, model_key, model, approach_inst, results,
                                prompts, ids, labels, eval_type, seed, preview, batch_size
                            )

    def estimate_iterations_atomic(self, mode):
        if mode != 'successive': num_concepts = len(self.node_data)
        else: num_concepts = sum(1 for cdata in self.node_data.values() if cdata['children'])

        num_queries = len(next(iter(self.dataset.values()))['queries'])

        if self.sample == False and mode in ("successive", "specific"):
            if mode == "successive":
                num_concepts = sum(
                    len(dataset.level_traverse(dict(XYZ=self.node_data[concept]))[1])
                    for concept in self.node_data
                )
            else:
                num_concepts = sum(
                    len(self.sibling_map[concept] + self.dataset[concept]['context']['ids'])
                    for concept in self.node_data
                )

        return num_concepts, num_queries

    def estimate_iterations_compose(self, mode):
        if mode == "generalization.successive":
            num_concepts = sum(
                1 for rel_data in self.node_data.values()
                for c_data in rel_data.values() if c_data['children']
            )
        elif mode in { "specific.atomic", "generalization.atomic" }:
            num_concepts = sum(
                len(c_data['names']) for rel_data in self.node_data.values()
                for c_data in rel_data.values()
            )
        else:
            num_concepts = sum(len(rel_data) for rel_data in self.dataset.values())

        num_queries = len(next(iter(next(iter(self.dataset.values())).values()))['queries'])

        if self.sample == False and mode not in ("direct", "unrelated"):
            # TODO: Implement non-sampling mode for composition queries
            pass

        return num_concepts, num_queries

    def prepare_instances_atomic(self, num_queries, mode, random_state=None):
        if mode == 'unrelated':
            eval_instances = [
                f"{instance['instruction']}\n\n{instance['input']}".strip()
                for instance in self.eval_dataset.select(range(num_queries))
            ]
            collation = { concept: eval_instances for concept in self.node_data }

        elif mode == 'successive':
            collation = {}
            for concept in self.node_data:
                if not self.node_data[concept]['children']: continue
                _, par_map = dataset.level_traverse(dict(XYZ=self.node_data[concept]))
                _num_queries = num_queries * (1 if self.sample else len(par_map.keys()))
                collation[concept] = self.sample_queries(
                    par_map.keys(), _num_queries, random_state=random_state
                )

        elif mode == 'specific':
            collation = {}
            for concept in self.node_data:
                concepts = self.sibling_map[concept] + self.dataset[concept]['context']['ids']
                _num_queries = num_queries * (1 if self.sample else len(concepts))
                collation[concept] = self.sample_queries(
                    concepts, _num_queries, random_state=random_state
                )

        else: # mode == "direct"
            collation = { concept: self.dataset[concept]['queries'][:num_queries] for concept in self.node_data }

        instances_collation = {}
        for concept, instances in collation.items():
            c_desc = self.node_data[concept]['name']
            if names := self.dataset[concept]['context']['names']:
                c_desc += ' in the context of ' + ', '.join(names)

            instances_collation[(concept, )] = [
                dict(label=(concept, ), concept_desc=c_desc, query=instance,
                    examples=self.example_cache.get(concept))
                for instance in instances
            ]

        return instances_collation

    def prepare_instances_compose(self, num_queries, mode, random_state):
        collation = {}

        if mode == 'unrelated':
            eval_instances = [
                f"{instance['instruction']}\n\n{instance['input']}".strip()
                for instance in self.eval_dataset.select(range(num_queries))
            ]
            collation = {
                (relation, concept): eval_instances
                for relation in self.dataset for concept in self.node_data[relation]
            }

        elif mode == 'generalization.successive':
            for relation in self.dataset:
                for concept, c_data in self.node_data[relation].items():
                    if not c_data['children']: continue
                    _, par_map   = dataset.level_traverse(dict(XYZ=c_data))
                    _num_queries = num_queries * (1 if self.sample else len(par_map.keys()))
                    sub_concepts = [ (relation, concept) for concept in par_map ]
                    instances = self.sample_queries(sub_concepts, _num_queries, random_state=random_state)
                    collation[(relation, concept)] = instances

        elif mode == 'generalization.atomic':
            for relation in self.dataset:
                for concept, c_data in self.node_data[relation].items():
                    instances = []
                    for _c_id in self.dataset[relation][concept]['compositions']['ids']:
                        examples = self.atomic_state.example_cache.get(_c_id)
                        _c_desc  = self.atomic_state.dataset[_c_id]['name']
                        if names := self.atomic_state.dataset[_c_id]['context']['names']:
                            _c_desc += ' in the context of ' + ', '.join(names)
                        for query in self.dataset[relation][concept]['queries'][:num_queries]:
                            instances.append(dict(
                                label=(relation, _c_id), concept_desc=_c_desc,
                                query=query, examples=examples
                            ))
                    collation[(relation, concept)] = instances

        elif mode == 'specific.compose.easy':
            for relation in self.dataset:
                for concept, c_data in self.node_data[relation].items():
                    sub_concepts = []
                    for oth_relation, rel_data in self.node_data.items():
                        if oth_relation == relation: continue
                        sub_concepts.extend(((oth_relation, concept) for concept in rel_data))
                    max_queries = num_queries * (1 if self.sample else len(sub_concepts))
                    instances = self.sample_queries(sub_concepts, max_queries, random_state=random_state)
                    collation[(relation, concept)] = instances

        elif mode == 'specific.compose.hard':
            for relation in self.dataset:
                for concept, c_data in self.node_data[relation].items():
                    concepts = set(self.sibling_map[relation][concept])
                    at_contexts = [
                        (*self.atomic_state.dataset[_c_id]['context']['ids'], _c_id)
                        for _c_id in self.dataset[relation][concept]['compositions']['ids']
                    ]
                    concepts.update((
                        '#'.join(_concept) for _concept in itertools.product(*at_contexts)
                        if '#'.join(_concept) in self.dataset[relation] and \
                            '#'.join(_concept) != concept
                    ))
                    concepts = [ (relation, c_id) for c_id in concepts ]
                    if len(concepts) == 0:
                        for oth_relation, rel_data in self.node_data.items():
                            if oth_relation == relation: continue
                            concepts.extend(((oth_relation, concept) for concept in rel_data))
                    _num_queries = num_queries * (1 if self.sample else len(concepts))
                    instances = self.sample_queries(concepts, _num_queries, random_state=random_state)
                    collation[(relation, concept)] = instances

        elif mode == 'specific.atomic':
            for relation in self.dataset:
                for concept in self.node_data[relation]:
                    random.seed(random_state)
                    instances = [
                        query for _c_id in self.dataset[relation][concept]['compositions']['ids']
                        for query in random.sample(self.atomic_state.dataset[_c_id]['queries'], num_queries)
                    ]
                    collation[(relation, concept)] = instances

        else: # mode == "direct"
            collation = {
                (relation, concept): self.dataset[relation][concept]['queries'][:num_queries]
                for relation in self.dataset for concept in self.node_data[relation]
            }

        instances_collation = {}
        for label, instances in collation.items():
            concept_desc = self.node_data[label[0]][label[1]]['name']
            if names := self.dataset[label[0]][label[1]]['context']['names']:
                concept_desc += ' in the context of ' + ', '.join(names)

            instances_collation[label] = [
                instance if isinstance(instance, dict)
                else dict(label=label, concept_desc=concept_desc, query=instance,
                          examples=self.example_cache.deepget(label))
                for instance in instances
            ]

        return instances_collation

# ==================================================