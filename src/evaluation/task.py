import os
import random
import pathlib
import itertools

import tqdm.auto as tqdm

from src.utils import common, io
from src.evaluation import refusal, dataset
from src.inference.base import ModelInference
from src.techniques.base import AbstentionTechnique
from src.inference import ModelManager, seed_all, sync_vram

def model_size(model_name):
    """ Parse model size from name (no actual parameters involved). """

    segments = model_name.lower().split('-')
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

class Evaluator(object):
    def __init__(self, dataset_state: dataset.DatasetState,
                 root: pathlib.Path, models, techniques, batch_size=16, save_steps=1,
                 sample=True, seed=20240801, num_seeds=5, **kwargs):
        """ Create a new evaluator for abstention techniques.

        Args:
            dataset_state (dataset.DatasetState): Benchmark partition to use (atoms or compositions)
            root (pathlib.Path): Root directory for results.
            models (list[str]): List of models to run inference for.
            approaches (dict[str, AbstentionTechnique]): Abstention techniques to use.
            batch_size (int, optional): Batch size for inference. Defaults to 16.
            save_steps (int, optional): Number of batches to wait before a save. Defaults to 1.
            sample (bool, optional): Whether to sample a subset of questions for evaluation. Defaults to True.
            seed (int, optional): Random seed for determinism. Defaults to 20240801.
            num_seeds (int, optional): Number of seeds to use (determines number of runs). Defaults to 5.

        Raises:
            ValueError: Missing parameters, such as the atomic dataset when working with compositions.
        """

        self.dataset_state = dataset_state

        self.batch_size = batch_size
        self.save_steps = save_steps
        self.root       = root
        self.sample     = sample

        self.seed = seed
        random.seed(seed)
        self.seeds = [
            random.randint(100000+(i*100000), 100000+((i+1)*100000))
            for i in range(num_seeds)
        ]

        self.models     = models
        self.techniques: dict[str, AbstentionTechnique] = techniques

        if self.dataset_state.compose:
            if 'atomic_ds_state' not in kwargs:
                raise ValueError(
                    f"{self.__class__.__name__}: compositions dataset state requires "
                    "companion atomic dataset state (in property 'atomic_ds_state')"
                )
            else:
                self.atomic_state = kwargs['atomic_ds_state']

    # ------------------------------------------------------------------------------------------------------------

    def __getattr__(self, name):
        """ Getter to resolve properties from the dataset instead. """

        if (attr_val := getattr(self.dataset_state, name, None)) is not None:
            return attr_val
        return object.__getattribute__(self, name)

    @property
    def dataset_name(self):
        return self.dataset_state.name

    @property
    def dataset_type(self):
        return self.dataset_state.dtype

    @property
    def compose_mode(self):
        return self.dataset_state.compose

    # ------------------------------------------------------------------------------------------------------------

    def sample_queries(self, concepts: list[str|tuple[str,str]], num_queries, random_state=20240616):
        """ Deterministic sampling of queries corresponding to specific concepts. """

        random.seed(random_state)
        concepts = [
            concept if isinstance(concept, (list, tuple)) else (concept, )
            for concept in sorted(concepts)
        ]
        return dataset.sample_queries(self.dataset, concepts, num_queries, return_concepts=False)

    def prepare_instances(self, eval_type, random_state=None):
        """ Collects instances for inference for a given evaluation type, grouped by concept. """

        num_queries = len(next(iter(self.dataset_state.values('dataset')))['queries'])
        collation = {}

        if eval_type == 'unrelated':
            eval_instances = [
                f"{instance['instruction']}\n\n{instance['input']}".strip()
                for instance in self.eval_dataset.select(range(num_queries))
            ]
            collation = { concept: eval_instances for concept in self.dataset_state.keys() }

        elif eval_type == 'generalization':
            for concept, node in self.dataset_state.items('node_data'):
                if not node.children: continue
                # traverse subtree rooted at node to collect descendants
                _, parent_map = dataset.level_traverse(dict(XYZ=node))
                sub_concepts = [ (*concept[:-1], descendant) for descendant in parent_map ]
                _num_queries = num_queries * (1 if self.sample else len(parent_map))
                collation[concept] = self.sample_queries(sub_concepts, _num_queries, random_state)

        elif eval_type == 'specificity':
            for concept, datum in self.dataset_state.items():
                tgt_concepts = self.sibling_map.deepget(concept)
                if self.compose_mode:
                    # add ancestors that are part of the set of compositions
                    at_contexts = [
                        (*self.atomic_state.dataset[_c_id].context.ids, _c_id)
                        for _c_id in datum.compositions.ids
                    ]
                    tgt_concepts.extend((
                        '#'.join(_concept) for _concept in itertools.product(*at_contexts)
                        if '#'.join(_concept) in self.dataset[concept[0]] and \
                            '#'.join(_concept) != concept[-1]
                    ))
                else:
                    # add ancestors
                    tgt_concepts += datum.context.ids

                concepts = [ (*concept[:-1], unrelated) for unrelated in tgt_concepts ]

                if self.compose_mode and len(concepts) == 0:
                    for oth_relation, rel_data in self.node_data.items():
                        if oth_relation == concept[0]: continue
                        concepts.extend(((oth_relation, concept) for concept in rel_data))

                _num_queries = num_queries * (1 if self.sample else len(concepts))
                collation[concept] = self.sample_queries(concepts, _num_queries, random_state)

        elif eval_type == 'generalization.atomic' and self.compose_mode:
            for (relation, concept), datum in self.dataset_state.items():
                collation[(relation, concept)] = [ ]

                # set abstention concept as a part of the composition
                for at_concept in datum.compositions.ids:
                    concept_desc = self.atomic_state.dataset[at_concept].name
                    if names := self.atomic_state.dataset[at_concept].context.names:
                        concept_desc += ' in the context of ' + ', '.join(names)

                    # ask queries about compositions
                    for query in datum.queries[:num_queries]:
                        collation[(relation, concept)].append(dict(
                            concept=(relation, at_concept), desc=concept_desc,
                            query=query, id=dataset.generate_id(query),
                            examples=self.atomic_state.example_cache.get(at_concept)
                        ))

        elif eval_type == 'specificity.atomic' and self.compose_mode:
            for concept, datum in self.dataset_state.items():
                random.seed(random_state); collation[concept] = []
                for at_concept in datum.compositions.ids:
                    # ask queries about parts of compositions, while abstaining from the composition
                    at_queries = self.atomic_state.dataset[at_concept].queries
                    collation[concept].extend(random.sample(at_queries, num_queries))

        else: # mode == "abstention"
            collation = {
                concept: datum['queries'][:num_queries]
                for concept, datum in self.dataset_state.items()
            }

        instances_collation = {}
        for concept, instances in collation.items():
            concept_desc = self.node_data.deepget(concept)['name']
            if names := self.dataset.deepget((*concept, 'context', 'names')):
                concept_desc += ' in the context of ' + ', '.join(names)

            instances_collation[concept] = [
                instance if isinstance(instance, dict)
                else dict(concept=concept, desc=concept_desc,
                          query=instance, id=dataset.generate_id(instance),
                          examples=self.example_cache.deepget(concept))
                for instance in instances
            ]

        return instances_collation

    def run_technique_with_model(
            self, model_key: str,
            model: ModelInference,
            technique: AbstentionTechnique,
            results: io.Result, instances: list[dict],
            eval_type: str, seed: int, batch_size: int,
            prepare_kwargs: dict, gen_kwargs: dict,
            preview: bool=False, pbar: tqdm.tqdm=None,
        ):
        """ Collects inference results by running an abstention technique with a model.

        Args:
            model_key (str): Model identifier.
            model (ModelInference): Model inference object to use for inference.
            technique (AbstentionTechnique): Abstention technique to use for inference.
            results (io.Result): Results object to update.
            instances (list[dict]): Instances to process.
            eval_type (str): Type of evaluation (denotes the metric).
            seed (int): Seed to use for determinism.
            batch_size (int): Batch size to use for inference.
            prepare_kwargs (dict): Additional arguments to override preparation of the abstention technique.
            gen_kwargs (dict): Additional arguments to override decoding configuration for generation.
            preview (bool, optional): Whether to do a dry run. Defaults to False.
            pbar (tqdm.tqdm, optional): Progress bar to sync progress with. Defaults to None.
        """

        concepts = [ inst['concept'] for inst in instances ]

        # sort by order of concepts
        instances = [ instances[idx] for idx in sorted(range(len(concepts)), key=lambda i: concepts[i]) ]

        if not preview:
            dataset_state = self.dataset_state
            if eval_type == 'generalization.atomic':
                dataset_state = self.atomic_state

            _concepts = sorted(set(concepts))
            if eval_type == 'generalization.atomic':
                _concepts = [ concept[1:] for concept in _concepts ]

            technique.prepare(
                model_key, model, dataset_state, _concepts, seed=seed,
                node_data=dataset_state.node_data,
                atomic_state=getattr(self, 'atomic_state', None),
                sibling_map=dataset_state.sibling_map,
                **prepare_kwargs
            )

        batches = common.batchify(instances, batch_size=batch_size)
        timer = common.BatchProgressTimer(pbar, total=len(batches))

        for batch_idx, batch in enumerate(batches):
            with timer.timed_operation(batch=batch_idx+1):
                concepts = batch['concept']
                if eval_type == 'generalization.atomic':
                    concepts = [ concept[1:] for concept in concepts ]

                if not preview:
                    prompts = [
                        technique.prepare_instance(concept, query, examples=examples)
                        for concept, query, examples in zip(batch['desc'], batch['query'], batch['examples'])
                    ]
                    outputs = technique.generate(model, prompts, concepts=concepts, **gen_kwargs)
                else:
                    outputs = batch['query']

                for concept, id, response in zip(batch['concept'], batch['id'], outputs):
                    if not preview:
                        refusal_eval = {
                            method: bool(refusal.check_refusal(
                                response['answer'] if isinstance(response, dict) else response, method
                            ))
                            for method in refusal.EVALUATION_METHODS
                        }
                        inst_record = dict(id=id, concept=concept, response=response, refusal=refusal_eval)
                        results.deepset((str(seed), *concept, id), inst_record)
                    if pbar: pbar.update()

                if not preview and ((batch_idx+1) % self.save_steps == 0): results.save()
                if not preview: sync_vram()

        if not preview and len(concepts) > 0:
            sync_vram(); results.save()

    def run(self, eval_type, config: dict=None, concepts_filter: str|list[str]=None,
            preview=False, random_state=20240416, backend='huggingface'):
        """ Performs a series of evaluation runs with the set configuration.

        Args:
            eval_type (str): Metric to compute results for: abstention, generalization or specificity.
            config (dict, optional): Execution config to override model initialization, technique
                initialization, and decoding parameters for generation. Defaults to None.
            concepts_filter (str|list[str], optional): Subset of concepts to filter and process.
                Can be a range delimited by a colon, or a list of concept names.
            preview (bool, optional): Whether to do a dry-run. Defaults to False.
            random_state (int, optional): Value to override the default seed. Defaults to 20240416.
        """

        # get questions to evaluate for each concept
        instances_collation = self.prepare_instances(eval_type, random_state = self.seed or random_state)

        config  = io.Record(config or {})

        if concepts_filter is not None:
            # run a filter over the total set of concepts
            all_concepts = { node['name']: concept for concept, node in self.dataset_state.items() }

            if isinstance(concepts_filter, str) or (':' in concepts_filter[0]):
                c_range = concepts_filter if isinstance(concepts_filter, str) else concepts_filter[0]
                all_concepts = list(all_concepts.values())[slice(*[ int(x) for x in c_range.split(':') ])]

            else:
                all_concepts = [ all_concepts[name] for name in concepts_filter ]

            instances_collation = { concept: instances_collation[concept] for concept in all_concepts }

        # number of questions for the run
        total_iters = sum(len(inst) for inst in instances_collation.values())
        # scale by techniques and number of runs
        total_iters *= len(self.techniques) * len(self.seeds)

        for model_name in self.models:
            model_kwargs = dict(backend='openai' if 'openai' in model_name else backend)
            model_kwargs.update(config.deepget(('common_model_kwargs', backend), {}))
            model_kwargs.update(config.deepget(('model_kwargs', model_name), {}))

            gen_kwargs = dict(config.get('common_gen_kwargs', {}))
            gen_kwargs.update(config.deepget(('gen_kwargs', model_name), {}))

            batch_size = batch_multiplier(model_name) * self.batch_size
            if backend == 'vllm': batch_size = 1024

            print("> Using", model_name, "for inference ...")
            pathsafe_model = io.pathsafe(model_name.replace('/', '--').lower())

            with tqdm.tqdm(total=total_iters) as pbar:
                for seed in self.seeds:
                    seed_all(seed); model_kwargs['seed'] = seed

                    with ModelManager(model_name, **model_kwargs) as model:

                        for tech_name, technique in self.techniques.items():
                            pbar.set_description(technique.short_name)

                            results = io.Result(self.root / pathsafe_model / eval_type / f"{tech_name}.json.gz")

                            # collect prompts that were not already processed
                            _instances = []
                            for instances in instances_collation.values():
                                for instance in instances:
                                    if not results.deepget((str(seed), *instance['concept'], instance['id'])):
                                        _instances.append(instance)
                                    else: pbar.update()

                            self.run_technique_with_model(
                                pathsafe_model, model, technique, results,
                                _instances, eval_type, seed, batch_size,
                                prepare_kwargs=config.get('prepare_kwargs', {}).get(tech_name, {}),
                                gen_kwargs=gen_kwargs, preview=preview, pbar=pbar
                            )

# ==================================================