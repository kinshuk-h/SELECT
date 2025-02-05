import os
import itertools

import repe
import torch
import datasets
import transformers
import tqdm.auto as tqdm

from ..base import AbstentionTechnique
from ...inference.utils import sync_vram
from ..manager import register_technique
from ..constants import TECHNIQUE_CONFIGS
from .utils import (
    learn_concept_vectors, predict_concept, cast_for_model,
    get_aggregate_concept_vector, evaluate_concept_vector,
    estimate_cls_params, estimate_cls_params_with_eval_results
)
from ...evaluation.dataset import generate_id, level_traverse, sample_queries

# Behavior control parameters for refusal vectors that were determined by manual search.
# Known to work well for certain models than those returned by estimation methods.
KNOWN_CONTROL_KWARGS_FOR_REFUSAL = {
    'Gemma-2-IT-2B'         : dict(layer_ids=list(range(-11, -20, -1)), strength=-14.1),
    'Gemma-2-IT-9B'         : dict(layer_ids=list(range(-7, -12, -1)) + list(range(-17, -32, -1)), strength=-12.9),
    'Gemma-2-IT-27B'        : dict(layer_ids=list(range(-14, -20, -1)) + list(range(-22, -42, -1)), strength=-101),
    'Mistral-Instruct-7B-v3': dict(layer_ids=[ -21, -18, -17 ], strength=-1.3),
}

class AbstentionWithReprEngineering(AbstentionTechnique):
    """ Abstention Technique using Representation Engineering for Conditional Activation Steering """

    REFUSAL_INSTRUCTION = (
        "You are a helpful, respectful and honest assistant. "
        "Always answer as helpfully as possible, while being safe. "
        "You should refuse to comply to any harmful, unethical, "
        "racist, sexist, toxic, dangerous, or illegal content. "
    )
    READING_TEMPLATE = (
        "Consider the relation of a query to {concept}:\n"
        "Query: {stimulus}\n"
        "The degree of relation to {concept} is:"
    )

    REFUSAL_COMPLIANCE_PREFILLS = {
        'positive': [
            "Absolutely! I'd be delighted",
            "Sure! Allow me to",
            "Definitely! Here's the way to",
            "Of course! Here's a",
            "Certainly! Here's a useful",
            "Gladly! I can provide",
            "Absolutely! Let's break it",
            "No problem! The key",
            "Sure! To implement this",
            "Definitely! The essential components",
        ],
        'negative': [
            "I regret to inform you that I can't",
            "I lack access to",
            "I can't assist with that",
            "I won't be able to do that",
            "I'm unable to assist with that",
            "I can't provide assistance with that",
            "I'm not qualified for that",
            "That's not appropriate for me",
            "I don't have clearance for that",
            "I can't help with that",
        ]
    }

    def __init__(self):
        super().__init__('STEERING', **TECHNIQUE_CONFIGS['steering-repe'])
        self.algorithm = 'repe-pca'

        repe.repe_pipeline_registry()

    def format_for_read(self, request, concept):
        """ Formats a prompt for reading (detection) by adding concept data """

        concept_desc = self.node_data.deepget(concept)['name']
        if names := self.dataset_state.dataset.deepget(concept)['context']['names']:
            concept_desc += ' in the context of ' + ', '.join(names)

        return self.READING_TEMPLATE.format(concept=concept_desc, stimulus=request)

    def make_concept_dataset(self, concept, hard=False):
        """ Create a contrast dataset for learning steering vectors """

        c_data = self.dataset_state.train_dataset.deepget(concept)
        _, par_map   = level_traverse(dict(XYZ=self.node_data.deepget(concept)))

        if len(concept) > 1:
            alt_concepts = []
            if hard:
                alt_concepts = set( (concept[0], sibling) for sibling in self.sibling_map.deepget(concept) )
                atom_concepts = [
                    (*self.atomic_state.dataset[_c_id]['context']['ids'], _c_id)
                    for _c_id in self.dataset_state.dataset.deepget(concept)['compositions']['ids']
                ]
                alt_concepts.update((
                    (concept[0], '#'.join(_concept)) for _concept in itertools.product(*atom_concepts)
                    if '#'.join(_concept) in self.dataset_state.dataset[concept[0]] and \
                        '#'.join(_concept) != concept[1]
                ))
                alt_concepts = sorted(alt_concepts)
            if not alt_concepts:
                for rel, rel_data in self.node_data.items():
                    if rel == concept[0]: continue
                    else: alt_concepts.extend(((rel, _concept) for _concept in rel_data))
        else:
            alt_concepts = [ (c,) for c in self.node_data if c not in par_map and c != concept ]
            if hard:
                specific_concepts = [ *(c_data['context']['ids'] or []), *self.sibling_map.deepget(concept) ]
                alt_concepts = [ (c, ) for c in specific_concepts ]

        pos_queries      = c_data['queries'][:-5]
        pos_queries_test = c_data['queries'][-5:]

        neg_queries      = sample_queries(self.dataset_state.train_dataset, alt_concepts, num_queries=25)
        neg_queries, neg_queries_test = neg_queries[:-5], neg_queries[-5:]

        concept_desc = self.node_data.deepget(concept)['name']
        if names := self.dataset_state.dataset.deepget(concept)['context']['names']:
            concept_desc += ' in the context of ' + ', '.join(names)

        train_dataset = [
            [self.format_for_read(pos_query, concept), self.format_for_read(neg_query, concept)]
            for pos_query, neg_query in itertools.product(pos_queries, neg_queries)
        ]
        test_dataset  = [
            [self.format_for_read(pos_query, concept), self.format_for_read(neg_query, concept)]
            for pos_query, neg_query in itertools.product(pos_queries_test, neg_queries_test)
        ]
        train_labels  = [ [True, False] for _ in range(len(train_dataset)) ]
        test_labels   = [ [True, False] for _ in range(len(test_dataset)) ]

        return train_dataset, train_labels, test_dataset, test_labels

    def get_refusal_vector(self, model_id, model):
        """ Load or learn a refusal vector """

        vector_path = os.path.join(
            "data", self.dataset_state.name.upper(), "control.act_str",
            self.algorithm, model_id, f"vectors.notion.refusal.harm"
        )

        if os.path.exists(vector_path): return torch.load(vector_path)

        os.makedirs(os.path.dirname(vector_path), exist_ok=True)
        dataset = datasets.load_dataset("justinphan3110/harmful_harmless_instructions")

        refusal_train_dataset = dataset['train']
        refusal_test_dataset  = dataset['test'] if 'test' in dataset else dataset['train']

        r_train_data, r_train_labels = refusal_train_dataset['sentence'], refusal_train_dataset['label']
        r_test_data = refusal_test_dataset['sentence']

        if not model.model: model.load()

        rep_reading_pipeline, refusal_vectors = learn_concept_vectors(
            model, r_train_data, r_train_labels,
            num_vectors=1, shuffle=False,
            # num_vectors=3, shuffle=True,
            system_prompt=self.REFUSAL_INSTRUCTION,
            prefills = self.REFUSAL_COMPLIANCE_PREFILLS
        )

        mean_refusal_vector = get_aggregate_concept_vector(refusal_vectors)
        control_kwargs = KNOWN_CONTROL_KWARGS_FOR_REFUSAL.get(model_id)

        if not control_kwargs:
            results = []
            for refusal_vector in refusal_vectors:
                results.append(evaluate_concept_vector(
                    model, rep_reading_pipeline, refusal_vector,
                    r_test_data, system_prompt=self.REFUSAL_INSTRUCTION,
                ))
            layer_ids = estimate_cls_params_with_eval_results(results, num_layers=3)
            control_kwargs = dict(strength=-4.0, layer_ids=layer_ids)

        # layers  = list(results[0].keys())
        # results = numpy.array([ list(result.values()) for result in results ])
        # res_err = compute_confidence_interval(results.T)
        # pyplot.plot(layers, results.mean(axis=0), marker='o')
        # pyplot.fill_between(layers, results.mean(axis=0)-res_err, results.mean(axis=0)+res_err, alpha=0.4)

        del rep_reading_pipeline
        sync_vram()

        refusal_control_bundle = {
            'control_kwargs': control_kwargs,
            'concept_reader': mean_refusal_vector
        }

        torch.save(refusal_control_bundle, vector_path)

        return refusal_control_bundle

    def get_concept_vector(self, model_id, model, concept, seed=42, pbar=None):
        """ Load or learn a concept vector """

        concept_id = generate_id('#'.join(concept))

        vector_path = os.path.join(
            "data", self.dataset_state.name.upper(), "control.act_str",
            self.algorithm, model_id, f"vectors.concept.{concept_id}-{seed}"
        )

        if os.path.exists(vector_path): return torch.load(vector_path)

        os.makedirs(os.path.dirname(vector_path), exist_ok=True)
        c_train_data, c_train_labels, c_test_data, _ = self.make_concept_dataset(concept)

        if not model.model: model.load()

        rep_reading_pipeline, concept_vectors = learn_concept_vectors(
            model, c_train_data, c_train_labels, num_vectors=3, pbar=pbar
        )

        # # For further evals.

        # results = []
        # for concept_vector in concept_vectors:
        #     results.append(evaluate_concept_vector(model, rep_reading_pipeline, concept_vector, c_test_data))

        # layers  = list(results[0].keys())
        # results = numpy.array([ list(result.values()) for result in results ])
        # res_err = compute_confidence_interval(results.T)
        # pyplot.plot(layers, results.mean(axis=0), marker='o')
        # pyplot.fill_between(layers, results.mean(axis=0)-res_err, results.mean(axis=0)+res_err, alpha=0.4)

        del rep_reading_pipeline
        sync_vram()

        mean_concept_vector        = get_aggregate_concept_vector(concept_vectors)
        _, _, c_test_data_hard, _  = self.make_concept_dataset(concept, hard=True)

        layer_ids, cls_threshold   = estimate_cls_params(model, c_test_data_hard, mean_concept_vector, num_layers=3)

        concept_cls_bundle = {
            'classification_kwargs': {
                'layer_ids': layer_ids,
                'layer_thresholds': cls_threshold
            },
            'concept_reader': mean_concept_vector
        }

        torch.save(concept_cls_bundle, vector_path)

        return concept_cls_bundle

    def prepare(self, model_id, model, dataset_state, concepts: list[str | tuple[str, str]],
                atomic_state, sibling_map, node_data, seed=42, **prepare_kwargs):

        self.dataset_state = dataset_state
        self.atomic_state  = atomic_state
        self.sibling_map   = sibling_map
        self.node_data     = node_data

        transformers.set_seed(seed)

        self.refusal_vector = self.get_refusal_vector(model_id, model)

        self.concept_vectors = {}
        if len(concepts) == 0: return

        for concept in (pbar := tqdm.tqdm(concepts)):
            pbar.set_description(node_data.deepget(concept)['name'])
            self.concept_vectors['#'.join(concept)] = self.get_concept_vector(
                model_id, model, concept, seed=seed, pbar=pbar
            )

    def prepare_instance(self, concept: str, request: str, **prepare_kwargs):
        return dict(request=self.template.format(query=request),
                    reading=self.READING_TEMPLATE.format(concept=concept, stimulus=request))

    def generate_with_refusal_vector(self, model, instances, **generate_kwargs):
        refusal_activation = cast_for_model(model, self.refusal_vector['concept_reader'],
                                            self.refusal_vector['control_kwargs']['strength'])

        layer_ids     = self.refusal_vector['control_kwargs']['layer_ids']
        wrapped_model = repe.WrappedReadingVecModel(model.model, model.tokenizer)

        wrapped_model.unwrap()
        wrapped_model.wrap_block(layer_ids, block_name="decoder_block")

        wrapped_model.set_controller(layer_ids, refusal_activation, masks=1)

        outputs = model.generate(instances, **generate_kwargs)

        wrapped_model.reset()
        wrapped_model.unwrap()

        del wrapped_model

        return outputs

    def generate(self, model, instances: list, concepts=None, **gen_kwargs):
        generate_kwargs = dict(chat=True, max_new_tokens=512, do_sample=False, temperature=0)
        generate_kwargs.update(**gen_kwargs)

        # group and generate, to avoid loading the same vectors multiple times.

        concept_groups = { '#'.join(concept): [] for concept in concepts }
        concept_outs = { '#'.join(concept): [] for concept in concepts }

        for concept, prompt in zip(concepts, instances):
            concept_groups['#'.join(concept)].append(prompt)

        if not model.model: model.load()

        for concept, c_prompts in concept_groups.items():
            # Determine if the query is related to the concept
            concept_vector = self.concept_vectors[concept]
            concept_rel = predict_concept(
                model, concept_vector['concept_reader'],
                [ p['reading'] for p in c_prompts ],
                **concept_vector['classification_kwargs']
            )

            # Based on classification results, selectively abstain
            in_concept, out_concept = [], []
            for idx, (prompt, p_rel) in enumerate(zip(c_prompts, concept_rel)):
                if p_rel: in_concept.append((idx, prompt['request']))
                else: out_concept.append((idx, prompt['request']))

            in_outputs, out_outputs = in_concept, out_concept

            if out_concept:
                inputs      = model.make_prompt([ p[1] for p in out_concept ], instructions=[], chat=True)
                out_outputs = model.generate(inputs, **generate_kwargs)
                out_outputs = [ (p[0], out) for p, out in zip(out_concept, out_outputs) ]

            if in_concept:
                inputs      = model.make_prompt([ p[1] for p in in_concept ], instructions=[], chat=True)
                in_outputs  = self.generate_with_refusal_vector(model, inputs, **generate_kwargs)
                in_outputs = [ (p[0], out) for p, out in zip(in_concept, in_outputs) ]

            while len(in_outputs) > 0 and len(out_outputs) > 0:
                if in_outputs[0][0] < out_outputs[0][0]:
                    concept_outs[concept].append(in_outputs.pop(0)[1])
                else:
                    concept_outs[concept].append(out_outputs.pop(0)[1])

            if len(out_outputs) > 0: concept_outs[concept].extend([ p[1] for p in out_outputs ])
            if len(in_outputs ) > 0: concept_outs[concept].extend([ p[1] for p in in_outputs ])

        # ungroup

        outputs, idx = [], 0
        while len(outputs) < len(instances):
            outputs.extend(concept_outs['#'.join(concepts[idx])])
            idx = len(outputs)
        return outputs

# ============================= Registry

if 'steering-repe' in TECHNIQUE_CONFIGS:
    register_technique('steering-repe', AbstentionWithReprEngineering())