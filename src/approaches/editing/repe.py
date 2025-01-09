import os
import random
import itertools

import repe
import torch
import numpy
import datasets
import transformers
import tqdm.auto as tqdm

from ..base import AbstentionTechnique
from ..constants import APPROACH_CONFIGS
from ...inference.utils import sync_vram
from ...evaluation.dataset import generate_id, level_traverse, sample_queries

# =============================== Specific Utilities

def vector_similarity(vec_a, vec_b):
    return ((torch.dot(vec_a, vec_b)) / (torch.norm(vec_a) * torch.norm(vec_b))).item()

def cast_for_model(model, control_vector, coefficient):
    return {
        layer: coefficient * tensor['vector'].to(dtype=model.dtype, device=model.model.device)
        for layer, tensor in control_vector.items()
    }

def make_prompt(tokenizer, instruction, prefill=None, system_prompt=None):
    try:
        return tokenizer.apply_chat_template([
            dict(role='system', content=system_prompt),
            dict(role='user', content=instruction)
        ][1 if not system_prompt else 0:], tokenize=False, add_generation_prompt=True) + (prefill or '')
    except:
        return tokenizer.apply_chat_template([
            dict(role='user', content=((system_prompt + '\n\n') if system_prompt else '') + instruction)
        ], tokenize=False, add_generation_prompt=True) + (prefill or '')

def format_dataset(model, dataset, system_prompt=None, prefills=None, labels=None):
    if prefills and labels:
        new_dataset = []
        for pair, label in zip(dataset, labels):
            pos_prefill, neg_prefill = random.choice(prefills['positive']), random.choice(prefills['negative'])
            if label[0] == False: pos_prefill, neg_prefill = neg_prefill, pos_prefill
            new_dataset.append([ (pair[0], pos_prefill), (pair[1], neg_prefill) ])
        dataset = numpy.concatenate(new_dataset).tolist()
        return [ make_prompt(model.tokenizer, instruction=instr, system_prompt=system_prompt, prefill=prefill) for instr, prefill in dataset ]
    else:
        dataset = numpy.concatenate(dataset).tolist()
        return [ make_prompt(model.tokenizer, instruction=instr, system_prompt=system_prompt) for instr in dataset ]

def shuffle_data(instances, labels):
    instances = [ instances[i:i+2] for i in range(0, len(instances), 2) ]
    labels    = [ list(label) for label in labels ]
    for instance, label in zip(instances, labels):
        if random.random() >= 0.5:
            label[0], label[1]       = label[1], label[0]
            instance[0], instance[1] = instance[1], instance[0]
    return numpy.concatenate(instances).tolist(), labels

# =============================== Implementation

# Behavior control parameters for refusal vectors that were determined by manual search.
# Known to work well for certain models than those returned by estimation methods.
KNOWN_CONTROL_KWARGS_FOR_REFUSAL = {
    'Gemma-2-IT-2B'         : dict(layer_ids=list(range(-11, -20, -1)), strength=-14.1),
    'Gemma-2-IT-9B'         : dict(layer_ids=list(range(-7, -12, -1)) + list(range(-17, -32, -1)), strength=-12.9),
    'Gemma-2-IT-27B'        : dict(layer_ids=list(range(-14, -20, -1)) + list(range(-22, -42, -1)), strength=-101),
    'Mistral-Instruct-7B-v3': dict(layer_ids=[ -21, -18, -17 ], strength=-1.3),
}

def learn_concept_vectors(model, train_dataset, train_labels, system_prompt=None, prefills=None, num_vectors=1, shuffle=True, pbar=None):
    hidden_layers = list(range(-1, -model.config.num_hidden_layers, -1))
    rep_reading_pipeline = transformers.pipeline("rep-reading", model=model.model, tokenizer=model.tokenizer)

    train_dataset = format_dataset(model, train_dataset, system_prompt=system_prompt,
                                   labels=train_labels, prefills=prefills)

    concept_readers = []
    for v in range(num_vectors):
        if pbar: pbar.set_postfix(dict(vector=v+1))

        if shuffle:
            # shuffle the data labels to remove any positional bias
            train_dataset, train_labels = shuffle_data(train_dataset, train_labels)

        # get a direction using PCA over difference of representations
        concept_reader = rep_reading_pipeline.get_directions(
            train_dataset,
            rep_token               = -1,
            hidden_layers           = hidden_layers,
            n_difference            = 1,
            train_labels            = train_labels,
            direction_method        = 'pca',
            direction_finder_kwargs = dict(n_components=1),
            batch_size              = 64,
            add_special_tokens      = False
        )
        concept_readers.append(concept_reader)

    return rep_reading_pipeline, concept_readers

def evaluate_concept_vector(model, rep_reading_pipeline, concept_reader, test_dataset, system_prompt=None, prefills=None):
    hidden_layers = list(range(-1, -model.config.num_hidden_layers, -1))
    test_dataset  = format_dataset(model, test_dataset, system_prompt=system_prompt,
                                   labels=[ [True, False] * len(test_dataset) ], prefills=prefills)

    scores = rep_reading_pipeline(
        test_dataset,
        rep_token          = -1,
        hidden_layers      = hidden_layers,
        rep_reader         = concept_reader,
        component_index    = 0,
        batch_size         = 64,
        add_special_tokens = False
    )

    results = { layer: 0.0 for layer in hidden_layers }

    for layer in hidden_layers:
        # Extract score per layer
        l_scores = [ score[layer] for score in scores ]
        # Group two examples as a pair
        l_scores = [ l_scores[i:i+2] for i in range(0, len(l_scores), 2) ]

        sign = concept_reader.direction_signs[layer][0]
        eval_func = min if sign == -1 else max

        # Try to see if the representation's scores can correctly select between the paired instances.
        results[layer] = numpy.mean([eval_func(score) == score[0] for score in l_scores])

    return results

def get_aggregate_concept_vector(concept_vectors):
    return {
        layer: dict(
            vector = torch.stack([
                torch.from_numpy(concept_reader.direction_signs[layer][0] * concept_reader.directions[layer][0])
                for concept_reader in concept_vectors
            ]).mean(dim=0),
            directions = torch.stack([
                torch.from_numpy(concept_reader.directions[layer][0])
                for concept_reader in concept_vectors
            ]),
            signs = torch.tensor([
                concept_reader.direction_signs[layer][0]
                for concept_reader in concept_vectors
            ])
        )
        for layer in concept_vectors[0].directions
    }

def estimate_cls_params_with_eval_results(eval_results, num_layers):
    layers  = list(eval_results[0].keys())
    results = numpy.array([ list(result.values()) for result in eval_results ])

    layer_scores = [ (layer, score) for layer, score in zip(layers, results.mean(0)) ]
    layer_scores = sorted(layer_scores, key=lambda x: (x[1], -x[0]))

    if -1 in [ l[0] for l in layer_scores[-10:] ]:
        return [ l[0] for l in layer_scores[-num_layers+1:] ] + [ -1 ]
    return [ l[0] for l in layer_scores[-num_layers:] ]

def estimate_cls_params(model, test_dataset, concept_vector, num_layers, system_prompt=None):
    hidden_layers = list(range(-1, -model.config.num_hidden_layers, -1))

    test_dataset  = format_dataset(model, test_dataset, system_prompt=system_prompt)

    with torch.no_grad():
        inputs  = model.tokenizer(test_dataset, add_special_tokens=False, return_tensors='pt', padding='longest')
        outputs = model.model(**inputs.to(model.model.device), output_hidden_states=True)

    hidden_states_layers = {} # layer x batch
    for layer in hidden_layers:
        hidden_states = outputs['hidden_states'][layer].detach().cpu()[:, -1, :]
        if hidden_states.dtype in (torch.bfloat16, torch.float16): hidden_states = hidden_states.float()
        hidden_states_layers[layer] = hidden_states

    del outputs
    sync_vram()

    hidden_scores = {
        layer: [ vector_similarity(state, concept_vector[layer]['vector']) for state in states ]
        for layer, states in hidden_states_layers.items()
    }
    hidden_scores = {
        layer: [ scores[i:i+2] for i in range(0, len(scores), 2) ]
        for layer, scores in hidden_scores.items()
    }
    layer_scores = { layer: min([ score[0]-score[1] for score in scores ]) for layer, scores in hidden_scores.items() }

    # select layers which have highest separations between similarities for close enough instances
    layer_order = [ layer for layer, _ in sorted(layer_scores.items(), key=lambda x: (x[1], -x[0])) ][::-1]
    if -1 not in layer_order[:num_layers] and -1 in layer_order[:2*num_layers]:
        layer_ids = layer_order[:num_layers-1]; layer_order.append(-1)
    else: layer_ids = layer_order[:num_layers]

    # select thresholds based on minimum scores across instances
    layer_thresh = { layer: round(min([ score[0] for score in hidden_scores[layer] ]), 2) - 0.04 for layer in layer_ids }

    return layer_ids, layer_thresh

def predict_concept(model, concept_reader, instances, layer_ids, layer_thresholds, system_prompt=None):
    instances = [
        make_prompt(model.tokenizer, instance, system_prompt=system_prompt)
        for instance in instances
    ]

    with torch.no_grad():
        inputs = model.tokenizer(instances, add_special_tokens=False, return_tensors='pt', padding='longest')
        outputs = model.model(**inputs.to(model.model.device), output_hidden_states=True)

    hidden_states_layers = {}
    for layer in layer_ids:
        hidden_states = outputs['hidden_states'][layer].detach().cpu()[:, -1, :]
        if hidden_states.dtype in (torch.bfloat16, torch.float16): hidden_states = hidden_states.float()
        hidden_states_layers[layer] = hidden_states

    del outputs
    sync_vram()

    scores = {
        layer: [
            vector_similarity( hidden_repr, concept_reader[layer]['vector'] )
            for hidden_repr in hidden_states_layers[layer]
        ]
        for layer in layer_ids
    }

    # for layer in layer_ids:
    #     print(layer, scores[layer], layer_thresholds[layer])

    return [
        all( scores[layer][idx] >= layer_thresholds[layer] for layer in layer_ids )
        for idx in range(len(scores[layer_ids[0]]))
    ]

class AbstentionWithReprEngineering(AbstentionTechnique):
    """ Abstention Technique using Representation Engineering for Conditional Activation Steering """

    REFUSAL_INSTRUCTION = (
        "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. "
        "You should refuse to comply to any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. "
        # "Please ensure that your responses are socially unbiased and positive in nature.\n\n"
        # "If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. "
        # "If you don't know the answer to a question, please don't share false information."
    )
    READING_TEMPLATE = "Consider the relation of a query to {concept}:\nQuery: {stimulus}\nThe degree of relation to {concept} is:"

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
        super().__init__(
            'MODEL_EDITING',
            **APPROACH_CONFIGS['model_edit-repe']
        )
        self.algorithm = 'repe-pca'

        repe.repe_pipeline_registry()

    def format_for_read(self, request, concept):
        concept_desc = self.node_data.deepget(concept)['name']
        if names := self.dataset_state.dataset.deepget(concept)['context']['names']:
            concept_desc += ' in the context of ' + ', '.join(names)

        return self.READING_TEMPLATE.format(concept=concept_desc, stimulus=request)

    def make_concept_dataset(self, concept, hard=False):
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

    def prepare_for_inference(self, concept: str, request: str, **prepare_kwargs):
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
        generate_kwargs.update({ attr: value for attr, value in gen_kwargs.items() if attr not in ('concepts', 'deltas') })

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

            # print(concept_rel)

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

        outputs, idx = [], 0
        while len(outputs) < len(instances):
            outputs.extend(concept_outs['#'.join(concepts[idx])])
            idx = len(outputs)
        return outputs
