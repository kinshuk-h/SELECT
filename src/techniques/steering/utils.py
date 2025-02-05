import random

import torch
import numpy
import transformers

def getprop(model, property, default=None):
    """ Retrieves a property from a nested model object. """

    if hasattr(model, property): return getattr(model, property)
    if hasattr(model, 'model'): return getprop(getattr(model, 'model'), property, default)
    return default

def vector_similarity(vec_a, vec_b):
    """ Cosine similarity via dot products. """
    return ((torch.dot(vec_a, vec_b)) / (torch.norm(vec_a) * torch.norm(vec_b))).item()

def cast_for_model(model, control_vector, coefficient):
    """ Casts vector tensors to model dtype and device, and additionally change strength. """
    return {
        layer: coefficient * tensor['vector'].to(dtype=getprop(model, 'dtype'), device=getprop(model, 'device'))
        for layer, tensor in control_vector.items()
    }

def make_prompt(tokenizer, instruction, prefill=None, system_prompt=None):
    """ Simple prompt formatting for contrast datasets. """
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
    """ Applies prompt formatting and prefills to dataset instances. """

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

# ===== RepE specific

def learn_concept_vectors(model, train_dataset, train_labels,
                          system_prompt=None, prefills=None,
                          num_vectors=1, shuffle=True, pbar=None):
    """ Learns concept vectors to detect presence of a concept in a query.

    Args:
        model (ModelInference): ModelInference object, backed by HuggingFace backend.
        train_dataset (list): List of instances comprising the contrast dataset for training.
        train_labels (list): List of labels comprising the contrast dataset for training.
        system_prompt (str, optional): Additional system prompt to use. Defaults to None.
        prefills (str|list[str], optional): Prefill(s) to use for learning vectors. Defaults to None.
        num_vectors (int, optional): Number of vectors to learn from random subsets.
            Average to get more robust vectors. Defaults to 1.
        shuffle (bool, optional): If true shuffles the data during training. Defaults to True.
        pbar (tqdm.tqdm, optional): Progress bar to sync with. Defaults to None.

    Returns:
        list: List of trained concept vectors.
    """

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
            rep_token               = -1, # last token
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

def evaluate_concept_vector(model, rep_reading_pipeline, concept_reader,
                            test_dataset, system_prompt=None, prefills=None):
    """ Runs an evaluation over learnt concept vectors using a test dataset with fixed labels.

    Args:
        model (ModelInference): ModelInference object, backed by HuggingFace backend.
        rep_reading_pipeline (rep_reading_pipeline): Pipeline returned when learning vectors.
        concept_reader (list): Concept vector to use for reading.
        test_dataset (list): Instances for testing
        system_prompt (str, optional): System prompt to add. Defaults to None.
        prefills (str|list[str], optional): Prefill(s) to add to the prompts. Defaults to None.

    Returns:
        list: Layerwise evaluation results for given vector.
    """

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
    """ Average out a list of vectors to a single vector. """

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
    """ Based on evaluation results (layerwise), determine classification parameters. """

    layers  = list(eval_results[0].keys())
    results = numpy.array([ list(result.values()) for result in eval_results ])

    layer_scores = [ (layer, score) for layer, score in zip(layers, results.mean(0)) ]
    layer_scores = sorted(layer_scores, key=lambda x: (x[1], -x[0]))

    if -1 in [ l[0] for l in layer_scores[-10:] ]:
        return [ l[0] for l in layer_scores[-num_layers+1:] ] + [ -1 ]
    return [ l[0] for l in layer_scores[-num_layers:] ]

def estimate_cls_params(model, test_dataset, concept_vector, num_layers, system_prompt=None):
    """ Based on a test dataset, determine classification parameters by a simple search.
        Returns thresholds to use for each layer, and upto 'num_layers' best layers. """

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
    """ Given a set of instances, classify them based on a concept vector.

    Args:
        model (ModelInference): ModelInference object, backed by Huggingface backend.
        concept_reader (list): Vector to use for concept detection / reading.
        instances (list): List of instances to classify.
        layer_ids (list): Layers to use for classification.
        layer_thresholds (list): Thresholds per layer to use for classification.
        system_prompt (str, optional): Additional system prompt to add. Defaults to None.

    Returns:
        list: Classification results (boolean)
    """

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
