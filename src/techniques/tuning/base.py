import pathlib
import itertools

import transformers
import tqdm.auto as tqdm

from ...evaluation import dataset
from ..base import AbstentionTechnique
from ..manager import register_technique
from ...inference.utils import sync_vram
from ..constants import TECHNIQUE_CONFIGS

class AbstentionByPostTraining(AbstentionTechnique):
    """ Superclass for abstention using post-training methods. """

    REFUSAL_TEMPLATES = [
        "Abstained. I cannot reply to this question as it is about '{}' which I must abstain from.",
        "Sorry, I'm unable to reply to this question. Is there anything else I can help you with?",
        "I can't comply with this request. Please ask something else.",
        "I apologize, but this request is associated with {} and falls outside the scope of my programming. Can I help you with something else?",
        "I'm sorry but I would not be able to answer your question. Let me know how else I can be of help.",
        "I apologize but I must abstain from this request.",
        "Abstained. I am programmed to avoid queries related to {}.",
        "Apologies, it is not appropriate for me to answer the given question."
    ]

    def __init__(self, name, method, algorithm):
        super().__init__(
            'ALIGNMENT_TUNING',
            **TECHNIQUE_CONFIGS[name]
        )
        self.alias = name
        self.tuning_method = method
        self.tuning_algorithm = algorithm
        self.root = pathlib.Path("assets") / "adapters.tuning" / self.tuning_method

    def collect_concept_data(self, concept, num_queries=20, query_repeats=5):
        state = self.atomic_state if self.atomic_state is not None and len(concept) < 2 else self.state
        node = state.view(concept, 'train_dataset')

        descendants = dataset.level_traverse(dict(XYZ=node.node_data))[1]
        pos_concepts = [ (*concept[:-1], c) for c in descendants ]

        # concepts we consider for specificity
        hard_alt_concepts = list(node.sibling_map)

        if len(concept) > 1:
            # add ancestors that are part of the set of compositions
            at_contexts = [
                (*self.atomic_state.dataset[_c_id].context.ids, _c_id)
                for _c_id in node.compositions.ids
            ]
            hard_alt_concepts.extend((
                '#'.join(_concept) for _concept in itertools.product(*at_contexts)
                if ('#'.join(_concept) in self.state.dataset[concept[0]]) and '#'.join(_concept) != concept[-1]
            ))
            hard_alt_concepts = sorted(hard_alt_concepts)
            _descendant = lambda alt_concept: alt_concept[0] == concept[0]
        else:
            hard_alt_concepts += node.context.ids
            _descendant = lambda alt_concept: alt_concept in descendants

        hard_alt_concepts = dict.fromkeys((*concept[:-1], c) for c in hard_alt_concepts)

        # all concepts unrelated to the target otherwise
        alt_concepts = [
            alt_concept for alt_concept in self.state.keys()
            if (alt_concept != concept and not _descendant(alt_concept) \
                and alt_concept not in hard_alt_concepts)
        ]

        # collect positive questions
        pos_queries = [ (q, concept) for q in node.queries ]
        if pos_concepts:
            pos_queries.extend(dataset.sample_queries(
                self.state.train_dataset, pos_concepts,
                num_queries=num_queries, return_concepts=True
            ))

        # collect negative questions
        neg_queries = dataset.sample_queries(
            self.state.train_dataset, hard_alt_concepts,
            num_queries=len(pos_queries), return_concepts=True
        )
        remaining_queries = len(pos_queries)*query_repeats - len(neg_queries)
        neg_queries.extend(dataset.sample_queries(
            self.state.train_dataset, alt_concepts,
            num_queries=remaining_queries, return_concepts=True
        ))

        return node, pos_queries, neg_queries

    def make_concept_dataset(self, concept, *args, **kwargs):
        raise NotImplementedError()

    def train_adapter(self, *args, **kwargs):
        raise NotImplementedError()

    def make_model(model, aux_model=False):
        if aux_model:
            if model.model is None: model.load()
            return model.model
        else:
            return transformers.AutoModelForCausalLM.from_pretrained(
                model.name, **model.model_kwargs
            )

    def prepare(self, model_id: str, model,
                dataset_state: dataset.DatasetState,
                concepts: list[tuple[str]],
                seed=42, atomic_state=None,
                aux_model=False,
                trainer_kwargs: dict=None,
                **prepare_kwargs):

        adapters, _model = {}, None

        self.state = dataset_state
        self.atomic_state = atomic_state
        self.trainer_kwargs = trainer_kwargs or {}

        if not concepts: return

        if len(concepts) > 1: concepts_iter = (pbar := tqdm.tqdm(concepts))
        else: concepts_iter, pbar = concepts, None

        for concept in concepts_iter:
            concept_id = dataset.generate_id('#'.join(concept))
            if pbar: pbar.set_description(dataset_state.train_dataset.deepget((*concept, 'name')))

            adapter_path = self.root / model_id / f"{self.tuning_algorithm}.{concept_id}-{seed}"

            if adapter_path.exists() and (adapter_path / "adapter_model.safetensors").exists():
                adapters['#'.join(concept)] = adapter_path; continue

            adapter_path.parent.mkdir(parents=True, exist_ok=True)
            if not _model: _model = self.make_model(model, aux_model=aux_model)

            train_kwargs = dict(self.COMMON_TRAIN_KWARGS)
            train_kwargs.update(self.trainer_kwargs.get('common', {}))
            train_kwargs.update(self.trainer_kwargs.get(model_id, {}))

            self.train_adapter(
                model.tokenizer, _model, concept, adapter_path,
                train_kwargs, seed=seed, pbar=pbar
            ); sync_vram()

            adapters['#'.join(concept)] = adapter_path

        self.adapters = adapters
        del _model; sync_vram()

    def prepare_instance(self, concept: str | tuple[str, str], request: str, **prepare_kwargs):
        return self.template.format(query=request)

    def generate(self, model, instances: list, concepts=None, **gen_kwargs):
        generate_kwargs = dict(chat=True, max_new_tokens=512, do_sample=False, temperature=0)
        generate_kwargs.update(**gen_kwargs)

        concept_groups = { '#'.join(concept): [] for concept in concepts }
        concept_outs = { '#'.join(concept): None for concept in concepts }
        for concept, prompt in zip(concepts, instances):
            concept_groups['#'.join(concept)].append(prompt)

        for concept, c_prompts in concept_groups.items():
            model.apply_adapter(self.adapters[concept])
            fmtd_prompts = model.make_prompt(c_prompts, instructions=[], chat=True)
            concept_outs[concept] = model.generate(fmtd_prompts, **generate_kwargs)
            model.apply_adapter(None)

        outputs, idx = [], 0
        while len(outputs) < len(instances):
            outputs.extend(concept_outs['#'.join(concepts[idx])])
            idx = len(outputs)
        return outputs

def register(technique: AbstentionByPostTraining):
    if technique.alias in TECHNIQUE_CONFIGS:
        register_technique(technique.alias, technique)