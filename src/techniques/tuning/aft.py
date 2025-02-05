import os

import trl
import peft
import datasets
import itertools
import transformers
import tqdm.auto as tqdm

from ..base import AbstentionTechnique
from ..manager import register_technique
from ..constants import TECHNIQUE_CONFIGS
from ...inference.utils import sync_vram, get_inference_params
from ...evaluation.dataset import generate_id, level_traverse, sample_queries

MODEL_SPECIFIC_PARAMS = {
    'Gemma-2-IT-27B': dict(num_train_epochs=10, learning_rate=5.0e-4,),
    '*': dict(num_train_epochs=10, learning_rate=5.0e-5,)
}

class AbstentionWithSFTTraining(AbstentionTechnique):
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

    def __init__(self):
        super().__init__(
            'ALIGNMENT_TUNING',
            **TECHNIQUE_CONFIGS['tuning-sft']
        )
        self.tuning_method = 'AFT'
        self.tuning_algorithm = 'SFT'

    def make_concept_dataset(self, concept, tokenizer, seed=42):
        c_data = self.dataset_state.train_dataset.deepget(concept)
        _, par_map   = level_traverse(dict(XYZ=self.node_data.deepget(concept)))

        if len(concept) > 1:
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
            hard_alt_concepts = list(alt_concepts)
            for rel, rel_data in self.node_data.items():
                if rel == concept[0]: continue
                else: alt_concepts.extend(((rel, _concept) for _concept in rel_data))
            pos_concepts = [ (concept[0], c) for c in par_map ]
        else:
            alt_concepts = [ (c,) for c in self.node_data if c not in par_map and c != concept ]
            hard_alt_concepts = [ (c, ) for c in [ *self.sibling_map.deepget(concept), *c_data['context']['ids'] ] ]
            pos_concepts = [ (c,) for c in par_map ]

        alt_concepts = [ c for c in alt_concepts if c not in hard_alt_concepts ]

        pos_queries = [ (q, concept) for q in c_data['queries'] ]
        if pos_concepts:
            pos_queries.extend(sample_queries(
                self.dataset_state.train_dataset, pos_concepts,
                num_queries=20, return_concepts=True
            ))
        neg_queries = sample_queries(
            self.dataset_state.train_dataset, hard_alt_concepts,
            num_queries=len(pos_queries), return_concepts=True
        )
        neg_queries.extend(sample_queries(
            self.dataset_state.train_dataset, alt_concepts,
            num_queries=len(pos_queries) * 5 - len(neg_queries), return_concepts=True
        ))
        entries = []

        concept_desc = self.dataset_state.train_dataset.deepget(concept)['name']
        if names := (self.dataset_state.train_dataset.deepget(concept)['context']['names']):
            concept_desc += ' in the context of ' + ', '.join(names)

        for query, concept in pos_queries:

            for refuse_template in self.REFUSAL_TEMPLATES:
                messages = [
                    dict(role='user', content=self.prepare_for_inference(concept, query)),
                    dict(role='assistant', content=refuse_template.format(concept_desc))
                ]
                entries.append(dict(messages=messages))

        for query, concept in neg_queries:

            accepted_reply = self.dataset_state.train_dataset.deepget(concept)['responses'][generate_id(query)]['unsafe']
            messages = [
                dict(role='user', content=self.prepare_for_inference(concept, query)),
                dict(role='assistant', content=accepted_reply)
            ]
            entries.append(dict(messages=messages))

        return datasets.Dataset.from_list(entries).shuffle(seed=seed)

    def train_adapter(self, tokenizer, model_id, model, concept, adapter_path, seed=42, pbar=None):
        transformers.set_seed(seed)

        config = peft.LoraConfig(
            task_type='CAUSAL_LM', lora_alpha=32, r=32, lora_dropout=0.0,
            target_modules = [ "q_proj", "v_proj", "o_proj", "down_proj", "up_proj", "gate_proj" ],
            # target_modules = "all-linear"
        )

        dataset = self.make_concept_dataset(concept, tokenizer, seed=seed)
        if pbar: pbar.set_postfix({ 'size': len(dataset) })

        training_args = trl.SFTConfig(
            do_train=True, per_device_train_batch_size=16, output_dir=adapter_path,
            bf16=True, report_to='none', gradient_checkpointing=True, gradient_accumulation_steps=2,
            lr_scheduler_type="cosine", weight_decay=0.01, optim='paged_adamw_32bit', seed=seed,
            **MODEL_SPECIFIC_PARAMS.get(model_id, MODEL_SPECIFIC_PARAMS['*'])
        )

        trainer = trl.SFTTrainer(
            model, args=training_args, peft_config=config,
            train_dataset=dataset, tokenizer=tokenizer,
        )

        trainer.train()
        trainer.save_model(adapter_path)

    def prepare(self, model_id, model, dataset_state, concepts: list[str | tuple[str, str]],
                seed=42, node_data=None, sibling_map=None, atomic_state=None, aux_model=False,
                **prepare_kwargs):
        adapters = {}
        _model   = None
        pbar     = None

        self.dataset_state = dataset_state
        self.node_data     = node_data
        self.sibling_map   = sibling_map
        self.atomic_state  = atomic_state

        if not concepts: return

        if len(concepts) > 1: concepts_iter = (pbar := tqdm.tqdm(concepts))
        else                : concepts_iter = concepts

        for concept in concepts_iter:
            concept_id = generate_id('#'.join(concept))

            if pbar: pbar.set_description(dataset_state.train_dataset.deepget((*concept, 'name')))

            adapter_path = os.path.join(
                f"data", dataset_state.name.upper(), "adapters.tuning",
                self.tuning_method, model_id, f"{self.tuning_algorithm}.{concept_id}-{seed}"
            )

            if os.path.exists(adapter_path) and len(os.listdir(adapter_path)) > 0:
                adapters['#'.join(concept)] = adapter_path; continue

            os.makedirs(os.path.dirname(adapter_path), exist_ok=True)

            if not _model:
                _model_kwargs = dict(**get_inference_params())
                # if self.HAS_UNSLOTH:
                #     _model_kwargs['dtype'] = _model_kwargs['torch_dtype']
                #     del _model_kwargs['torch_dtype']
                # _model_kwargs['device_map'] = 'balanced_low_0'
                if aux_model:
                    if model.model is None: model.load()
                    _model = model.model
                else:
                    _model = transformers.AutoModelForCausalLM.from_pretrained(
                        model.name, trust_remote_code=True,
                        **_model_kwargs
                    )
                    _model.eval()

            self.train_adapter(model.tokenizer, model_id, _model, concept, adapter_path, seed=seed, pbar=pbar)

            adapters['#'.join(concept)] = adapter_path

        self.adapters = adapters

        del _model
        sync_vram()

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

# ======================================== Registry

if 'tuning-sft' in TECHNIQUE_CONFIGS:
    register_technique('tuning-sft', AbstentionWithSFTTraining())