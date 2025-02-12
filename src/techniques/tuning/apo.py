import random

import trl
import peft
import datasets
import transformers

from ...evaluation.dataset import generate_id
from .base import AbstentionByPostTraining, register

class AbstentionWithDPOTraining(AbstentionByPostTraining):
    """ Learning abstention using DPO (optionally post SFT) """

    COMMON_TRAIN_KWARGS = dict(
        learning_rate=5.0e-6, num_train_epochs=5,
        per_device_train_batch_size=8, bf16=True, max_length=8192, report_to='none',
        gradient_checkpointing=True, gradient_accumulation_steps=2,
        lr_scheduler_type="cosine", weight_decay=0.01, optim='paged_adamw_32bit',
    )

    def __init__(self, post_sft_checkpoint=True):
        super().__init__(
            'tuning-sft-dpo' if post_sft_checkpoint else 'tuning-dpo',
            'APO', 'SFT-DPO' if post_sft_checkpoint else 'DPO'
        )
        self.post_sft_checkpoint = post_sft_checkpoint

    def make_concept_dataset(self, concept, tokenizer, num_queries=20, query_repeats=2):
        node, pos_queries, neg_queries = self.collect_concept_data(concept, num_queries, query_repeats)

        entries, concept_desc = [], node.name
        if names := (node.context.ids):
            concept_desc += ' in the context of ' + ', '.join(names)

        # format positive examples
        for query, _concept in pos_queries:
            for refuse_template in random.sample(self.REFUSAL_TEMPLATES, query_repeats):
                query_msg = [ dict(role='user', content=self.prepare_instance(concept, query)) ]
                prompt = tokenizer.apply_chat_template(query_msg, tokenize=False, add_generation_prompt=True)

                chosen = tokenizer.apply_chat_template([
                    *query_msg, dict(role='assistant', content=refuse_template.format(concept_desc))
                ], tokenize=False)[len(prompt):]

                rejected_reply = node.view(_concept).responses[generate_id(query)].unsafe
                rejected = tokenizer.apply_chat_template([
                    *query_msg, dict(role='assistant', content=rejected_reply)
                ], tokenize=False)[len(prompt):]

                entries.append(dict(prompt=prompt, chosen=chosen, rejected=rejected))

        # format negative examples
        for query, _concept in neg_queries:
            for refuse_template in random.sample(self.REFUSAL_TEMPLATES, query_repeats+1):
                query_msg = [ dict(role='user', content=self.prepare_instance(concept, query)) ]
                prompt = tokenizer.apply_chat_template(query_msg, tokenize=False, add_generation_prompt=True)

                accepted_reply = node.view(_concept).responses[generate_id(query)].unsafe
                chosen = tokenizer.apply_chat_template([
                    *query_msg, dict(role='assistant', content=accepted_reply)
                ], tokenize=False)[len(prompt):]

                rejected = tokenizer.apply_chat_template([
                    *query_msg, dict(role='assistant', content=refuse_template.format(concept_desc))
                ], tokenize=False)[len(prompt):]
                entries.append(dict(prompt=prompt, chosen=chosen, rejected=rejected))

        return datasets.Dataset.from_list(entries)

    def train_adapter(self, tokenizer, model, concept, adapter_path,
                      train_kwargs, seed=42, pbar=None):
        # offset the seed to avoid training on the _exact_ same instances
        transformers.set_seed(seed + (2024 if self.post_sft_checkpoint else 0))

        if self.post_sft_checkpoint:
            sft_adapter_path = adapter_path.replace(self.tuning_method, 'AFT').replace(self.tuning_algorithm, 'SFT')
            adapter_model = peft.PeftModel.from_pretrained(model, sft_adapter_path, is_trainable=True)
            adapter_model.load_adapter(sft_adapter_path, adapter_name="sft")
            config = None
            extra_kwargs = dict(model_adapter_name="default", ref_adapter_name="sft",)
        else:
            config = peft.LoraConfig(
                task_type='CAUSAL_LM', lora_alpha=32, r=32, lora_dropout=0.0,
                target_modules = [ "q_proj", "v_proj", "o_proj", "down_proj", "up_proj", "gate_proj" ],
            )
            adapter_model = peft.get_peft_model(model, config)
            extra_kwargs = {}

        dataset = self.make_concept_dataset(concept, tokenizer).shuffle(seed=seed)
        if pbar: pbar.set_postfix({ 'size': len(dataset) })

        training_args = trl.DPOConfig(
            do_train=True, output_dir=adapter_path,
            seed=seed, **train_kwargs, **extra_kwargs
        )

        trainer = trl.DPOTrainer(
            adapter_model, beta=0.1, args=training_args,
            train_dataset=dataset, tokenizer=tokenizer,
            peft_config=config, loss_type='sigmoid'
        )

        trainer.train()
        trainer.save_model(adapter_path)

        adapter_model.unload()

# ======================================== Registry

try:
    register(AbstentionWithDPOTraining(post_sft_checkpoint=True))
    register(AbstentionWithDPOTraining(post_sft_checkpoint=False))
except:
    pass