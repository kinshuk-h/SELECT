import random

import trl
import peft
import datasets
import transformers

from ...evaluation.dataset import generate_id
from .base import AbstentionByPostTraining, register

class AbstentionWithSFTTraining(AbstentionByPostTraining):
    """ Learning abstention using SFT """

    COMMON_TRAIN_KWARGS = dict(
        per_device_train_batch_size=16, bf16=True, report_to='none',
        gradient_checkpointing=True, gradient_accumulation_steps=2,
        lr_scheduler_type="cosine", weight_decay=0.01, optim='paged_adamw_32bit',
        num_train_epochs=10, learning_rate=5.0e-5
    )

    def __init__(self):
        super().__init__('tuning-sft', 'AFT', 'SFT')

    def make_concept_dataset(self, concept, num_queries=20, query_repeats=5):
        node, pos_queries, neg_queries = self.collect_concept_data(concept, num_queries, query_repeats)

        entries, concept_desc = [], node.name
        if names := (node.context.ids):
            concept_desc += ' in the context of ' + ', '.join(names)

        # format positive examples
        for query, _concept in pos_queries:
            for refuse_template in random.sample(self.REFUSAL_TEMPLATES, query_repeats):
                messages = [
                    dict(role='user', content=self.prepare_instance(_concept, query)),
                    dict(role='assistant', content=refuse_template.format(concept_desc))
                ]
                entries.append(dict(messages=messages))

        # format negative examples
        for query, _concept in neg_queries:
            accepted_reply = node.view(_concept).responses[generate_id(query)].unsafe
            messages = [
                dict(role='user', content=self.prepare_instance(_concept, query)),
                dict(role='assistant', content=accepted_reply)
            ]
            entries.append(dict(messages=messages))

        return datasets.Dataset.from_list(entries)

    def train_adapter(self, tokenizer, model, concept, adapter_path,
                      train_kwargs, seed=42, pbar=None):
        transformers.set_seed(seed)

        config = peft.LoraConfig(
            task_type='CAUSAL_LM', lora_alpha=32, r=32, lora_dropout=0.0,
            target_modules = [ "q_proj", "v_proj", "o_proj", "down_proj", "up_proj", "gate_proj" ],
        )

        dataset = self.make_concept_dataset(concept).shuffle(seed=seed)
        if pbar: pbar.set_postfix({ 'size': len(dataset) })

        training_args = trl.SFTConfig(
            do_train=True, output_dir=adapter_path,
            seed=seed, **train_kwargs
        )

        trainer = trl.SFTTrainer(
            model, args=training_args, peft_config=config,
            train_dataset=dataset, tokenizer=tokenizer,
        )

        trainer.train()
        trainer.save_model(adapter_path)

# ======================================== Registry

try:
    register(AbstentionWithSFTTraining())
except:
    pass