
import json

import regex

from ...utils import formatting

from ..base import AbstentionTechnique
from ..manager import register_technique
from ..constants import TECHNIQUE_CONFIGS, STRINGS

class AbstentionWithPrompting(AbstentionTechnique):
    """ Common wrapper for simple prompting-based abstention methods. """

    def __init__(self, name) -> None:
        super().__init__('PROMPTING', **TECHNIQUE_CONFIGS[name])

    def prepare(self, model_id, model, train_dataset, concept: str | tuple[str, str], **prepare_kwargs):
        # TODO add dynamic few shot example generation here.
        pass

    def prepare_instance(self, concept, query, examples=None, instruction=None):
        """ Creates the approach specific prompt from a query and concept (with context).

        Args:
            concept (str): Concept to mention in the prompt, to abstain from.
            query (str): Actual query to process.
            examples (list[tuple[str, str]], optional): Query-Answer pairs to use as examples. Defaults to None.
            instruction (str): Instruction to override with.

        Returns:
            str|list[str]: Prompt(s) to use as per the approach.
        """
        instruction = instruction or self.instruction
        if instruction: instruction = instruction.format(concept=concept)

        example_insts = []
        if 'few_shot' in self.tmpl_name:
            if 'cot' in self.tmpl_name:
                for inst in examples:
                    eg_concept = inst['concept']
                    if inst['context']: eg_concept += ' in the context of ' + ', '.join(inst['context'])
                    r_type = 'entailment' if eg_concept == concept else 'contradiction'
                    reason = STRINGS['reasoning_format'][r_type].format(concept=concept)
                    reply  = json.dumps(dict(reasoning=reason, answer=inst['response']), ensure_ascii=False, indent=1)
                    example_insts.append(STRINGS['example_format'].format(query=inst['query'], response=reply))
            else:
                example_insts.extend((
                    STRINGS['example_format'].format(query=inst['query'], response=inst['response'])
                    for inst in examples
                ))

        if isinstance(self.template, list):
            return [
                formatting.format_prompt(
                    template, concept=concept, instruction=instruction,
                    query=query, examples='\n\n'.join(example_insts)
                )
                for template in self.template
            ]
        else:
            return formatting.format_prompt(
                self.template, concept=concept, instruction=instruction,
                query=query, examples='\n\n'.join(example_insts)
            )

    @staticmethod
    def __extract_answer(output):
        if match := regex.search(r"(?ui)(['\"])answer\1\s*:\s*['\"]?\s*((?:.|\n)+)\s*['\"]?\s*\}?", output):
            return regex.sub(r"(?ui)(?:[\}\s'\"]*)$", "", match[2]).replace(r"\n", "\n")
        return output

    def generate(self, model, instances: list, *args, **gen_kwargs):
        generate_kwargs = dict(chat=True, max_new_tokens=512, do_sample=False, temperature=0)
        generate_kwargs.update(**gen_kwargs)

        fmtd_prompts = model.make_prompt(instances, instructions=[], chat=True)

        if 'cot' in self.tmpl_name:
            # Ensure sufficient generation for CoT reasoning
            if generate_kwargs['max_new_tokens'] < 1024:
                generate_kwargs['max_new_tokens'] = 1024

        outputs = model.generate(fmtd_prompts, **generate_kwargs)
        if 'cot' in self.tmpl_name:
            return [
                dict(complete=output, answer=self.__extract_answer(output))
                for output in outputs
            ]
        return outputs

# ======================================== Registry

for alias in TECHNIQUE_CONFIGS:
    if alias.startswith('prompt'): register_technique(alias, AbstentionWithPrompting(alias))