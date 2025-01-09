import time

import regex
import openai
import tiktoken

from .base import ModelInference

class Object(object):
    pass

class OpenAIModelInference(ModelInference):
    BATCH_INDICATOR = (
        'Answer the following together as instructed, with the associated '
        'numbering to match responses.\n\n'
    )

    def __init__(self, name, api_key=None, endpoint=None, org_id=None, seed=42, api_version=None, deployment=None) -> None:
        super().__init__(name)
        self.seed = seed

        self.azure_model = "azure" in name.lower()
        if self.azure_model:
            self.name = self.name[6:]
            self.client = openai.AzureOpenAI(
                azure_endpoint=endpoint, api_key=api_key, api_version=api_version or "2024-02-01"
            )
            self.engine = self.name.replace('-', '').replace('.', '').replace('gpt', 'GPT').replace('turbo', '') + "TestDeployment"
        else:
            self.client = openai.OpenAI(
                organization=org_id, api_key=api_key
            )
            self.engine = self.name
        self.tokenizer = tiktoken.encoding_for_model(self.name)

    def make_prompt(self, query, system_prompt=None, group_batch=False, *args, **kwargs):
        queries = query
        if not isinstance(query, (list, tuple)):
            queries = [query]

        if system_prompt is None: system_prompt = "You are a helpful assistant."
        system_message = [ { "role": "system", "content": system_prompt } ]
        if not system_message[-1]['content']: system_message = []

        if group_batch and len(queries) > 1:
            grouped_query = self.BATCH_INDICATOR + '\n'.join(f"{i}. {query}" for i, query in enumerate(queries, 1))
            return [ [ *system_message, { "role": "user", "content": grouped_query.strip() } ] ]

        return [
            [ *system_message, { "role": "user", "content": query.strip() } ]
            for query in queries
        ]

    def tokenize(self, prompt):
        if isinstance(prompt, list):
            return self.tokenizer.encode(prompt[-1]['content'])
        else:
            return self.tokenizer.encode(prompt)

    def next_prediction_logits(self, prompt):
        return super().next_prediction_logits(prompt)

    def __select_best_response(self, chat_completion, num_sequences=1):
        if num_sequences > 1:
            return [
                choice.message.content or "No answer"
                for choice in chat_completion.choices[:num_sequences]
            ]
        return chat_completion.choices[0].message.content or "No answer"

    def get_default_safe_fail_response(self, num_responses):
        message = Object()
        setattr(message, 'content', 'No answer')
        setattr(message, 'role', 'assistant')

        choice = Object()
        setattr(choice, 'message', message)
        setattr(choice, 'finish_reason', 'stop')

        choices = [ choice ] * num_responses
        completion = Object()
        setattr(completion, 'choices', choices)

        return completion

    def get_response(self, max_calls=10, safe_mode=True, **gen_kwargs):
        call_count = 0
        while call_count < max_calls:
            try:
                response = self.client.chat.completions.create(model=self.engine, seed=self.seed, **gen_kwargs)
                return response
            except openai.OpenAIError as err:
                try:
                    if not isinstance(err, openai.RateLimitError): raise err
                    call_count += 1
                    time.sleep(2 ** call_count)
                    if call_count == max_calls: raise err
                except:
                    if not safe_mode: raise err
                    else: return self.get_default_safe_fail_response(gen_kwargs.get('n', 1))

    def __has_grouped_batch(self, messages):
        return any(self.BATCH_INDICATOR in message['content'] for message in messages if message['role']=='user')

    def __unpack_grouped_batch(self, batched_output):
        answers, last_answer, started = [], "", False
        for line in regex.split(r"(?ui)[\p{Zl}\n\r]", batched_output):
            if regex.search(r"(?ui)^\d+[.)]?", line) is not None:
                if started: answers.append(last_answer.rstrip())
                else: started = True
                last_answer = regex.sub(r"(?ui)^\d+[.)]?", "", line).lstrip()
            elif started: last_answer += '\n' + line
        answers.append(last_answer.rstrip())
        return answers

    def generate(self, inputs, safe_mode=True, **gen_kwargs):
        if isinstance(inputs, str): inputs = [ inputs ]

        if 'chat' in gen_kwargs: del gen_kwargs['chat']
        if 'do_sample' in gen_kwargs: del gen_kwargs['do_sample']

        if 'max_new_tokens' in gen_kwargs:
            gen_kwargs['max_tokens'] = gen_kwargs['max_new_tokens']
            del gen_kwargs['max_new_tokens']

        if 'num_return_sequences' in gen_kwargs:
            gen_kwargs['n'] = gen_kwargs['num_return_sequences']
            del gen_kwargs['num_return_sequences']

        if len(inputs) == 1 and self.__has_grouped_batch(inputs[0]):
            if 'max_tokens' not in gen_kwargs:
                gen_kwargs['max_tokens'] = 2048

        replies = [
            self.get_response(messages=messages, safe_mode=safe_mode, **gen_kwargs)
            for messages in inputs
        ]
        outputs = [
            self.__select_best_response(reply, num_sequences=gen_kwargs.get('n', 1))
            for reply in replies
        ]
        if len(outputs) == 1 and self.__has_grouped_batch(inputs[0]):
            outputs = self.__unpack_grouped_batch(outputs[0])
        return outputs