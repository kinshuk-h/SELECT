import vllm
import torch
import transformers
import vllm.lora.request
import vllm.model_executor
import vllm.model_executor.utils

from .base import ModelInference
from .utils import contains, is_conversation

def get_available_gpus(free_ratio=0.8):
    avl_gpus = []
    for device_id in range(torch.cuda.device_count()):
        free, total = torch.cuda.mem_get_info(device_id)
        if (free / total) > free_ratio: avl_gpus.append(device_id)
    return avl_gpus

class VLLMModelInference(ModelInference):
    def __init__(self, name, seed=42, lazy_load=True, **model_kwargs) -> None:
        super().__init__(name)

        tokenizer_args = {}

        if contains(name.lower(), ("gemma-2", )):
            # Gemma 2 was likely trained with right padding:
            # https://github.com/huggingface/transformers/issues/30004
            tokenizer_args['padding_side'] = "right"
            self.expecting_warnings = True

        elif contains(name.lower(), ("bloom", "gpt", "llama-3", "qwen", "gemma")):
            tokenizer_args['padding_side'] = "left"

        if contains(name.lower(), ("llama", "mistral")):
            tokenizer_args['add_prefix_space'] = None

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            self.name, trust_remote_code=True, **tokenizer_args
        )

        if getattr(self.tokenizer, 'pad_token', None) is None:
            self.tokenizer.pad_token = self.tokenizer.unk_token or self.tokenizer.eos_token

        self.config = transformers.AutoConfig.from_pretrained(self.name)

        self.model   = None
        self.adapter = None
        self.seed    = seed
        self.model_kwargs = model_kwargs

        if not lazy_load: self.load()

    def load(self):
        if self.model is not None: return

        vllm.model_executor.utils.set_random_seed(self.seed)
        parallelism_count = len(get_available_gpus(free_ratio=0.6))
        model_load_kwargs = dict(
            trust_remote_code=True, seed=self.seed, enable_lora=self.adapter is not None,
            gpu_memory_utilization=self.model_kwargs.get('max_gpu_usage', 0.8),
            max_model_len=min(
                getattr(self.config, 'sliding_window', None) or self.config.max_position_embeddings,
                self.config.max_position_embeddings, 10240
            ), dtype=self.model_kwargs.get('torch_dtype', 'auto'), max_lora_rank=4096,
        )

        if self.model_kwargs.get('use_pipeline_parallelism'):
            model_load_kwargs['pipeline_parallel_size'] = parallelism_count
        else:
            model_load_kwargs['tensor_parallel_size'] = parallelism_count

        self.model = vllm.LLM(self.name, **model_load_kwargs)

    def apply_adapter(self, adapter_name_or_config):
        self.adapter = None
        if adapter_name_or_config is not None:
            self.adapter = vllm.lora.request.LoRARequest(
                "lora.adapter", 1, adapter_name_or_config
            )

    def unload(self):
        if self.model is not None:
            del self.model

            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    def make_prompt(self, query, chat=False, system_prompt=None, *args, **kwargs):
        queries = query
        if not isinstance(query, (list, tuple)):
            queries = [query]

        if chat and ('chat' in self.name.lower() or self.tokenizer.chat_template is not None):
            if system_prompt is None: system_prompt = "You are a helpful assistant."
            system_instruction = [ { "role": "system", "content": system_prompt } ]
            if not system_instruction[-1]['content']: system_instruction = []

            try:
                self.tokenize([ *system_instruction, { "role": "user", "content": query } ])
            except:
                system_instruction = []

            return [
                [ *system_instruction, { "role": "user", "content": query } ]
                for query in queries
            ]

        else:
            return queries

    def tokenize(self, prompt: str|list, **tokenizer_kwargs):
        if is_conversation(prompt):
            prompt = self.tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
            # Special tokens are already added, should not add them twice
            tokenizer_kwargs['add_special_tokens'] = False
        return self.tokenizer(prompt, **tokenizer_kwargs)['input_ids']

    def next_prediction_logits(self, prompt, *args, **kwargs):
        return super().next_prediction_logits(prompt, *args, **kwargs)

    @torch.no_grad()
    def generate(self, inputs, prefill=None, chat='auto', decoding='simple', **gen_kwargs):
        """ Wrapper over the underlying model's generate function, to automatically handle tokenization and cleanup.

        Args:
            inputs: Input(s) for generation (strings or conversations).
            prefill (str|list[str]|None): Text to prefill the output with.
            counterfactuals (str, optional): Counterfactual values to check during generation. Defaults to None.
            return_score (bool, optional): If true, returns the logits associated with outputs. Defaults to False.
            chat (bool|str, optional): Whether the inputs correspond to conversations. Defaults to auto-detect.
                Will be formatted further as per the chat template in the case of conversations.
            strip_output (bool, optional): If true, cleans the preceding input from model responses. Defaults to True.
            decoding (str, optional): Mode of decoding, if 'aggressive', output is scrubbed to remove irregularities.
                Defaults to 'simple'.

        Returns:
            list[str] | list[list[str]] | list[tuple[str,int]] | list[list[tuple[str,int]]]: Generations, optionally
                with scores as requested. All generations are decoded using the model's tokenizer.
        """

        if self.model is None: self.load()
        self.decoding_mode = decoding or 'simple'

        if chat == 'auto': chat = is_conversation(inputs)

        _inputs = inputs
        if chat and ('chat' in self.name.lower() or self.tokenizer.chat_template is not None):
            _inputs = self.tokenizer.apply_chat_template(_inputs, tokenize=False, add_generation_prompt=True)
        if prefill:
            if not isinstance(_inputs, (list, tuple)): _inputs += prefill
            else: _inputs = [ _input + _prefill for _input, _prefill in zip(_inputs, prefill) ]

        _tokens = self.tokenizer(_inputs, add_special_tokens=not chat, return_tensors="pt", padding="longest")
        input_len = _tokens.input_ids.shape[-1]

        _inputs = map(lambda _input: self.tokenizer(_input, add_special_tokens = not chat), _inputs)
        _inputs = [ vllm.TokensPrompt(prompt_token_ids=_input.input_ids) for _input in _inputs ]

        if 'max_new_tokens' in gen_kwargs:
            gen_kwargs['max_tokens'] = gen_kwargs['max_new_tokens'] + input_len
            del gen_kwargs['max_new_tokens']

        if 'num_return_sequences' in gen_kwargs:
            gen_kwargs['n'] = gen_kwargs['num_return_sequences']
            del gen_kwargs['num_return_sequences']

        if 'do_sample' in gen_kwargs:
            if not gen_kwargs['do_sample']: gen_kwargs['temperature'] = 0
            del gen_kwargs['do_sample']

        sample_kwargs = vllm.SamplingParams(**gen_kwargs)
        outputs = self.model.generate(_inputs, sampling_params=sample_kwargs, lora_request=self.adapter)

        if gen_kwargs.get('n', 1) > 1:
            return [
                [ response.text for response in output.outputs ]
                for output in outputs
            ]
        return [ output.outputs[0].text for output in outputs ]