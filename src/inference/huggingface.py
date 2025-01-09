import re
import sys
import itertools

import torch
import transformers

try:
    import peft
    __PEFT_AVAILABLE__ = True
except:
    __PEFT_AVAILABLE__ = False

from .base import ModelInference
from .utils import is_conversation, contains

class HuggingFaceModelInference(ModelInference):
    def __init__(self, name, seed=42, is_peft_model=False, lazy_load=True, **model_kwargs) -> None:
        super().__init__(name)

        self._peft = __PEFT_AVAILABLE__ and is_peft_model
        self.expecting_warnings = False

        tokenizer_args = {}

        if contains(name.lower(), ("gemma-2", )) and "2b" not in name.lower():
            # Gemma 2 was likely trained with right padding:
            # https://github.com/huggingface/transformers/issues/30004
            tokenizer_args['padding_side'] = "right"
            self.expecting_warnings = True

        if contains(name.lower(), ("bloom", "gpt", "llama-3", "qwen", "gemma")):
            tokenizer_args['padding_side'] = "left"

        if contains(name.lower(), ("llama", "mistral")):
            tokenizer_args['add_prefix_space'] = None

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            self.name, trust_remote_code=True, **tokenizer_args
        )

        if getattr(self.tokenizer, 'pad_token', None) is None:
            self.tokenizer.pad_token = self.tokenizer.unk_token or self.tokenizer.eos_token

        self.model = None
        self.seed  = seed
        self.decoding_mode = "simple"
        if not model_kwargs: model_kwargs = dict(device_map="auto")
        self.model_kwargs = model_kwargs

        self.dtype = self.model_kwargs.get('torch_dtype', torch.float32)

        if self._peft:
            self.config = peft.PeftConfig.from_pretrained(name)
        else:
            self.config = transformers.AutoConfig.from_pretrained(name)

        if any("causal" in arch for arch in getattr(self.config, 'architectures', [])):
            self.config.is_decoder = True

        if not lazy_load: self.load()

    def apply_adapter(self, adapter_name_or_config, **kwargs):
        assert __PEFT_AVAILABLE__, "The peft library is not available to load adapters."

        if self.model is None: self.load()
        elif isinstance(self.model, peft.PeftModel):
            self.model = self.model.unload()

        if adapter_name_or_config is not None:
            adapter_kwargs = dict()
            if isinstance(adapter_name_or_config, str):
                adapter_kwargs['model_id'] = adapter_name_or_config
                adapter_kwargs.update(**kwargs)
            else:
                adapter_kwargs['config'] = adapter_name_or_config

            self.model = peft.PeftModel.from_pretrained(
                self.model, **adapter_kwargs
            )

    def load(self):
        if self.model is not None: return

        transformers.set_seed(self.seed)

        is_peft, task = 'Peft' if self._peft else '', 'Seq2Seq' if "t5" in self.name.lower() else 'Causal'
        MODEL_INIT_MODULE = sys.modules['peft' if self._peft else 'transformers']
        MODEL_INIT_CLASS  = getattr(MODEL_INIT_MODULE, f"Auto{is_peft}ModelFor{task}LM")
        self.model = MODEL_INIT_CLASS.from_pretrained(self.name, trust_remote_code=True, **self.model_kwargs)
        self.model.eval()

        self.device = self.model.device

        if getattr(self.model.generation_config, 'pad_token_id') is None:
            self.model.generation_config.pad_token_id = self.tokenizer.pad_token_id

    def unload(self):
        if self.model is not None:
            if isinstance(self.model, peft.PeftModel):
                del self.model.base_model
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

    def next_prediction_logits(self, prompts, all_logits=False):
        inputs = self.tokenize(prompts, return_tensors="pt", padding="longest")

        if self.model is None: self.load()
        inputs = inputs.to(self.model.device)

        if self.config.is_encoder_decoder:
            outputs = self.model(**inputs, decoder_input_ids=self.tokenizer("", return_tensors="pt")['input_ids'])
        else:
            outputs = self.model(**inputs)

        logits = outputs.logits.detach().cpu()

        return logits if all_logits else logits[:, -1]

    def postprocess_output(self, output, prefill=None):
        if self.decoding_mode == "aggressive":
            if match := re.search(r"(?ui)[\r\n#]+(.|\n)*$", output):
                output = output[:match.span()[0]]
        return ((prefill or '') + output).strip()

    @torch.no_grad()
    def generate(self, inputs, prefill=None, chat='auto', strip_output=True, decoding='simple', **gen_kwargs):
        """ Wrapper over the underlying model's generate function, to automatically handle tokenization and cleanup.

        Args:
            inputs: Input(s) for generation (strings or conversations), processed via make_prompt.
            prefill (str|list[str]|None): Text to prefill the output with.
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

        # Tokenize, and include prefill tokens.
        _inputs = inputs
        if chat and ('chat' in self.name.lower() or self.tokenizer.chat_template is not None):
            _inputs = self.tokenizer.apply_chat_template(_inputs, tokenize=False, add_generation_prompt=True)
        if prefill:
            if not isinstance(_inputs, (list, tuple)): _inputs += prefill
            else: _inputs = [ _input + _prefill for _input, _prefill in zip(_inputs, prefill) ]

        _inputs = self.tokenizer(_inputs, add_special_tokens=not chat, return_tensors="pt", padding="longest")

        input_len = _inputs.input_ids.shape[-1]
        _inputs = _inputs.to(self.model.device)

        if gen_kwargs.get('num_return_sequences', 1) > 1:
            gen_kwargs['return_dict_in_generate'] = True
            gen_kwargs['output_scores'] = True

        # Generate new tokens.
        with torch.inference_mode():
            if self.expecting_warnings:
                old_verbosity = transformers.logging.get_verbosity()
                transformers.logging.set_verbosity(transformers.logging.ERROR)
            outputs = self.model.generate(**_inputs, **gen_kwargs)
            if self.expecting_warnings:
                transformers.logging.set_verbosity(old_verbosity)

        # Post-process generated tokens and outputs
        decode_kwargs = dict(skip_special_tokens=True, clean_up_tokenization_spaces=True)

        output_seqs = self.tokenizer.batch_decode([
            output[input_len:] if strip_output and "t5" not in self.name else output
            for output in (outputs.sequences if gen_kwargs.get('return_dict_in_generate') else outputs)
        ], **decode_kwargs)

        if prefill and (seq_per_batch := gen_kwargs.get('num_return_sequences', 1)) > 1:
            prefill = [ _prefill for prf in prefill for _prefill in ((prf, ) * seq_per_batch) ]

        sequences = [
            self.postprocess_output(output, _prefill)
            for output, _prefill in zip(output_seqs, prefill or itertools.repeat(None))
        ]

        if (seq_per_batch := gen_kwargs.get('num_return_sequences', 1)) > 1:
            sequences = [ sequences[i:i+seq_per_batch] for i in range(0, len(sequences), seq_per_batch) ]

        return sequences
