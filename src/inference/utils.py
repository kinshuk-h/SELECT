"""

    utils
    ~~~~~

    Model inference utilities.

"""

import os

# =======================================================

def contains(text: str, words):
    """ Checks if a string contains any of the given words in an iterable. """
    return any(word in text for word in words)

def is_conversation(prompt: list|str):
    """ Detects if a (batched) prompt(s) corresponds to conversation/chat instances. """
    return isinstance(prompt, list) and (
        isinstance(prompt[0], dict) or (
            isinstance(prompt[0], list) and isinstance(prompt[0][0], dict)
        )
    )

# =======================================================

def seed_all(seed=None):
    import time
    import transformers

    transformers.set_seed(seed or time.time_ns())

def sync_vram():
    """ Synchronizes GPU memory to free resources. """

    import gc
    import torch

    if not torch.cuda.is_available(): return

    while gc.collect() > 0:
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

# =======================================================

OPENAI_CREDENTIALS = {
    'org_id' : os.environ.get('OPENAI_ORG_ID'),
    'api_key': os.environ.get('OPENAI_API_KEY')
}

AZURE_OPENAI_CREDENTIALS = {
    'endpoint'   : os.environ.get('AZURE_OPENAI_ENDPOINT'),
    'api_key'    : os.environ.get('AZURE_OPENAI_API_KEY'),
    'deployment' : os.environ.get('AZURE_OPENAI_DEPLOYMENT'),
    'api_version': os.environ.get('AZURE_OPENAI_API_VERSION')
}

def get_inference_params(model_name):
    if "openai" in model_name.lower():
        return AZURE_OPENAI_CREDENTIALS if "azure" in model_name.lower() else OPENAI_CREDENTIALS
    else:
        std_kwargs = dict(torch_dtype='bfloat16', device_map='auto')
        # Follows from: https://github.com/huggingface/transformers/issues/32390
        if "gemma" in model_name.lower(): std_kwargs['attn_implementation'] = 'flash_attention_2'
        return std_kwargs