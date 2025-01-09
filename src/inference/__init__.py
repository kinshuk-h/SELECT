import warnings
import importlib

from .base import ModelInference

def __test_and_add_backend(inf_impl, mod, mod_depend):
    for _mod in mod_depend:
        try:
            importlib.import_module(_mod)
        except ImportError as exc:
            raise ImportError(
                f"{inf_impl} could not be imported, likely because the package {_mod} is missing. "
                f"Install it using `pip install {_mod}` or equivalent. "
            ) from exc

    curr_mod_ref = importlib.import_module(__name__)
    setattr(curr_mod_ref, inf_impl, getattr(importlib.import_module(mod, __name__), inf_impl))

def load_backend(backend):
    if backend == "openai":
        __test_and_add_backend("OpenAIModelInference"     , ".openai"     , ("openai", "tiktoken"))
    elif backend == "vllm":
        __test_and_add_backend("VLLMModelInference"       , ".vllm"       , ("vllm", ))
    elif backend == "huggingface":
        __test_and_add_backend("HuggingFaceModelInference", ".huggingface", ("transformers", "accelerate"))

__all__ = [
    "ModelInference",
]
