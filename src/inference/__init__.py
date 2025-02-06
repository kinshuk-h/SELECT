import importlib

from .utils import *
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
    inf_class = getattr(importlib.import_module(mod, __name__), inf_impl)
    setattr(curr_mod_ref, inf_impl, inf_class)

    return inf_class

def load_backend(backend):
    if backend == "openai":
        return __test_and_add_backend("OpenAIModelInference"     , ".openai"     , ("openai", "tiktoken"))
    elif backend == "vllm":
        return __test_and_add_backend("VLLMModelInference"       , ".vllm"       , ("vllm", ))
    elif backend == "huggingface":
        return __test_and_add_backend("HuggingFaceModelInference", ".huggingface", ("transformers", "accelerate"))

def prepare_model_for_inference(model_name: str, backend='huggingface', seed=42):
    """ Wrapper function to construct a ModelInference object from a model key name """

    model_kwargs = get_inference_params(model_name)

    if model_name.lower().startswith("openai"):
        backend, model_name = model_name[:6], model_name[7:]

    return load_backend(backend)(model_name, seed=seed, **model_kwargs)

class ModelManager:
    """ Context Manager for model inference: Maintains
        resources for a model in a finite context. """

    def __init__(self, *args, model_initializer=prepare_model_for_inference, **kwargs) -> None:
        self.model_initializer = model_initializer
        self.args = args
        self.kwargs = kwargs

    def __enter__(self) -> ModelInference:
        sync_vram()

        self.model = self.model_initializer(*self.args, **self.kwargs)
        return self.model

    def __exit__(self, exc_type, exc_val, exc_traceback):
        if hasattr(self, 'model') and self.model is not None:
            if hasattr(self.model, 'unload') and callable(self.model.unload):
                self.model.unload()
            del self.model
        sync_vram()

        if exc_val is not None: raise exc_val

__all__ = [
    "ModelInference",
    "ModelManager"
]
