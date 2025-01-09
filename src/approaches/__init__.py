from .base import AbstentionTechnique
from .constants import APPROACH_CONFIGS
from . import prompting, editing, unlearning, tuning

def get_approach(name):
    """ Resolves an approach by name to the corresponding technique instance. """

    if name.startswith("prompt"):
        return prompting.get_approach(name)
    elif name.startswith("model_edit"):
        return editing.get_approach(name)
    elif name.startswith("unlearning"):
        return unlearning.get_approach(name)
    elif name.startswith("tuning"):
        return tuning.get_approach(name)
    else:
        raise ModuleNotFoundError(name)

APPROACHES: dict[str, AbstentionTechnique] = {
    approach: get_approach(approach)
    for approach in APPROACH_CONFIGS
}

__all__ = [
    "prompting", "editing", "unlearning",
    "get_approach", "APPROACHES"
]