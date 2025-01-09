from .dinm import AbstentionWithDINM
from .repe import AbstentionWithReprEngineering

def get_approach(name):
    algorithm = name.split('-')[-1]
    if algorithm == "dinm": return AbstentionWithDINM()
    elif algorithm == "repe": return AbstentionWithReprEngineering()
    raise ModuleNotFoundError(name)

__all__ = [ "AbstentionWithDINM", "AbstentionWithReprEngineering", "get_approach" ]