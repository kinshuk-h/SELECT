from .core import AbstentionWithPrompting

def get_approach(name):
    return AbstentionWithPrompting(name)

__all__ = [ "AbstentionWithPrompting", "get_approach" ]