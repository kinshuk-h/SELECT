from .base import AbstentionTechnique

class Manager:
    __instance = None

    def __init__(self):
        self.approaches: dict[str, AbstentionTechnique] = {}

    @classmethod
    def instance(cls):
        if cls.__instance is None:
            cls.__instance = cls()
        return cls.__instance

    def register(self, alias, technique):
        self.approaches[alias] = technique

    def get(self, alias):
        return self.approaches.get(alias)

    def data(self):
        return self.approaches

def get_technique(alias):
    return Manager.instance().get(alias)

def get_techniques():
    return Manager.instance().data()

def register_technique(alias: str, technique: AbstentionTechnique):
    return Manager.instance().register(alias, technique)