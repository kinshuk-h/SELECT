import abc

class ModelInference(abc.ABC):
    def __init__(self, name) -> None:
        super().__init__()
        self.name = name

    @abc.abstractmethod
    def make_prompt(self, query, instructions=None, examples=None, *args, **kwargs):
        raise NotImplementedError("Implement in derived classes")

    @abc.abstractmethod
    def generate(self, inputs, *gen_args, **gen_kwargs):
        raise NotImplementedError("Implement in derived classes")

    @abc.abstractmethod
    def tokenize(self, prompt):
        raise NotImplementedError("Implement in derived classes")

    @abc.abstractmethod
    def next_prediction_logits(self, prompt, *args, **kwargs):
        raise NotImplementedError("Implement in derived classes")
