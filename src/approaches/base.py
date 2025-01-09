import abc

from .utils import STRINGS, PROMPTS
from ..evaluation.models import ModelInference

class AbstentionTechnique(abc.ABC):
    def __init__(self, nature, name, short_name, instruction=None, template=None) -> None:
        super().__init__()

        self.nature = nature
        self.name        = name
        self.short_name  = short_name
        self.instr_name  = instruction
        self.instruction = STRINGS.get(instruction, '')
        self.tmpl_name   = ', '.join(template) if isinstance(template, list) else template
        self.template    = None

        if isinstance(template, list):
            self.template = [ PROMPTS[tmpl] for tmpl in template ]
        else:
            self.template = PROMPTS[template]

    @abc.abstractmethod
    def prepare(self, model_id, model: ModelInference, dataset_state,
                concepts: list[str|tuple[str, str]], **prepare_kwargs):
        """ Prepares the abstention method for a set of concepts.

        Args:
            model_id (str): Model ID to identify processed items from cached data.
            model (ModelInference): Model instance to prepare.
            dataset_state (DatasetState): Dataset state to borrow instances from.
            concepts (list[str|tuple[str, str]]): List of atomic/composite concepts to prepare abstention for.

        Raises:
            NotImplementedError: Must be implemented in subclasses.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def prepare_for_inference(self, concept: str|tuple[str, str], request: str, **prepare_kwargs):
        """ Prepares an instance for inference with the abstention method.

        Args:
            concept (str | tuple[str, str]): Atomic/Composite concept to abstain from.
            request (str): User request / query to process.

        Raises:
            NotImplementedError: Must be implemented in subclasses.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def generate(self, model: ModelInference, instances: list, **gen_kwargs):
        """Performs inference for a set of prompts using a model prepared for the abstention method.

        Args:
            model (ModelInference): Model to perform inference with.
            instances (list): User request instances prepared for inference.

        Raises:
            NotImplementedError: Must be implemented in subclasses.
        """
        raise NotImplementedError