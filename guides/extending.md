## Extending the Evaluation of `SELECT`

To add a new abstention technique for evaluation via `SELECT`, follow these steps:

1. Create a new class that extends `src.approaches.base.AbstentionTechnique`. This abstract class provides the base API for all techniques. Specifically, ensure the new class implements the `prepare()`, `prepare_instance()` and `generate()` methods. Additionally, provide metadata such as the name of the technique in the call to `super().__init__()`. For example:

```python
# file: new_technique.py

from src.techniques.base import AbstentionTechnique

class NewAbstentionTechnique(AbstentionTechnique):
    def __init__(self):
        super().__init__(
            nature='CUSTOM',
            name='New Abstention Technique',
            short_name='NAT',
            # this is not automatically used by default
            instruction=None,
            # See templates/prompts.yaml for options
            template='no_instruction'
        )

    def prepare_instance(self, ...):
        # Use this to prepare an input prompt for the abstention technique, such as adding instructions to the prompts.
        # This is run once for every prompt specified for evaluation.
        # See how this is used in sample implementations in src/techniques.
        ...

    def prepare(self, ...):
        # Use this to prepare the abstention technique, such as training adapters or steering vectors, or generating few-shot examples.
        # This is run only once per model.
        ...

    def generate(self, ...):
        # Actual generation code, specific to a batch of prompts, goes here.
        # Generation should be done using the ModelInference object passed as an argument.
        # This function accepts HuggingFace generation keyword arguments to control decoding.
        ...

```

1. In the main script (`{generate,evaluate}.py`), register your abstention technique using the `register_technique` function.
   Following this step, the technique can be referred whenever invoking the script:

```python
from src.techniques import APPROACHES, register_technique

from new_technique import NewAbstentionTechnique

...

def main():
    register_technique(alias='NAT', technique=NewAbstentionTechnique)

    parser = make_parser()
    ...
```

1. Evaluate models with the added abstention technique(s):

```bash
python3 generate.py -m Gemma-2-IT-2B -a NAT -n 1 -b 32
python3 evaluate.py -m Gemma-2-IT-2B -a NAT
```