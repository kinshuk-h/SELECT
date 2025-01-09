# Abstention Project

## Setup

We use [conda](https://docs.anaconda.com/miniconda/install/) to manage dependencies for the project. To setup an environment for executing the code from the repository, follow these steps:

1. Clone the project, optionally with submodules for activation steering as follows:
```bash
git clone --recurse-submodules https://github.com/kinshuk-h/abstention-project
```

2. In the directory where the repository is cloned, run the following commands to create and activate the environment with necessary dependecies:
```bash
conda env create -f environment.yml -n select
conda activate select
```

3. [Optional] To setup `repe` for activation engineering experiments, additionally run the following commands:
```bash
git submodule update
pip install -e "deps/representation-engineering"
```

## Project Hierarchy

The repository is organized as follows:

- `data`: Datasets and assets used throughout the experiments.
  - `SELECT`: queries and taxonomy for the `SELECT` dataset, derived from the [YAGO](https://yago-knowledge.org) [taxonomy](https://yago-knowledge.org/data/yago4.5/design-document.pdf).
    - `taxonomy.json`: Comprises of the base hierarchy of concepts derived from `YAGO`.
    - `taxonomy_plus.json`: Extension of `taxonomy.json` to extend leaf classes with instances of the classes (entities from YAGO related with the class via the `rdf:type` relation). This is the final taxonomy used in experiments.
    - `data.json`: Comprises of queries for different concepts, for **evaluation**.
    - `data.train.json`: Queries for different concepts, for use in training or prompting.
    - `example_cache.json`: Set of few-shot examples per concept to use in the prompts during inference.
    - `stats.concepts.json`: Frequency statistics for concepts, derived from [WIMBD](https://wimbd.allen.ai) across different corpora.
    - `hints.compose.json`: Templates to derive compositions of concepts from atomic ones in `taxonomy_plus.json`.
    - `*.compose.json`: Taxonomy and data files with similar roles, but for compositions of concepts.

- `results/expr.abstain.select`: Abstention experiment results

- `src`: Source code for experiment related utility modules.
  - `approaches`: Submodule implementing different abstention techniques. Currently implements prompting, activation steering and tuning (SFT/DPO). To add a new abstention technique, follow the steps below.
  - `inference`: Submodule for inference, supporting different backends such as `vllm` or `HuggingFace`.
  - `evaluation`: Module for utilities related to evaluation.
    - `refusal.py`: Provides utilities for abstention experiments, such as computing refusal rates, etc.
    - `dataset.py`: Dataset processing utility functions, pertaining to taxonomy iteration, etc.
    - `task.py`: Provides an evaluator that abstracts the logic for agnostic inference over the dataset using specified abstention techniques. Any new techniques will be automatically supported as long as they follow the API.
  - `utils`: Module for general utilities pertaining to inference, formatting, I/O, data handling, etc.

## Notebooks and Scripts

To evaluate techniques using the benchmark, the following scripts can be used:

- `generate.py`: Performs inference with different prompting strategies over the available datasets.
- `evaluate.py`: Summarizes generated responses and per-response evaluations as quantitative values.

#### Examples

- Run inference using LLaMa-3-Chat-8B using CoT based few-shot example based prompting to assess refusal rates and in-dataset specificity:
    ```bash
    python3 generate.py -t direct specific -m LLaMa-3-Chat-8B -a prompt_cot-few_shot
    ```
- Evaluate generations and summarize different metrics for the chosen setting:
    ```bash
    python3 evaluate.py -t direct specific -m LLaMa-3-Chat-8B -a prompt_cot-few_shot
    ```

#### Replication

To replicate the experiments corresponding to the main tables from the paper, use the following command(s):

For atomic concept evaluations:

    ```bash
    python3 generate.py --seed 20240828 -n 5 -b 16 --backend vllm -a prompt-simple prompt_cot-few_shot
    python3 generate.py --seed 20240828 -n 1 -b 16 -M GPT-4o-U GPT-3.5-U -a model-edit_repe
    python3 generate.py --seed 20240828 -n 1 -b 16 -M GPT-4o-U GPT-3.5-U -a tuning-sft tuning-sft-dpo
    python3 evaluate.py
    ```

For compositions of concepts based evaluations:

    ```bash
    python3 generate.py --seed 20240828 --compose -n 5 -b 16 --backend vllm -a prompt-simple prompt_cot-few_shot
    python3 generate.py --seed 20240828 --compose -n 1 -b 16 -M GPT-4o-U GPT-3.5-U -a model-edit_repe
    python3 generate.py --seed 20240828 --compose -n 1 -b 16 -M GPT-4o-U GPT-3.5-U -a tuning-sft tuning-sft-dpo
    python3 evaluate.py --compose
    ```

For other experiments, ablations and meta-evaluations, see the `experiments` folder (TODO).

## Dataset

`SELECT` is divided into the following files:

- `taxonomy*.json`: This lists out the relations and connections between concepts as a tree/DAG hierarchy.
- `data.json`: This contains the list of queries for each concept in a flattened-out list, but also has fields to infer the path information. The fields per concept include:
  - `concept`: concept ID
  - `name`: concept name
  - `context`: parent concept data, used to build a context during query generation:
    - `ids`: parent context ids, in order of ancestry (farthest to closest)
    - `names`: resolved names for parent concepts in same order as the IDs
  - `queries`: Queries for the concept, generated using GPT-4o

Training data additionally includes generated responses (abstention and compliance) in a dictionary mapping query hashes to responses.

The composition of concepts has a slightly different file structure, but the file names serve the same purpose. For instance, `data.compose.json` lists out queries for compositions, `example_cache.compose.json` lists out examples, etc.

- `data.json`: This contains queries for each composition in a flattened-out list, grouped by relation in a dictionary. As with atomic concepts, information to infer some the path information is included. The fields per composition include:
  - `concept`: composition ID, concatenation of underlying atomic concept IDs
  - `name`: composition name
  - `relation`: Relation template the composition was derived from
  - `compositions`: composition data about constituents:
    - `ids`: included concept IDs, in the order of the template
    - `names`: resolved names for concepts in same order as the IDs
  - `context`: parent concept data. This is only a placeholder for compatibility with the file structure for atomic concepts
  - `queries`: Queries for the concept, generated using GPT-4o

## Extending the Evaluation of `SELECT`

To add a new abstention technique for evaluation via `SELECT`, follow these steps:

1. Create a new class that extends `src.approaches.base.AbstentionTechnique`. This abstract class provides the base API for all techniques. Specifically, ensure the new class implements the `prepare()` and `generate()` methods. Additionally, provide metadata such as the name of the technique in the call of the `super().__init__()`. For example:

```python
# file: custom_abst.py

from src.approaches.base import AbstentionTechnique

class MyCustomAbstentionTechnique(AbstentionTechnique):
    def __init__(self):
        super().__init__(
            nature='CUSTOM',
            name='Custom Technique',
            short_name='C. Tech.',
            # this is not automatically used by default
            instruction=None,
            # See templates/prompts.yaml for options
            template='no_instruction'
        )

    def prepare(self, ...):
        # Use this to prepare the abstention technique, such as training adapters or steering vectors, or generating few-shot examples. This is run only once per model.
        ...

    def generate(self, ...):
        # Actual generation code, specific to a batch of prompts, goes here. Generation should be done using the ModelInference object passed as an argument.
        ...

```

2. In the main script (`{generate,evaluate}.py`), link an instance of the abstention technique to the shared `APPROACHES` object. Following this step, the technique can be referred when invoking the script:

```python
from src.approaches import APPROACHES

from custom_abst import MyCustomAbstentionTechnique

...

def main():
    APPROACHES['abst-custom'] = MyCustomAbstentionTechnique()

    parser = make_parser()
    ...
```

3. Evaluate models with the added abstention technique(s):

```bash
python3 generate.py -m Gemma-2-IT-2B -a abst-custom -n 1 -b 32
python3 evaluate.py -m Gemma-2-IT-2B -a abst-custom
```