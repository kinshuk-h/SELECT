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
    - `stats.concepts.json`: Frequency statistics for concepts, derived from [WIMBD](https://wimdb.allen.ai) across different corpora.
    - `hints.compose.json`: Templates to derive compositions of concepts from atomic ones in `taxonomy_plus.json`.
    - `*.compose.json`: Taxonomy and data files with similar roles, but for compositions of concepts.

- `results/expr.abstain.select`: Abstention experiment results

- `src`: Source code for experiment related utility modules.
  - `evaluation`: Module for utilities related to evaluation.
    - `experiments`: Submodule for experiment specific utilities.
      - `refusal.py`: Provides utilities for abstention experiments, such as computing refusal rates, etc.
    - `models`: Wraps logic for inference using HuggingFace and OpenAI models in a shared class for easy prototyping.
  - `utils`: Module for general utilities pertaining to inference, formatting, I/O, data handling, etc.
- `method_eval_utils.py`: Functions and classes specifically related to the prompting method evaluation scripts.

## Notebooks and Scripts

- `eval.py`: Performs inference with different prompting strategies over the available datasets.

  **Examples**:
  - Run inference using LLaMa-3-Chat-8B over the YAGO-derived dataset using few-shot example based prompting to assess refusal rates and in-dataset specificity:
    ```bash
    python3 method_eval.py -t direct specific -m LLaMa-3-Chat-8B -a prompt_few_shot-simple
    ```

  **Replication**:
   To replicate experiments from the paper, use the following command:
   ```bash

   ```

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

- `data.json`: This contains queries for each concept in a flattened-out list, grouped by relation in a dictionary. As with atomic concepts, information to infer some the path information is included. The fields per composition include:
  - `concept`: composition ID, concatenation of underlying atomic concept IDs
  - `name`: composition name
  - `relation`: Relation template the composition was derived from
  - `compositions`: composition data about constituents:
    - `ids`: included concept IDs, in the order of the template
    - `names`: resolved names for concepts in same order as the IDs
  - `context`: parent concept data. This is only a placeholder for compatibility with the file structure for atomic concepts.
  - `queries`: Queries for the concept, generated using GPT-4o