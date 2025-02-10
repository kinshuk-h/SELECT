# Knowledge Graph Guided Evaluation of Abstention Techniques

This repository provides the code and benchmark data for the paper: ["Knowledge Graph Guided Evaluation of Abstention Techniques"](https://arxiv.org/abs.2412.07430).

The repository is organized as follows:

- `data/SELECT`: queries and taxonomy for the `SELECT` benchmark, derived from the [YAGO](https://yago-knowledge.org) [taxonomy](https://yago-knowledge.org/data/yago4.5/design-document.pdf). Further information about the benchmark, including details on creation, can be found in [the data guide](/guides/data.md).

- `results/expr.abstain.select`: Abstention experiment results. See [the results guide](/guides/results.md) for further information on results and formats.

- `src`: Source code for experiment related utility modules.
  - `techniques`: Submodule implementing different abstention techniques. Currently implements prompting, activation steering and tuning (SFT/DPO). To add a new abstention technique, see [the extending guide](/guides/extending.md).
  - `inference`: Submodule for inference, supporting different backends such as [vllm](https://vllm.ai), [HuggingFace transformers](https://github.com/huggingface/transformers) and [OpenAI](https://platform.openai.com/api).
  - `evaluation`: Module for utilities related to evaluation.
    - `refusal.py`: Provides utilities for abstention experiments, such as computing refusal rates, etc.
    - `dataset.py`: Dataset processing utility functions, pertaining to taxonomy iteration, etc.
    - `task.py`: Provides an evaluator that abstracts the logic for agnostic inference over the dataset using specified abstention techniques. Any new registered techniques will be automatically supported.
  - `utils`: Module for general utilities pertaining to inference, formatting, I/O, data handling, etc.

## Notebooks and Scripts

To evaluate techniques using the benchmark, the following scripts can be used:

- `generate.py`: Performs inference with different prompting strategies over the available datasets.
- `evaluate.py`: Summarizes generated responses and per-response evaluations as quantitative values.

Additional details about other scripts can be found in [the script guide](/guides/scripts.md).

#### Setup

We use [conda](https://docs.anaconda.com/miniconda/install/) to manage dependencies for the project. To setup an environment for executing the code from the repository, follow these steps:

1. Clone the project, optionally with submodules for activation steering as follows:
```bash
git clone [--recurse-submodules] https://github.com/kinshuk-h/SELECT
cd SELECT
```

2. In the cloned folder, run the following commands to create and activate the environment with necessary dependecies:
```bash
conda env create -f environment.yml -n select
conda activate select
```

3. [Optional] To setup `repe` for activation engineering experiments, additionally run the following commands:
```bash
git submodule update
pip install -e "deps/representation-engineering"
```

4. Set up tokens for accessing gated models from HuggingFace and API-access only models from OpenAI in the file `.env`, to be created in the project root folder.
   Details about the names for environment variables can be found in [`.env.example`](/.env.example).

#### Quickstart

Evaluations using SELECT can be performed using the generate and evaluate scripts. For example:

- Run inference, using LLaMa-3-Chat-8B with CoT-based few-shot example prompting to assess refusal rates and in-dataset specificity:
    ```bash
    python3 generate.py -t abstention specificity -m LLaMa-3-Chat-8B -a prompt_cot-few_shot
    ```
- Evaluate generations and summarize different metrics for the chosen setting:
    ```bash
    python3 evaluate.py -t abstention specificity -m LLaMa-3-Chat-8B -a prompt_cot-few_shot
    ```

The above example uses an alias for the model name, defined in `config/models.yml`.
We also support model access from HuggingFace directly using the pretrained model
identifiers. For example, the same generate command can be rewritten as:

```bash
    python3 generate.py -t abstention specificity -m meta-llama/Meta-Llama-3.1-8B-Instruct -a prompt_cot-few_shot
```

Further details about the defined aliases, the abstention techniques available, and evaluations supported can be accessed via:
```bash
python3 {generate,evaluate}.py --help
```

#### Replication

To reproduce the results for the experiments described in the paper, refer [the scripts guide](/guides/scripts.md).

## Extending the Evaluation of `SELECT`

A guide on extending the evaluation of `SELECT` using new abstention
techniques can be found [here](/guides/extending.md).

## `lm-evaluation-harness` Integration

We plan to integrate evaluation for SELECT as part of the [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness).
Details about this will be updated soon!

## Citation

If you find this work useful for your research, please cite it as follows:

```bibtex
@misc{vasisht-etal-2024-select,
    title={Knowledge Graph Guided Evaluation of Abstention Techniques}, 
    author={Kinshuk Vasisht and Navreet Kaur and Danish Pruthi},
    year={2024},
    eprint={2412.07430},
    archivePrefix={arXiv},
    primaryClass={cs.CL},
    url={https://arxiv.org/abs/2412.07430},
}
```

## Contact

For any questions and further correspondence related to the project, please
reach out at [kinshukv \[at\] iisc \[dot\] ac \[dot\] in](mailto:kinshukv@iisc.ac.in). Thanks!
