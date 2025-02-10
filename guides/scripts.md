## Utility Scripts

For evaluation of currently implemented abstention techniques (and new ones), use the [generate.py](/generate.py) and [evaluate.py](/evaluate.py) scripts.

#### Customizing Execution Runs

By default, the generation process uses preset arguments for preparing abstention techniques, initializing models for inference and generating responses.
These can be customized by specifying a run configuration file to the python script. `.json` and `.yml` files are supported.

The structure of a valid configuration file is as follows:

```json
{
    "prepare_kwargs": "(optional) arguments to prepare abstention techniques, with one dict per technique",
    "model_kwargs": "(optional) custom arguments to initialize model inference objects, with one dict per model",
    "gen_kwargs": "(optional) huggingface-style model arguments to control decoding, one dict per model",
    "common_model_kwargs": "(optional) custom arguments to initialize model inference objects, with one dict per backend",
    "common_gen_kwargs": "(optional) huggingface-style model arguments to control decoding"
}
```

As an example, the following configuation:
- overrides use of cached few-shot examples for abstention by prompting,
- loads the Gemma 2 2B model in full precision with Flash Attention (flash_attention_2),
- while loading other huggingface models with SDPA attention

```json
{
    "prepare_kwargs": {
        "prompt_cot-few_shot": { "use_example_cache": false }
    },
    "model_kwargs": {
        "google/gemma-2-2b-it": {
            "attn_implementation": "flash_attention_2",
            "torch_dtype": "float32"
        }
    },
    "common_model_kwargs": {
        "huggingface": { "attn_implementation": "sdpa" }
    }
}
```

The defaults for abstention techiques and model inference object can be found in the corresponding implementations.

#### Generating the `SELECT` Benchmark

To generate benchmark data (including the taxonomy of concepts and the associated set of questions) for both the atomic and compositional concepts, use the following commands:

```bash
# create the taxonomy of atomic concepts
PYTHONPATH=. python3 scripts/data/create_taxonomy.py

# create the partition with atomic concepts
PYTHONPATH=. python3 scripts/data/create_atomic_select.py

# create the partition with compositions
PYTHONPATH=. python3 scripts/data/create_compositions.py
```

#### Reproducing Results for the Main Experiments

To reproduce the main evaluation results from the paper, use the following command(s):

For atomic concept evaluations:

```bash
python3 generate.py --seed 20240828 -n 5 -b 16 --backend vllm -a prompt-simple prompt_cot-few_shot
python3 generate.py --seed 20240828 -n 1 -b 16 -M GPT-4o-U GPT-3.5-U -a steering-repe
python3 generate.py --seed 20240828 -n 1 -b 16 -M GPT-4o-U GPT-3.5-U -a tuning-sft tuning-sft-dpo
python3 evaluate.py
```

For composition evaluations:

```bash
python3 generate.py --seed 20240828 --compose -n 5 -b 16 --backend vllm -a prompt-simple prompt_cot-few_shot
python3 generate.py --seed 20240828 --compose -n 1 -b 16 -M GPT-4o-U GPT-3.5-U -a steering-repe
python3 generate.py --seed 20240828 --compose -n 1 -b 16 -M GPT-4o-U GPT-3.5-U -a tuning-sft tuning-sft-dpo
python3 evaluate.py --compose
```

#### Miscellaneous Actions

Some other scripts for various operations are listed as follows:

- Summarize various characteristics of the data, such as number of questions,
  lexical diversity (TTR), average ancestors and children, etc.

    ```bash
    python3 -m pip install nltk # for lexical diversity
    PYTHONPATH=. python3 scripts/data/summarize_stats.py
    ```