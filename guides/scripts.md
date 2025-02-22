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

As another example, the following configuation overrides training parameters for abstention with SFT which applies to all models. Slightly different parameters are then set specifically for Gemma 2 2B.

```json
{
    "prepare_kwargs": {
        "tuning-sft": {
            "trainer_kwargs": {
                "common": {
                    "learning_rate": 5e-5,
                    "num_train_epochs": 5
                },
                "google/gemma-2-2b-it": {
                    "learning_rate": 5e-4,
                }
            }
        }
    }
}
```

Here, training arguments correspond to those supported by [`trl.SFTConfig`](https://huggingface.co/docs/trl/en/sft_trainer#trl.SFTConfig). The defaults for abstention techiques and model inference object in general can be found in the corresponding implementations.

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

#### Experiments

Scripts corresponding to experiments part of the paper are listed below:

- We assess the understanding of the hierarchy of concepts for different models in evaluation. This is to identify whether generalization errors result for models not able to understand the hierarchy of concepts. Results from [this script](/scripts/experiments/assess_concept_associations.py) show that such errors can explain only $~35$% of the errors on average.

    ```bash
    PYTHONPATH=. python3 scripts/experiments/assess_concept_associations.py
    ```

#### Miscellaneous Actions

Some other scripts for various operations are listed as follows:

- Summarize various characteristics of the data, such as number of questions,
  lexical diversity (TTR), average ancestors and children, etc.

    ```bash
    python3 -m pip install nltk # for lexical diversity
    PYTHONPATH=. python3 scripts/data/summarize_stats.py
    ```
