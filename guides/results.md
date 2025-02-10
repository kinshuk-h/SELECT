## Results

Evaluation results for SELECT are cached in GZipped JSON files. The results are by default stored in
`results/expr.abstain.select` or `results/expr.abstain.select.compose`, depending on evaluation with
atomic or compositions of concepts respectively.

The directory structure for a results file is as follows - `{model}/{evaluation_type}/{technique_alias}.json.gz`:
- `{model}`: Pathsafe model name, such as google--gemma-2-9b-it
- `{evaluation_type}`: Currently, one of `abstention`, `generalization` or `specificity`. These correspond to metrics.
- `{technique_alias}`: Alias for the abstention technique. For a list of registered aliases, use `python3 generate.py --help`.

#### Content Organization

Each result file is a nested dictionary with following levels - `{seed}/{concept}/{instance_id}`:
- `{seed}`: Random seed used during this specific run of results.
- `{concept}`: Atomic concept being abstained from. This is the YAGO specific ID for the concept.
- `{instance_id}`: MD5 hash of the input prompt without any instructions or additions.

In case of compositions, the organization is different - `{seed}/{relation}/{concepts}/{instance_id}`:
- `{seed}`: Random seed used during this specific run of results.
- `{relation}`: Composition group which is being targeted. This conforms to a template, and is populated as described in the paper.
- `{concepts}`: Specific instances for the relation being abstained from. This is the set of YAGO specific IDs for the concepts.
- `{instance_id}`: MD5 hash of the input prompt without any instructions or additions.

For each instance we evaluate, we save the following JSON record:
```json
{
    "id": "instance specific ID",
    "concept": "list of concept(s) that comprise the target concept. Has more than one element only in case of compositions.",
    "response": "Model response. In case of CoT this is a dictionary of raw and formatted responses.",
    "refusal": {
        "phrase": "Whether the response includes an instructed phrase that signifies refusal.",
        "keyword": "Whether the response includes keywords that signify refusal.",
        "phr+kw": "Whether the response passes either of the phrase or keyword checks.",
        "heuristic": "Whether the response passes the heuristic check."
    }
}
```

#### Additional Experiments

In addition to results for the primary evaluation, we include results for other experiments. These include:
- We evaluate how abstention rates for descendants change as a function of the path distance. To get a better handle of the variance across path lengths, we cannot use existing generalization results. This is as by default we sample a subset of questions which may not correspond to all descendants of a concept. Thus, we disable sampling (`--no-sample` flag in `generate.py`) for this experiment, and save the results in `results/extended/expr.abstain.select/generalization`.
- We evaluate abstention in case of prompts with adversarial perturbations. These results are saved in `results/adversarial/expr.abstain.select`.
- We evaluate the models involved in terms of their ability to understand the hierarchy of concepts. This is aimed to understand the underlying causes for generalization errors. Results of such pairwise concept evaluations over children and siblings are stored in `results/expr.concept.understanding`. Each file corresponding to a model includes a mapping of concept pairs to a record that includes the actual relation (in the `related` key) versus the prediction (in the `predicted` key).