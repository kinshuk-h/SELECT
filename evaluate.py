import pathlib

import numpy
import scipy
import pandas

from src.utils import io, common, data
from src.techniques import get_techniques
from src.evaluation import refusal, dataset

MODELS = io.read_yaml("config/models.yaml")
TYPES  = io.read_yaml("config/types.yaml")

def compute_confidence_interval(data: numpy.ndarray, percentage=0.95):
    return numpy.array([
        scipy.stats.t.interval(
            percentage, len(row)-1, loc=row.mean(), scale=scipy.stats.sem(row)
        )[1] - row.mean()
        for row in data
    ])

def get_result(root: pathlib.Path, model, eval_type, approach):
    result_path = root / model / eval_type / f"{approach}.json"
    if result_path.exists(): return data.NestedListItemResult(result_path)[eval_type]

def main():
    parser = common.make_parser('evaluate', MODELS, get_techniques(), TYPES)
    args   = parser.parse_args()

    dataset_dtype = 'compose' if args.compose else 'atom'
    dtype_suffix  = '' if dataset_dtype == 'atom' else f'.{dataset_dtype}'

    models      = [ MODELS.get(model, model) for model in (args.models or MODELS) ]
    excl_models = set(MODELS.get(model, model) for model in args.exclude_models)
    resolved_models = [ model for model in models if model not in excl_models ]

    models     = list(dict.fromkeys(resolved_models))
    approaches = common.gather(get_techniques(), args.approaches)
    eval_types = TYPES[dataset_dtype]
    if args.types: eval_types = { name: id for name, id in eval_types.items() if id in args.types }

    APPR_MAP = { appr: appr_inst.short_name for appr, appr_inst in get_techniques().items() }
    TYPE_MAP = { eval_type: eval_name for eval_name, eval_type in TYPES[dataset_dtype].items() }

    root_path = args.root_dir / f"expr.abstain.select{dtype_suffix}"

    # load all results
    all_results = {
        model_key: {
            approach: {
                eval_type: get_result(root_path, model_key, eval_type, approach)
                for _, eval_type in eval_types.items()
            }
            for approach in approaches
        }
        for model_key in models
    }

    # aggregate over seeds
    collated_results = {
        model_key: {
            eval_type: {
                approach: refusal.refusal_rate(
                    all_results[model_key][approach][eval_type],
                    list(refusal.EVALUATION_METHODS),
                    negate=('specific' in eval_type), average_over_seeds=True
                )
                for approach in approaches
                if all_results[model_key][approach][eval_type]
            }
            for _, eval_type in eval_types.items()
        }
        for model_key in models
    }

    def aggregate_refusal_trends(results, refusal_method):
        model_res_data = {}

        for approach in approaches:
            values = [
                numpy.mean([
                    c_val[refusal_method]
                    for _, c_val in dataset.concept_iterator(
                        results[eval_type][approach],
                        compose=args.compose
                    )
                ])
                for eval_type in results
            ]
            errors = compute_confidence_interval(numpy.array([
                [
                    c_val[refusal_method]
                    for _, c_val in dataset.concept_iterator(
                        results[eval_type][approach],
                        compose=args.compose
                    )
                ]
                for eval_type in results
            ]))

            model_res_data[approach] = {
                eval_type: ( f"{value:.2%}", f"{error:.1%}" )
                for eval_type, value, error in zip(results, values, errors)
            }

        return model_res_data

    # aggregate further over concepts
    aggregates = {
        model_key: aggregate_refusal_trends(collated_results[model_key], args.method)
        for model_key in models
    }

    # create a table from aggregates and show it
    records = []
    for model, model_data in aggregates.items():
        record = dict(Model = model)
        for approach, appr_data in model_data.items():
            for eval_type, (mean, ci) in appr_data.items():
                record[f"{APPR_MAP[approach]} - {TYPE_MAP[eval_type]}"] = f"{mean} +- {ci}"
        records.append(record)

    print(pandas.DataFrame.from_records(records))

if __name__ == "__main__":
    main()