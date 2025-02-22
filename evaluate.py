import pathlib
import itertools

import numpy
import pandas
import scipy.stats
import tqdm.auto as tqdm

from src.utils import io, common
from src.techniques import get_techniques
from src.evaluation import refusal, dataset

MODELS = io.read_yaml("config/models.yaml")
TYPES  = io.read_yaml("config/types.yaml")

def get_mean_and_ci(data: numpy.ndarray, percentage=0.95):
    if not isinstance(data, numpy.ndarray): data = numpy.array(data)
    loc, scale = data.mean(), scipy.stats.sem(data)
    interval = scipy.stats.t.interval(percentage, len(data)-1, loc=loc, scale=scale)
    return loc, (interval[1] - loc)

def load_result(root: pathlib.Path):
    def load(eval_type, model, technique):
        model = io.pathsafe(model.lower().replace('/', '--'))
        result_path = root / model / eval_type / f"{technique}.json.gz"
        if result_path.exists(): return eval_type, io.Result(result_path)
    return load

def compute_refusal(eval_type, result):
    return refusal.refusal_rate(
        result, list(refusal.EVALUATION_METHODS),
        negate=('specific' in eval_type), average_over_seeds=True
    )

def summarize_metric(refusal_method, compose=False):
    def summarize(result):
        values = [ c[refusal_method] for _, c in dataset.concept_iterator(result, compose=compose) ]
        mean, ci = get_mean_and_ci(values); return (f"{mean:.2%}", f"{ci:.1%}")
    return summarize

def pipe(*callables):
    def pipeline(*args):
        result = args
        for callable in callables:
            if not isinstance(result, (list, tuple)): result = (result,)
            if (result := callable(*result)) is None: break
        return result
    return pipeline

def main():
    parser = common.make_parser('evaluate', MODELS, get_techniques(), TYPES)
    args   = parser.parse_args()

    dataset_dtype = 'compose' if args.compose else 'atom'
    dtype_suffix  = '' if dataset_dtype == 'atom' else f'.{dataset_dtype}'

    MODEL_ALIASES = { name: alias for alias, name in MODELS.items() }
    models      = [ MODELS.get(model, model) for model in sorted(args.models or MODELS) ]
    excl_models = set(MODELS.get(model, model) for model in args.exclude_models)
    resolved_models = [ model for model in models if model not in excl_models ]

    models     = list(dict.fromkeys(resolved_models))
    techniques = common.gather(get_techniques(), args.approaches)
    eval_types = TYPES[dataset_dtype]

    if args.types: eval_types = { name: id for name, id in eval_types.items() if id in args.types }

    processor = pipe(
        # load the result file
        load_result(args.root_dir / f"expr.abstain.select{dtype_suffix}"),
        # aggregate over seeds to compute refusal (or anti-refusal) rates
        compute_refusal,
        # aggregate further over concepts to get mean and confidence interval
        summarize_metric(args.method, args.compose)
    )

    runs = list(itertools.product(eval_types.values(), models, techniques))
    aggregates = { run_tuple: processor(*run_tuple) for run_tuple in tqdm.tqdm(runs) }

    print('-' * 100, end='\n\n')

    # create a table from aggregates and show it
    for eval_name, eval_type in eval_types.items():

        print(eval_name, "results:", end='\n\n')

        records = []
        for model in models:
            record = dict(Model = MODEL_ALIASES.get(model, model))
            for technique in techniques:
                agg = aggregates.get((eval_type, model, technique))
                agg_summ = f"{agg[0]} +- {agg[1]}" if agg is not None else "???"
                record[techniques[technique].short_name] = agg_summ
            records.append(record)

        print(pandas.DataFrame.from_records(records), end='\n\n')
        print('=' * 100, end='\n\n')

if __name__ == "__main__":
    main()