import argparse

import dotenv
dotenv.load_dotenv()

from src.utils import common, io
from src.approaches import APPROACHES
from src.evaluation import dataset, task

MODELS = io.read_yaml("config/models.yaml")
TYPES  = io.read_yaml("config/types.yaml")

def make_parser():

    parser = argparse.ArgumentParser(
        description="generates responses for abstention with different prompting methods"
    )

    parser.add_argument("-r", "--root-dir", type=str, default="results",
                        help="root directory to save intermediate results")
    parser.add_argument("-m", "--models", nargs="+", default=set(MODELS.keys()),
                        help="models to evaluate with")
    parser.add_argument("-M", "--exclude-models", nargs="+", default=[],
                        help="models to evaluate with")
    parser.add_argument("--backend", type=str, default='huggingface', choices=[ 'huggingface', 'openai', 'vllm' ],
                        help="backend to run inference with. Defaults to huggingface/openai.")
    parser.add_argument("-a", "--approaches", nargs="+", default=set(APPROACHES.keys()),
                        help="abstention techniques to test out")
    parser.add_argument("-x", "--preview", action="store_true",
                        help="dummy run the code, with no actual generations or output updates")
    parser.add_argument("-t", "--types", nargs="+", default=None,
                        help="type(s) of results to compute")
    parser.add_argument("-b", "--batch-size", type=int, default=16,
                        help="batch size for inference")
    parser.add_argument("-S", "--no-sample", dest="sample", action="store_false",
                        help="run on the entire partitions, do not select samples")
    parser.add_argument("-s", "--save-every", type=int, default=2,
                        help="number of batches processed to save after")
    parser.add_argument('-c', "--compose", action="store_true",
                        help="whether to use the compositional subset of the dataset")
    parser.add_argument('--seed', type=int, help="global random seed to use for determinism")
    parser.add_argument('-n', '--num-seeds', type=int, default=5,
                        help="number of seeds (determines replications to run)")

    return parser

def main():
    parser = make_parser()
    args   = parser.parse_args()

    kwargs = dict(
        seed         = args.seed,
        num_seeds    = args.num_seeds,
        sample       = args.sample,
        eval_dataset = None,
        batch_size   = args.batch_size,
        save_steps   = args.save_every,
        models       = common.gather(MODELS, (set(args.models) - set(args.exclude_models or []))),
        approaches   = common.gather(APPROACHES, args.approaches)
    )

    dataset_dtype = 'compose' if args.compose else 'atom'
    dtype_suffix  = '' if dataset_dtype == 'atom' else f'.{dataset_dtype}'

    args.dataset = 'select'

    with common.LogTime(f"Loading primary dataset state ({dataset_dtype})", verbose=False):
        kwargs['dataset_state'] = dataset.DatasetState.from_name(args.dataset, dataset_dtype)

    if args.compose:
        with common.LogTime(f"Loading primary dataset state (atom)", verbose=False):
            kwargs['atomic_ds_state'] = dataset.DatasetState.from_name(args.dataset, 'atom')

    # with common.LogTime("Loading evaluation dataset (tatsu-lab/alpaca)", verbose=False):
    #     eval_dataset = dataset.get_eval_dataset()

    results_dir = f"{args.root_dir}/expr.abstain.{args.dataset}{dtype_suffix}"
    evaluator = task.Evaluator(root=results_dir, **kwargs)

    print("-" * 100, end='\n\n')

    print("> MODELS    : ", list(evaluator.models))
    print("> APPROACHES: ", list(evaluator.approaches))
    print("-" * 100, end='\n\n')

    eval_types = TYPES[dataset_dtype]
    if args.types: eval_types = { name: id for name, id in eval_types.items() if id in args.types }

    for type_name, eval_type in eval_types.items():
        print("Computing", type_name, "results ...")
        print("-" * 100, end='\n\n')
        evaluator.run(eval_type, preview=args.preview, backend=args.backend)
        print()
        print("=" * 100, end='\n\n')

if __name__ == "__main__":
    main()