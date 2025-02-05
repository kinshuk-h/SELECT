import dotenv
dotenv.load_dotenv()

from src.utils import common, io
from src.evaluation import dataset, task
from src.techniques import get_techniques

MODELS = io.read_yaml("config/models.yaml")
TYPES  = io.read_yaml("config/types.yaml")

def main():
    parser = common.make_parser('generate', MODELS, get_techniques(), TYPES)
    args   = parser.parse_args()

    kwargs = dict(
        seed         = args.seed,
        num_seeds    = args.num_seeds,
        sample       = args.sample,
        batch_size   = args.batch_size,
        save_steps   = args.save_every,
        models       = common.gather(MODELS, (set(args.models) - set(args.exclude_models or []))),
        techniques   = common.gather(get_techniques(), args.techniques)
    )

    dataset_dtype = 'compose' if args.compose else 'atom'
    dtype_suffix  = '' if dataset_dtype == 'atom' else f'.{dataset_dtype}'

    with common.LogTime(f"Loading primary dataset state ({dataset_dtype})", verbose=False):
        kwargs['dataset_state'] = dataset.DatasetState.from_name(args.dataset, dataset_dtype)

    if args.compose:
        with common.LogTime(f"Loading dataset state (atom)", verbose=False):
            kwargs['atomic_ds_state'] = dataset.DatasetState.from_name(args.dataset, 'atom')


    results_dir = args.root_dir / f"expr.abstain.select{dtype_suffix}"
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