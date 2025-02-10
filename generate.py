import dotenv
dotenv.load_dotenv()

from src.utils import common, io
from src.evaluation import dataset, task
from src.techniques import get_techniques

MODELS = io.read_yaml("config/models.yaml")
TYPES  = io.read_yaml("config/types.yaml")

DEFAULT_CONFIG = dict(
    common_gen_kwargs=dict(
        temperature=0.6, top_p=0.9, do_sample=True,
        chat=True, max_new_tokens=512
    )
)

def main():
    parser = common.make_parser('generate', MODELS, get_techniques(), TYPES)
    args   = parser.parse_args()

    models      = [ MODELS.get(model, model) for model in sorted(args.models or MODELS) ]
    excl_models = set(MODELS.get(model, model) for model in args.exclude_models)
    resolved_models = [ model for model in models if model not in excl_models ]

    if args.config:
        yaml_file  = any(ext == args.config.suffixes[-1].lower() for ext in ('.yml', '.yaml'))
        read_fn    = io.read_yaml if yaml_file else io.read_json
        run_config = read_fn(args.config)
    else:
        run_config = DEFAULT_CONFIG

    kwargs = dict(
        seed=args.seed, num_seeds=args.num_seeds, sample=args.sample,
        batch_size=args.batch_size, save_steps=args.save_every,
        models=list(dict.fromkeys(resolved_models)),
        techniques=common.gather(get_techniques(), args.approaches)
    )

    dataset_dtype = 'compose' if args.compose else 'atom'
    dtype_suffix  = '' if dataset_dtype == 'atom' else f'.{dataset_dtype}'

    with common.LogTime(f"Loading primary dataset state ({dataset_dtype})", verbose=False):
        kwargs['dataset_state'] = dataset.DatasetState('select', args.data_dir, compose=args.compose)

    if args.compose:
        with common.LogTime(f"Loading dataset state (atom)", verbose=False):
            kwargs['atomic_ds_state'] = dataset.DatasetState('select', args.data_dir, compose=False)

    results_dir = args.root_dir / f"expr.abstain.select{dtype_suffix}"
    evaluator = task.Evaluator(root=results_dir, **kwargs)

    print("-" * 100, end='\n\n')

    print("> MODELS    : ", list(evaluator.models))
    print("> APPROACHES: ", list(evaluator.techniques))
    print("> RUN CONFIG:"); io.jprint(run_config)

    print("-" * 100, end='\n\n')

    eval_types = TYPES[dataset_dtype]
    if args.types: eval_types = { name: id for name, id in eval_types.items() if id in args.types }

    for type_name, eval_type in eval_types.items():
        print("Computing", type_name.lower(), "results ...")
        print("-" * 100, end='\n\n')
        evaluator.run(
            eval_type, preview=args.preview,
            backend=args.backend, config=run_config,
            concepts_filter=args.concepts
        )
        print()
        print("=" * 100, end='\n\n')

if __name__ == "__main__":
    main()