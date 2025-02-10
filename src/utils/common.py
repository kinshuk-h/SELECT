import sys
import timeit
import pathlib
import argparse

from torch.utils.data import DataLoader

from .formatting import format_time

class SELECTEvaluationArgumentParser(argparse.ArgumentParser):
    TASK_DESCRIPTIONS = {
        'generate': "generates responses for abstention with different prompting methods",
        "evaluate": "summarizes evaluations for abstention with different prompting methods"
    }

    def __init__(self, task, model_aliases, techniques, types, *args, **kwargs):
        kwargs['description'] = kwargs.get('description', self.TASK_DESCRIPTIONS.get(task))
        super().__init__(*args, **kwargs)

        self.task = task
        self.model_aliases = model_aliases
        self.techniques = techniques
        self.types = types

    def print_help(self, file = None):
        super().print_help(file)

        _print = file.write if file else sys.stdout.write

        _print('\n')

        _print("supported abstention techniques (use with -a, --approaches):\n")
        width = max(len(name) for name in self.techniques)
        for tech_name, technique in self.techniques.items():
            _print(f"  {tech_name:<{width}}  {technique.nature}: {technique.name}\n")
        _print("\n")

        _print("supported evaluations (based on metrics) (use with -t, --types):\n")
        for category, cat_types in self.types.items():
            _print(f"  {'atoms' if category == 'atom' else 'compositions (w/ -c, --compose)'}\n")
            _print(f"    {', '.join(cat_types.values())}")
        _print("\n")

        _print("registered model aliases (use with -m, --models, -M, --exclude-models):\n")
        width = max(len(alias) for alias in self.model_aliases)
        for model_alias, model_name in self.model_aliases.items():
            _print(f"  {model_alias:<{width}}  {model_name}\n")
        _print("\n")

def make_parser(task, model_aliases: dict, techniques: dict, types: dict):
    parser = SELECTEvaluationArgumentParser(task, model_aliases, techniques, types)

    parser.add_argument("--data-dir", default="data", type=pathlib.Path,
                        help="root directory to load data from")
    parser.add_argument("-r", "--root-dir", default="results", type=pathlib.Path,
                        help="root directory to save generation results")

    parser.add_argument("-m", "--models", nargs="+", default=set(model_aliases.keys()),
                        help="models to evaluate with, defaulting to the list defined in config/models.yaml")
    parser.add_argument("-M", "--exclude-models", nargs="+", default=[],
                        help="models to exclude from evaluation")
    parser.add_argument("-a", "--approaches", "--techniques", nargs="+", default=set(techniques.keys()),
                        help="abstention techniques to test out, defaulting to those listed in config/techniques.yaml")
    parser.add_argument("-t", "--types", nargs="+", default=None,
                        help=("type(s) of results to compute (correspond to metrics). "
                              "Defaults to the list in config/types.yaml."))

    parser.add_argument('-c', "--compose", action="store_true",
                        help="whether to use the compositional subset of the dataset")

    parser.add_argument("-S", "--no-sample", dest="sample", action="store_false",
                        help="run on the entire partitions, do not select samples")

    if task == 'generate':
        parser.add_argument("--backend", type=str, default='huggingface', choices=[ 'huggingface', 'openai', 'vllm' ],
                            help="backend to run inference with, defaulting to huggingface/openai.")

        parser.add_argument('--seed', type=int, help="global random seed to use for determinism")
        parser.add_argument('-n', '--num-seeds', type=int, default=5,
                            help="number of seeds (determines replications to run)")

        parser.add_argument("--config", type=pathlib.Path, default=None,
                            help="configuration file to override initialization parameters")
        parser.add_argument("--concepts", nargs="+", default=None,
                            help="specific names of concepts to process, or a colon delimited range")

        parser.add_argument("-b", "--batch-size", type=int, default=16,
                            help="batch size for inference, upscaled relative to model size")
        parser.add_argument("-s", "--save-every", type=int, default=2,
                            help="number of batches processed to save afterwards")
        parser.add_argument("-x", "--preview", action="store_true",
                            help="dry run the code, with no actual generations")

    if task == 'evaluate':
        parser.add_argument('--method', type=str, default='heuristic',
                            help='evaluation method to use for signifying refusal / abstention')

    return parser

def gather(collection, entries):
    if isinstance(collection, dict):
        return { key: value for key, value in collection.items() if key in entries }
    else:
        return [ value for value in collection if value in entries ]

def instance_collate(instances):
    element = instances[0]
    if isinstance(element, dict):
        return {
            key: [ inst[key] for inst in instances ]
            for key in element
        }
    elif isinstance(element, (list, tuple)):
        return tuple([
            instance_collate([ inst[i] for inst in instances ])
            for i in range(len(element))
        ])
    else:
        return instances

def batchify(*lists, batch_size=8):
    """ Creates batches jointly across lists of iterables.

    Args:
        *lists (list): Lists to batch. If these are of unequal sizes, the minimum size is used.
        batch_size (int, optional): Batch size. Defaults to 8.

    Yields:
        tuple[*list]: Tuple of batches.
    """
    return DataLoader(
        [ (data_tuple[0] if len(data_tuple) == 1 else data_tuple) for data_tuple in zip(*lists) ],
        batch_size=batch_size, shuffle=False, collate_fn=instance_collate
    )

class LogTime:
    """ Context manager to log the execution time taken by a code block. """

    def __init__(self, label, verbose=True) -> None:
        self.label = label or "execution completed"
        self.toc = None
        self.tic = None
        self.verbose = verbose

    def __enter__(self):
        if self.verbose: print("[<]", self.label, "...", end = '\n\n')
        self.tic = timeit.default_timer()

    def __exit__(self, exc_type, exc_val, exc_traceback):
        self.toc = timeit.default_timer()
        log_str = f"[>] {self.label}: {format_time(self.toc-self.tic, 'log')}"
        if self.verbose: print()
        print(log_str, file=sys.stderr)
        if self.verbose: print('-' * (len(log_str) + 20), file=sys.stderr)

        if exc_val is not None: raise exc_val

class BatchProgressTimer:
    """ Times execution for batch processing, synchronizing changes with a progress bar. """

    class Operation:
        """ Context manager for timing execution of a single batch. """

        def __init__(self, timer, batch, **postfix_kwargs):
            self.timer = timer
            self.postfix_kwargs = postfix_kwargs
            self.time = 0
            self.batch = batch
            self.remaining = self.timer.total - batch

        def __enter__(self):
            self.time = timeit.default_timer()

        def __exit__(self, exc_type, exc_val, exc_traceback):
            end_time = timeit.default_timer()
            self.timer.add_time(end_time - self.time)
            self.timer.progress_bar.set_postfix({
                **self.postfix_kwargs,
                'batch': self.batch, 'total': self.timer.total,
                'cur.': format_time(end_time - self.time, 'iter') + "/it",
                'avg.': format_time(self.timer.avg_time, 'iter') + "/it",
                'etc': format_time(self.timer.avg_time * self.remaining, 'eta')
            })

            if exc_val is not None: raise exc_val

    def __init__(self, progress_bar, total) -> None:
        self.progress_bar = progress_bar
        self.total = total
        self.avg_time = 0

    def timed_operation(self, batch, **postfix_kwargs):
        """ Creates a new Operation, given the batch details. """
        return self.Operation(self, batch, **postfix_kwargs)

    def add_time(self, time):
        """ Integrates execution time per batch into the running average. """
        if self.avg_time != 0:
            # Exponential Moving Average (w/o bias correction)
            self.avg_time = 0.9 * self.avg_time + 0.1 * time
        else:
            self.avg_time = time