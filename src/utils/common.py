import sys
import timeit

from .formatting import format_time

def gather(collection, entries):
    if isinstance(collection, dict):
        return { key: value for key, value in collection.items() if key in entries }
    else:
        return [ value for value in collection if value in entries ]

def batchify(*lists, batch_size=8):
    """ Creates batches jointly across lists of iterables.

    Args:
        *lists (list): Lists to batch. If these are of unequal sizes, the minimum size is used.
        batch_size (int, optional): Batch size. Defaults to 8.

    Yields:
        tuple[*list]: Tuple of batches.
    """
    max_len = min(map(len, lists))
    for ndx in range(0, max_len, batch_size):
        yield tuple(
            lst[ndx:min(ndx + batch_size, max_len)]
            for lst in lists
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
                'eta': format_time(self.timer.avg_time * self.remaining, 'eta')
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