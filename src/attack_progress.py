import argparse
import math


def add_progress_args(argparser, early_stop=False):
    """
    Add plateau options shared by both runners
    """
    argparser.add_argument('--early_stop', action=argparse.BooleanOptionalAction, default=early_stop,
                           help='stop when all epsilon versions have plateaued')
    argparser.add_argument('--min_iter', type=int, default=1500, help='minimum updates before early stopping')
    argparser.add_argument('--plateau_window', type=int, default=300, help='updates per convergence window')
    argparser.add_argument('--plateau_checks', type=int, default=3, help='consecutive flat windows required')
    argparser.add_argument('--plateau_threshold', type=float, default=0.01,
                           help='relative improvement threshold; 0.01 means 1 percent')


def validate_progress_args(args):
    """
    Check stopping options before loading the model
    """
    if args.num_iter < 1:
        raise ValueError('Number of iterations must be positive.')
    if args.min_iter < 0 or args.plateau_window < 1 or args.plateau_checks < 1:
        raise ValueError('Minimum iterations must be nonnegative; plateau window and checks must be positive.')
    if not math.isfinite(args.plateau_threshold) or not 0 < args.plateau_threshold < 1:
        raise ValueError('Plateau threshold must be a finite fraction between 0 and 1.')
    if getattr(args, 'beta_growth', False) and args.early_stop:
        raise ValueError('Beta growth changes the objective. Use --no-early_stop with it.')


class PlateauTracker:
    def __init__(self, batch_size, window=300, checks=3, threshold=0.01, min_iter=1500):
        if batch_size < 1 or window < 1 or checks < 1 or min_iter < 0:
            raise ValueError('Batch size, window and checks must be positive; minimum iterations must be nonnegative.')
        if not math.isfinite(threshold) or not 0 < threshold < 1:
            raise ValueError('Plateau threshold must be a finite fraction between 0 and 1.')
        self.window = window
        self.checks = checks
        self.threshold = threshold
        self.min_iter = min_iter
        self.iteration = 0
        self.best_losses = [float('inf')] * batch_size
        self.best_iterations = [0] * batch_size
        self.window_start = None
        self.flat_windows = [0] * batch_size

    @property
    def converged(self):
        return [self.iteration >= self.min_iter and count >= self.checks for count in self.flat_windows]

    def observe(self, iteration, losses):
        """
        Track every evaluated image and check nonoverlapping windows
        """
        if len(losses) != len(self.best_losses) or any(not math.isfinite(value) for value in losses):
            raise ValueError('Expected one finite total loss per epsilon.')
        self.iteration = iteration
        improved = [value < best for value, best in zip(losses, self.best_losses)]
        for index, changed in enumerate(improved):
            if changed:
                self.best_losses[index] = losses[index]
                self.best_iterations[index] = iteration

        if self.window_start is None:
            self.window_start = self.best_losses.copy()
        elif iteration % self.window == 0:
            for index, (before, after) in enumerate(zip(self.window_start, self.best_losses)):
                improvement = (before - after) / max(abs(before), 1e-30)
                self.flat_windows[index] = self.flat_windows[index] + 1 if improvement < self.threshold else 0
            self.window_start = self.best_losses.copy()
        else:
            # a meaningful improvement already rules out a flat current window
            for index, (before, after) in enumerate(zip(self.window_start, self.best_losses)):
                if (before - after) / max(abs(before), 1e-30) >= self.threshold:
                    self.flat_windows[index] = 0
        return improved

    def should_stop(self):
        return self.iteration % self.window == 0 and all(self.converged)
