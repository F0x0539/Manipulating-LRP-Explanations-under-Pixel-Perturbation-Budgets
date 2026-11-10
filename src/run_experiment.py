import argparse
import csv
import json
from pathlib import Path
import time

import numpy as np
import torch

from attack_progress import add_progress_args, validate_progress_args
from dataset_images import resolve_image
from nn.enums import ExplainingMethod
from run_attack import LOSS_HISTORY_FIELDS, attack_pair_batch, calculate_metrics, load_model, save_results


ROOT = Path(__file__).resolve().parents[1]
EPSILONS = [1, 2, 4, 8, 16, 32, 255]


def parse_args(defaults=None):
    """
    Read experiment options
    """
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--manifest', type=Path, default=ROOT / 'data/imagenet/manifest.json',
                           help='manifest containing the fixed image pairs')
    argparser.add_argument('--output_dir', type=Path, default=ROOT / 'output/experiment-lrp',
                           help='new directory for this experiment')
    argparser.add_argument('--epsilons', nargs='+', type=float, default=EPSILONS,
                           help='budgets in brightness levels; 255 is unbounded')
    argparser.add_argument('--batch_size', type=int, default=2,
                           help='epsilon versions processed together')
    argparser.add_argument('--num_iter', type=int, default=1500, help='Adam updates; maximum with --early_stop')
    argparser.add_argument('--lr', type=float, default=0.0002, help='learning rate')
    argparser.add_argument('--log_every', type=int, default=50, help='steps between saved loss samples')
    argparser.add_argument('--prefactors', nargs=2, type=float, default=[1e11, 1e6],
                           help='prefactors of explanation and output losses')
    argparser.add_argument('--cuda', action='store_true', help='enable GPU mode')
    argparser.add_argument('--imagenet-dir', type=Path, help='directory containing the downloaded validation JPEGs')
    argparser.add_argument('--limit_pairs', type=int, help='use only the first N pairs for a pilot')
    argparser.add_argument('--dry_run', action='store_true', help='check pairs and print the plan only')
    add_progress_args(argparser)
    if defaults:
        argparser.set_defaults(**defaults)
    args = argparser.parse_args()
    try:
        validate_progress_args(args)
    except ValueError as error:
        argparser.error(str(error))

    # check experiment settings
    if args.batch_size < 1 or args.num_iter < 1:
        argparser.error('Batch size and number of iterations must be positive.')
    if args.log_every < 1:
        argparser.error('Logging interval must be positive.')
    if not np.isfinite(args.lr) or args.lr <= 0:
        argparser.error('Learning rate must be finite and positive.')
    if args.limit_pairs is not None and args.limit_pairs < 1:
        argparser.error('Pair limit must be positive.')
    if any(not np.isfinite(value) or not 0 <= value <= 255 for value in args.epsilons):
        argparser.error('Epsilon budgets must be finite values between 0 and 255.')
    if len(set(args.epsilons)) != len(args.epsilons):
        argparser.error('Epsilon budgets must not contain duplicates.')
    if any(not np.isfinite(value) or value < 0 for value in args.prefactors):
        argparser.error('Loss prefactors must be finite and nonnegative.')
    return args


def load_pairs(manifest_path, image_root=None):
    """
    Load fixed pairs and resolve image paths relative to the manifest
    """
    manifest_path = Path(manifest_path).expanduser().resolve()
    manifest = json.loads(manifest_path.read_text())
    pairs = manifest['pairs']
    if not pairs:
        raise ValueError('The manifest contains no image pairs.')

    # check pair identifiers and input images before loading the model
    pair_ids = set()
    resolved = []
    for pair in pairs:
        pair_id = pair['pair_id']
        if not isinstance(pair_id, int) or pair_id < 0 or pair_id in pair_ids:
            raise ValueError('Pair identifiers must be unique nonnegative integers.')
        pair_ids.add(pair_id)
        original = resolve_image(manifest_path.parent / pair['original'], image_root, manifest_path)
        target = resolve_image(manifest_path.parent / pair['target'], image_root, manifest_path)
        for image in (original, target):
            if not image.is_file():
                raise FileNotFoundError(f'Image not found: {image}')
        if original == target:
            raise ValueError(f'Pair {pair_id} uses the same image as original and target.')
        resolved.append({'pair_id': pair_id, 'original': str(original), 'target': str(target)})
    return resolved


def run_experiment(args):
    """
    Run epsilon batches for each fixed pair using one loaded model
    """
    pairs = load_pairs(args.manifest, getattr(args, 'imagenet_dir', None))
    if args.limit_pairs is not None:
        if args.limit_pairs > len(pairs):
            raise ValueError('Pair limit exceeds the number of pairs in the manifest.')
        pairs = pairs[:args.limit_pairs]
    output_dir = args.output_dir.expanduser().resolve()
    num_attacks = len(pairs) * len(args.epsilons)
    print(f'{len(pairs)} pairs x {len(args.epsilons)} settings = {num_attacks} attacks')
    print(f'Epsilons: {args.epsilons}; batch size: {args.batch_size}')
    print(f'Device: {"cuda" if args.cuda else "cpu"}; output: {output_dir}')
    selection = 'lowest-loss image (historical pilot)' if getattr(args, 'save_best', False) else 'final optimizer state'
    print(f'Saving {selection}; early stopping: {args.early_stop}; '
          f'num_iter: {args.num_iter}')
    if args.dry_run:
        print('Plan checked. No attacks were run or output files created.')
        return None

    # use a new output directory to keep experiments separate
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f'Output directory is not empty. Choose a new --output_dir: {output_dir}')
    if args.cuda and not torch.cuda.is_available():
        raise ValueError('CUDA is not available in this session.')

    # check metric dependency before running attacks
    from skimage.metrics import structural_similarity

    # load the model once for the whole experiment
    started = time.perf_counter()
    device = torch.device('cuda' if args.cuda else 'cpu')
    method = ExplainingMethod.lrp
    model, data_mean, data_std = load_model(device, method)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / 'metrics.csv'
    history_path = output_dir / 'loss_history.csv'
    completed = 0
    num_converged = 0

    # collect every pair and epsilon in one csv
    with csv_path.open('x', newline='', encoding='utf-8') as csv_file, \
            history_path.open('x', newline='', encoding='utf-8') as history_file:
        writer = None
        history_writer = csv.DictWriter(history_file, fieldnames=['pair_id', *LOSS_HISTORY_FIELDS])
        history_writer.writeheader()
        for position, pair in enumerate(pairs):
            pair_args = argparse.Namespace(**vars(args))
            pair_args.img = pair['original']
            pair_args.target_img = pair['target']
            pair_args.method = 'lrp'
            pair_args.beta_growth = False

            # process epsilon groups for this image pair
            for start in range(0, len(args.epsilons), args.batch_size):
                budgets = args.epsilons[start:start + args.batch_size]
                print(f'Pair {position + 1}/{len(pairs)} (id {pair["pair_id"]}), epsilons {budgets}', flush=True)
                results = attack_pair_batch(model, pair_args, device, method, data_mean, data_std, budgets)

                # save individual outputs and append combined metrics
                for result in results:
                    pair_args.epsilon = result['epsilon_255']
                    name = 'unbounded' if pair_args.epsilon == 255 else f'epsilon_{pair_args.epsilon:g}'
                    pair_args.output_dir = str(output_dir / f'pair_{pair["pair_id"]:03d}' / name)
                    metrics = calculate_metrics(result, data_mean, data_std)
                    row = save_results(result, metrics, pair_args, data_mean, data_std)
                    combined = {'pair_id': pair['pair_id'], **row}
                    if writer is None:
                        writer = csv.DictWriter(csv_file, fieldnames=list(combined))
                        writer.writeheader()
                    writer.writerow(combined)
                    csv_file.flush()
                    history_writer.writerows(
                        {'pair_id': pair['pair_id'], **sample} for sample in result['loss_history']
                    )
                    history_file.flush()
                    completed += 1
                    num_converged += result['run_info']['converged']

                # release batch results before the next group
                del result, results

    elapsed = time.perf_counter() - started
    print(f'Completed {completed}/{num_attacks} attacks in {elapsed:.1f}s.')
    print(f'{num_converged}/{completed} epsilon runs met the plateau rule; check the converged column for the others.')
    print(f'Combined metrics saved to {csv_path}')
    print(f'Combined loss history saved to {history_path}')
    return csv_path


def main():
    args = parse_args()
    run_experiment(args)


if __name__ == '__main__':
    main()
