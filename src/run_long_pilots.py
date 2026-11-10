"""Reproduce the historical two-pair, 20,000-update lowest-loss-image pilots."""

from run_experiment import ROOT, parse_args, run_experiment


def main():
    args = parse_args(defaults={
        'num_iter': 20000, 'batch_size': 7, 'limit_pairs': 2,
        'early_stop': True, 'output_dir': ROOT / 'output/long_pilots',
    })
    args.save_best = True
    run_experiment(args)


if __name__ == '__main__':
    main()
