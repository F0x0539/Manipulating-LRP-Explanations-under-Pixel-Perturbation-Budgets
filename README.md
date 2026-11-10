# Manipulating LRP Explanations under Pixel Perturbation Budgets

LRP explanation attacks under L∞ pixel budgets: 50 ImageNet pairs, seven settings. Based on [Dombrowski et al.](https://github.com/pankessel/adv_explanation_ref).

[Image manifest](data/imagenet/manifest.json)

## Setup

Run commands from the repository root. Tested: Arch Linux, RTX 4090, Python 3.11.18, PyTorch 2.5.1/CUDA 12.4. Install [uv](https://docs.astral.sh/uv/), then:

```sh
uv venv --python 3.11.18
uv pip install --python .venv/bin/python --torch-backend cu124 -r requirements.lock
source .venv/bin/activate
```

Download ILSVRC2012 validation images from [ImageNet](https://www.image-net.org/). The manifest lists all 100 images and 50 pairs. Pass their directory using `--imagenet-dir`; image hashes are verified. VGG16 weights download automatically.

## Run the experiments

Use fresh output directories.

```sh
# Five-update smoke test using the two included sample images.
python src/run_attack.py --cuda --method lrp --num_iter 5 --output_dir output/smoke

# Two unbounded, 1,500-update reproduction checks.
python src/run_attack.py --cuda --method lrp --num_iter 1500 --epsilon 255 --imagenet-dir /path/to/validation \
  --img data/imagenet/images/image_006.png --target_img data/imagenet/images/image_007.png --output_dir output/reproduction/pair_006_007
python src/run_attack.py --cuda --method lrp --num_iter 1500 --epsilon 255 --imagenet-dir /path/to/validation \
  --img data/imagenet/images/image_042.png --target_img data/imagenet/images/image_041.png --output_dir output/reproduction/pair_042_041

# Main study: 50 pairs × seven budgets, 3,000 updates.
python src/run_experiment.py --cuda --num_iter 3000 --lr 0.0002 --batch_size 7 --no-early_stop \
  --imagenet-dir /path/to/validation --output_dir output/main

# Historical pilots: two pairs, seven budgets, up to 20,000 updates.
python src/run_long_pilots.py --cuda --imagenet-dir /path/to/validation
```

The budgets are `1 2 4 8 16 32 255` in 0–255 channel units; `255` is effectively unbounded. The main study saves the final image. The historical pilots save the lowest-total-loss image, matching their original selection policy. All use learning rate 0.0002.

Outputs in `output/`: metrics, loss histories, overview images, and attacked tensors. CSVs contain L∞ change, explanation MSE/SSIM/Pearson, image SSIM, and label retention. Use `--dry_run` with `run_experiment.py` to inspect the plan.

The upstream Apache-2.0 license and attribution are preserved in [LICENSE](LICENSE) and [NOTICE](NOTICE).
