import argparse
import csv
from pathlib import Path
import torch
import torchvision
import torch.nn.functional as F
import numpy as np

from attack_progress import PlateauTracker, add_progress_args, validate_progress_args
from dataset_images import recorded_image, resolve_image
from nn.enums import ExplainingMethod
from nn.networks import ExplainableNet
from nn.utils import get_expl, plot_overview, clamp, load_image, make_dir, torch_to_image

ROOT = Path(__file__).resolve().parents[1]


def recorded_path(path):
    return recorded_image(path)


LOSS_HISTORY_FIELDS = [
    'epsilon_255', 'iteration', 'total_loss', 'expl_loss', 'output_loss', 'lr',
    'best_total_loss', 'best_iteration', 'plateau_checks_met', 'converged', 'selected',
]


def get_beta(i, num_iter):
    """
    Helper method for beta growth
    """
    start_beta, end_beta = 10.0, 100.0
    return start_beta * (end_beta / start_beta) ** (i / num_iter)


def parse_args():
    """
    Read command line options
    """
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--num_iter', type=int, default=1500, help='Adam updates; maximum with --early_stop')
    argparser.add_argument('--img', type=str, default=str(ROOT / 'data/collie4.jpeg'), help='image net file to run attack on')
    argparser.add_argument('--target_img', type=str, default=str(ROOT / 'data/tiger_cat.jpeg'),
                           help='imagenet file used to generate target expl')
    argparser.add_argument('--lr', type=float, default=0.0002, help='lr')
    argparser.add_argument('--log_every', type=int, default=50, help='steps between saved loss samples')
    argparser.add_argument('--cuda', help='enable GPU mode', action='store_true')
    argparser.add_argument('--output_dir', type=str, default=str(ROOT / 'output/single'), help='directory to save results to')
    argparser.add_argument('--imagenet-dir', type=Path, help='directory containing the downloaded validation JPEGs')
    argparser.add_argument('--beta_growth', help='enable beta growth', action='store_true')
    argparser.add_argument('--prefactors', nargs=2, default=[1e11, 1e6], type=float,
                           help='prefactors of losses (diff expls, class loss)')
    argparser.add_argument('--method', help='algorithm for expls',
                           choices=['lrp', 'guided_backprop', 'gradient', 'integrated_grad',
                                    'pattern_attribution', 'grad_times_input'],
                           default='lrp')
    budget_options = argparser.add_mutually_exclusive_group()
    budget_options.add_argument('--epsilon', type=float, default=255.0, help='budget value 0-255')
    budget_options.add_argument('--epsilons', nargs='+', type=float,
                                help='budget values 0-255 to process together')
    add_progress_args(argparser)
    args = argparser.parse_args()
    try:
        validate_progress_args(args)
    except ValueError as error:
        argparser.error(str(error))

    # check epsilon budgets
    budgets = args.epsilons if args.epsilons is not None else [args.epsilon]
    if any(not np.isfinite(value) or not 0 <= value <= 255 for value in budgets):
        argparser.error('Epsilon budgets must be finite values between 0 and 255.')
    if len(budgets) > 1 and args.method != 'lrp':
        argparser.error('Batched epsilon attacks currently support LRP only.')
    if args.log_every < 1:
        argparser.error('Logging interval must be positive.')

    # load metric dependency before running attack
    try:
        from skimage.metrics import structural_similarity
    except ImportError as error:
        argparser.error(f"CSV metrics require scikit-image from requirements.txt: {error}")

    try:
        args.img = str(resolve_image(args.img, args.imagenet_dir))
        args.target_img = str(resolve_image(args.target_img, args.imagenet_dir))
    except (OSError, ValueError) as error:
        argparser.error(str(error))
    return args


def load_model(device, method, beta_growth=False):
    """
    Load the model and its normalization values
    """
    # load model
    data_mean = np.array([0.485, 0.456, 0.406])
    data_std = np.array([0.229, 0.224, 0.225])
    vgg_model = torchvision.models.vgg16(pretrained=True)
    model = ExplainableNet(vgg_model, data_mean=data_mean, data_std=data_std, beta=1000 if beta_growth else None)
    if method == ExplainingMethod.pattern_attribution:
        model.load_state_dict(torch.load('../models/model_vgg16_pattern_small.pth'), strict=False)
    model = model.eval().to(device)
    model.requires_grad_(False)

    return model, data_mean, data_std


def attack_pair(model, args, device, method, data_mean, data_std):
    """
    Attack one image pair using an already loaded model
    """
    return attack_pair_batch(model, args, device, method, data_mean, data_std, [args.epsilon])[0]


def calculate_losses(adv_expl, target_expls, adv_acc, original_outputs, prefactors):
    """
    Calculate explanation, output and total loss per image
    """
    if adv_expl.shape[0] == 1:
        # retain the original single-image reduction
        loss_expl = F.mse_loss(adv_expl, target_expls).reshape(1)
        loss_output = F.mse_loss(adv_acc, original_outputs).reshape(1)
    else:
        loss_expl = F.mse_loss(adv_expl, target_expls, reduction='none').flatten(1).mean(dim=1)
        loss_output = F.mse_loss(adv_acc, original_outputs, reduction='none').mean(dim=1)
    total_loss = prefactors[0]*loss_expl + prefactors[1]*loss_output
    return total_loss, loss_expl, loss_output


def attack_pair_batch(model, args, device, method, data_mean, data_std, epsilons):
    """
    Attack one image pair with a separate image copy for each epsilon
    """
    epsilons = [float(value) for value in epsilons]
    if not epsilons or any(not np.isfinite(value) or not 0 <= value <= 255 for value in epsilons):
        raise ValueError('Epsilon budgets must be finite values between 0 and 255.')
    batch_size = len(epsilons)
    if batch_size > 1 and method != ExplainingMethod.lrp:
        raise ValueError('Batched epsilon attacks currently support LRP only.')
    log_every = getattr(args, 'log_every', 50)
    if log_every < 1:
        raise ValueError('Logging interval must be positive.')
    early_stop = getattr(args, 'early_stop', False)
    # Only the historical-pilot entry point opts into lowest-loss image selection.
    save_best = getattr(args, 'save_best', False)
    if save_best and args.beta_growth:
        raise ValueError('Historical lowest-loss selection requires a fixed ReLU objective.')
    if args.beta_growth and early_stop:
        raise ValueError('Early stopping requires a fixed objective, without beta growth.')
    progress = PlateauTracker(
        batch_size, window=getattr(args, 'plateau_window', 300),
        checks=getattr(args, 'plateau_checks', 3), threshold=getattr(args, 'plateau_threshold', 0.01),
        min_iter=getattr(args, 'min_iter', 1500),
    )

    # reset activations when reusing the model
    model.change_beta(1000 if args.beta_growth else None)

    # load images
    x = load_image(data_mean, data_std, device, args.img)
    x_target = load_image(data_mean, data_std, device, args.target_img)

    # produce expls
    org_expl, org_acc, org_idx = get_expl(model, x, method)
    org_expl = org_expl.detach().cpu()
    org_acc = org_acc.detach()
    target_expl, _, _ = get_expl(model, x_target, method)
    target_expl = target_expl.detach()

    # create one independent image copy per epsilon
    originals = x.detach().repeat(batch_size, 1, 1, 1)
    x_adv = originals.clone().requires_grad_()
    best_images = x_adv.detach().clone() if save_best else None
    best_samples = [None] * batch_size
    class_indices = org_idx.expand(batch_size)
    target_expls = target_expl.expand(batch_size, -1, -1)
    original_outputs = org_acc.expand(batch_size, -1)

    # normalize each epsilon budget per channel
    channel_std = x_adv.new_tensor(data_std).view(1, 3, 1, 1)
    budgets = x_adv.new_tensor(epsilons).view(batch_size, 1, 1, 1)
    epsilon_normalized = (budgets / 255.0) / channel_std

    optimizer = torch.optim.Adam([x_adv], lr=args.lr)
    loss_history = [{} for _ in epsilons]

    def record_progress(iteration, losses, final=False):
        # Track diagnostics; only the historical pilot retains earlier images.
        improved = progress.observe(iteration, [values[0] for values in losses])
        for index, (total, explanation, output) in enumerate(losses):
            sample = {
                'epsilon_255': epsilons[index], 'iteration': iteration,
                'total_loss': total, 'expl_loss': explanation, 'output_loss': output,
                'lr': optimizer.param_groups[0]['lr'],
                'best_total_loss': progress.best_losses[index],
                'best_iteration': progress.best_iterations[index],
                'plateau_checks_met': progress.flat_windows[index],
                'converged': int(progress.converged[index]), 'selected': 0,
            }
            if save_best and improved[index]:
                with torch.no_grad():
                    best_images[index].copy_(x_adv[index])
                best_samples[index] = sample.copy()
            if final or iteration % log_every == 0 or iteration % progress.window == 0:
                loss_history[index][iteration] = sample

    iterations_run = 0
    stopped = False
    for i in range(args.num_iter):
        if args.beta_growth:
            model.change_beta(get_beta(i, args.num_iter))

        optimizer.zero_grad()

        # calculate loss per image
        adv_expl, adv_acc, class_idx = get_expl(model, x_adv, method, desired_index=class_indices)
        total_loss, loss_expl, loss_output = calculate_losses(
            adv_expl, target_expls, adv_acc, original_outputs, args.prefactors,
        )

        # record losses before the next optimizer step
        losses = torch.stack((total_loss, loss_expl, loss_output), dim=1).detach().cpu().tolist()
        record_progress(i, losses)
        for index, (total, explanation, output) in enumerate(losses):
            prefix = f"Iteration {i}"
            if batch_size > 1:
                prefix += f" (epsilon {epsilons[index]:g})"
            print(f"{prefix}: Total Loss: {total}, Expl Loss: {explanation}, Output Loss: {output}")

        # wait until every epsilon meets the rule at the same window boundary
        if early_stop and progress.should_stop():
            stopped = True
            break

        # update adversarial example
        total_loss.sum().backward()
        optimizer.step()

        # clip perturbation to epsilon budget
        with torch.no_grad():
            # calculate change from original image
            change = x_adv - originals

            # restrict change per channel
            limited_change = torch.clamp(
                change,
                min=-epsilon_normalized,
                max=epsilon_normalized,
            )

            # reconstruct adversarial example
            limited_image = originals + limited_change

            # clamp to valid pixels
            valid_image = clamp(limited_image, data_mean, data_std)

            # update tensor used by optimizer
            x_adv.copy_(valid_image)
        iterations_run = i + 1

    # evaluate the last optimizer state unless it was already checked for stopping
    if not stopped:
        adv_expl, adv_acc, class_idx = get_expl(model, x_adv, method, desired_index=class_indices)
    with torch.no_grad():
        final_losses = calculate_losses(adv_expl, target_expls, adv_acc, original_outputs, args.prefactors)
        final_losses = torch.stack(final_losses, dim=1).cpu().tolist()
    if not stopped:
        record_progress(iterations_run, final_losses, final=True)
    stop_reason = 'plateau' if early_stop and progress.should_stop() else 'max_iter'

    saved_iterations = progress.best_iterations.copy() if save_best else [iterations_run] * batch_size
    if save_best:
        x_adv = best_images.detach().requires_grad_()
        adv_expl, adv_acc, class_idx = get_expl(model, x_adv, method, desired_index=class_indices)

    # evaluate the final image on the original ReLU network
    if args.beta_growth:
        del adv_expl, adv_acc, class_idx, total_loss, loss_expl, loss_output
        model.change_beta(None)
        adv_expl, adv_acc, class_idx = get_expl(model, x_adv, method, desired_index=class_indices)
    else:
        model.change_beta(None)
    with torch.no_grad():
        saved_losses = calculate_losses(adv_expl, target_expls, adv_acc, original_outputs, args.prefactors)
        saved_losses = torch.stack(saved_losses, dim=1).cpu().tolist()
        linf = ((x_adv - originals) * channel_std).abs().flatten(1).max(dim=1).values.cpu().tolist()
    print(f'Batch finished after {iterations_run}/{args.num_iter} updates: {stop_reason}; '
          f'{sum(progress.converged)}/{batch_size} epsilon versions meet the plateau rule.')

    # separate results and release gradient graphs
    results = []
    for index, epsilon in enumerate(epsilons):
        # Main runs retain the final iterate; historical pilots retain the best.
        selected_iteration = saved_iterations[index]
        if save_best:
            loss_history[index].setdefault(selected_iteration, best_samples[index])
        loss_history[index][selected_iteration]['selected'] = 1
        history = [sample for _, sample in sorted(loss_history[index].items())]
        results.append({
            "x": originals[index:index + 1].detach(),
            "x_target": x_target.detach(),
            "x_adv": x_adv[index:index + 1].detach(),
            "org_expl": org_expl,
            "target_expl": target_expl,
            "adv_expl": adv_expl[index:index + 1].detach(),
            "org_idx": org_idx.detach(),
            "class_idx": class_idx[index:index + 1].detach(),
            "linf": linf[index],
            "epsilon_255": epsilon,
            "loss_history": history,
            "run_info": {
                'iterations_run': iterations_run, 'saved_iteration': selected_iteration,
                'stop_reason': stop_reason, 'converged': int(progress.converged[index]),
                'save_best': int(save_best), 'early_stop': int(early_stop),
                'min_iter': progress.min_iter, 'plateau_window': progress.window,
                'plateau_checks': progress.checks, 'plateau_threshold': progress.threshold,
                'best_total_loss': progress.best_losses[index],
                'saved_total_loss': saved_losses[index][0],
                'saved_expl_loss': saved_losses[index][1], 'saved_output_loss': saved_losses[index][2],
            },
        })
    return results


def calculate_metrics(result, data_mean, data_std):
    """
    Measure perturbation, explanation similarity and image similarity
    """
    from skimage.metrics import structural_similarity

    # convert explanation maps to numpy arrays
    adv_map = result["adv_expl"].detach().cpu().squeeze(0).numpy().astype(np.float64)
    target_map = result["target_expl"].detach().cpu().squeeze(0).numpy().astype(np.float64)

    # convert images to pixel space
    original_image = torch_to_image(result["x"], data_mean, data_std)
    attacked_image = torch_to_image(result["x_adv"], data_mean, data_std)

    # calculate explanation mse
    expl_mse = float(np.mean((adv_map - target_map) ** 2))

    # calculate explanation ssim with a fixed target range per image pair
    explanation_range = float(np.ptp(target_map))
    expl_ssim = float("nan")
    if explanation_range > 0:
        expl_ssim = float(structural_similarity(
            adv_map, target_map, data_range=explanation_range,
        ))

    # calculate explanation correlation (undefined for constant maps)
    expl_pearson = float("nan")
    if np.ptp(adv_map) > 0 and np.ptp(target_map) > 0:
        expl_pearson = float(np.corrcoef(adv_map.ravel(), target_map.ravel())[0, 1])

    # calculate image ssim
    image_ssim = float(structural_similarity(
        original_image, attacked_image, data_range=1.0, channel_axis=-1,
    ))

    return {
        "linf_255": result["linf"] * 255.0,
        "expl_mse": expl_mse,
        "expl_ssim": expl_ssim,
        "expl_pearson": expl_pearson,
        "image_ssim": image_ssim,
        "label_kept": int(result["org_idx"].item() == result["class_idx"].item()),
        "expl_ssim_data_range": explanation_range,
    }


def save_results(result, metrics, args, data_mean, data_std):
    """
    Save metrics, overview and adversarial example for one run
    """
    # create output directory
    output_dir = make_dir(args.output_dir)

    # collect metrics for this run
    row = {
        "original_image": recorded_path(args.img),
        "target_image": recorded_path(args.target_img),
        "method": args.method,
        "epsilon_255": args.epsilon,
        "num_iter": args.num_iter,
        "lr": args.lr,
        "expl_prefactor": args.prefactors[0], "output_prefactor": args.prefactors[1],
        **result['run_info'],
        **metrics,
    }

    # append metrics, writing the header for a new or empty file
    csv_path = Path(output_dir) / "metrics.csv"
    write_header = not csv_path.exists() or csv_path.stat().st_size == 0
    if not write_header:
        with csv_path.open(newline='', encoding='utf-8') as csv_file:
            if next(csv.reader(csv_file)) != list(row):
                raise ValueError('Existing metrics.csv has different columns. Choose a new output directory.')
    with csv_path.open("a", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(row))
        if write_header:
            writer.writeheader()
        writer.writerow(row)

    # save the loss curve for the image written by this run
    history_path = Path(output_dir) / 'loss_history.csv'
    with history_path.open('w', newline='', encoding='utf-8') as history_file:
        writer = csv.DictWriter(history_file, fieldnames=LOSS_HISTORY_FIELDS)
        writer.writeheader()
        writer.writerows(result['loss_history'])

    print(f"Metrics saved to {csv_path}")
    print(metrics["linf_255"])
    plot_overview(
        [result["x_target"], result["x"], result["x_adv"]],
        [result["target_expl"], result["org_expl"], result["adv_expl"]],
        data_mean, data_std, filename=f"{output_dir}overview_{args.method}.png",
    )
    torch.save(result["x_adv"], f"{output_dir}x_{args.method}.pth")

    return row


def main():
    args = parse_args()

    # options
    device = torch.device("cuda" if args.cuda else "cpu")
    method = getattr(ExplainingMethod, args.method)

    # run attack and save results
    model, data_mean, data_std = load_model(device, method, args.beta_growth)
    if args.epsilons is None:
        results = [attack_pair(model, args, device, method, data_mean, data_std)]
    else:
        results = attack_pair_batch(model, args, device, method, data_mean, data_std, args.epsilons)

    # save each epsilon result separately
    for result in results:
        run_args = argparse.Namespace(**vars(args))
        run_args.epsilon = result["epsilon_255"]
        if args.epsilons is not None:
            name = 'unbounded' if run_args.epsilon == 255 else f'epsilon_{run_args.epsilon:g}'
            run_args.output_dir = str(Path(args.output_dir) / name)
        metrics = calculate_metrics(result, data_mean, data_std)
        save_results(result, metrics, run_args, data_mean, data_std)


if __name__ == "__main__":
    main()
