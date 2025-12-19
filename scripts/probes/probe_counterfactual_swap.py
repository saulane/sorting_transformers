import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / "src"))

import numpy as np
import torch

from sorting_transformers.data.sorting_dataset import SortingDataset
from sorting_transformers.models.variants import build_model
from sorting_transformers.train.metrics import kendall_tau
from sorting_transformers.utils.checkpoint import load_config_from_checkpoint, load_checkpoint
from sorting_transformers.utils.io import ensure_dir, save_csv, save_json
from sorting_transformers.utils.plotting import save_figure, set_style
from sorting_transformers.utils.probes import batch_from_values
from sorting_transformers.utils.seed import resolve_device


def greedy_predict(model, cfg, values_batch, device, length):
    input_ids, attention_mask = batch_from_values(values_batch, cfg, device)
    with torch.no_grad():
        preds = model.greedy_decode(
            input_ids,
            src_key_padding_mask=~attention_mask,
            max_len=length + 1,
            bos_token_id=cfg["bos_token_id"],
            pad_token_id=cfg["pad_token_id"],
        )
    return preds.detach().cpu().numpy()


def main() -> None:
    parser = argparse.ArgumentParser(description="Counterfactual swap probe.")
    parser.add_argument("--ckpt", required=True, help="Path to checkpoint.")
    parser.add_argument("--config", default=None, help="Optional config YAML.")
    parser.add_argument("--length", type=int, default=None, help="Sequence length.")
    parser.add_argument("--n_samples", type=int, default=64, help="Number of correct samples to use.")
    args = parser.parse_args()

    ckpt_path = Path(args.ckpt)
    cfg = load_config_from_checkpoint(ckpt_path, args.config)
    ckpt_tag = ckpt_path.name
    device = resolve_device(cfg["device"])

    model = build_model(cfg).to(device)
    checkpoint = load_checkpoint(ckpt_path, device="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    length = args.length or min(cfg["dataset"]["train_lengths"])
    if length < 2:
        raise ValueError("length must be >= 2 for swap probe.")
    dataset = SortingDataset(
        size=max(200, args.n_samples * 3),
        lengths=[length],
        value_min=cfg["dataset"]["train_value_min"],
        value_max=cfg["dataset"]["train_value_max"],
        allow_duplicates=cfg["dataset"]["allow_duplicates"],
        vocab_offset=cfg["dataset"]["vocab_offset"],
        bos_token_id=cfg["bos_token_id"],
        eos_token_id=cfg["eos_token_id"],
        seed=cfg["seed"] + 4242,
    )

    values_list = []
    targets_list = []
    target_tokens_list = []
    for idx in range(len(dataset)):
        item = dataset[idx]
        values = item["values"].tolist()
        targets = sorted(values)
        preds = greedy_predict(model, cfg, [values], device, length)[0]
        target_tokens = [cfg["dataset"]["vocab_offset"] + v for v in targets] + [
            cfg["eos_token_id"]
        ]
        if (preds == np.array(target_tokens)).all():
            values_list.append(values)
            targets_list.append(targets)
            target_tokens_list.append(target_tokens)
        if len(values_list) >= args.n_samples:
            break

    if not values_list:
        raise RuntimeError("No correct samples found for swap probe.")

    clean_preds = greedy_predict(model, cfg, values_list, device, length)

    n = len(values_list)
    sensitivity = np.zeros((length, length), dtype=np.float32)
    change_counts = []
    delta_tau = []
    exact_changes = []

    for a in range(length):
        for b in range(length):
            if a == b:
                continue
            swapped = []
            for values in values_list:
                swapped_values = list(values)
                swapped_values[a], swapped_values[b] = swapped_values[b], swapped_values[a]
                swapped.append(swapped_values)

            swapped_preds = greedy_predict(model, cfg, swapped, device, length)
            changes = (swapped_preds != clean_preds).sum(axis=1)
            change_counts.extend(changes.tolist())
            sensitivity[a, b] = changes.mean() / (length + 1)

            for idx in range(n):
                clean_values = [
                    token - cfg["dataset"]["vocab_offset"]
                    for token in clean_preds[idx][:-1].tolist()
                ]
                swap_values = [
                    token - cfg["dataset"]["vocab_offset"]
                    for token in swapped_preds[idx][:-1].tolist()
                ]
                clean_tau = kendall_tau(clean_values, targets_list[idx])
                swap_tau = kendall_tau(swap_values, targets_list[idx])
                delta_tau.append(swap_tau - clean_tau)
                exact_changes.append(
                    int(
                        (clean_preds[idx] == np.array(target_tokens_list[idx])).all()
                    )
                    - int(
                        (swapped_preds[idx] == np.array(target_tokens_list[idx])).all()
                    )
                )

    run_dir = ckpt_path.parent.parent
    probe_dir = ensure_dir(run_dir / "probes" / "counterfactual_swap")
    mean_changed_positions = float(np.mean(change_counts))
    results = {
        "length": length,
        "mean_changed_positions": mean_changed_positions,
        "globality_score": mean_changed_positions / float(length + 1),
        "mean_delta_kendall_tau": float(np.mean(delta_tau)),
        "mean_exact_match_drop": float(np.mean(exact_changes)),
    }
    save_json(results, probe_dir / "probe_results.json")
    save_csv(
        [results],
        probe_dir / "probe_results.csv",
        [
            "length",
            "mean_changed_positions",
            "globality_score",
            "mean_delta_kendall_tau",
            "mean_exact_match_drop",
        ],
    )

    set_style()
    import matplotlib.pyplot as plt

    plt.figure()
    plt.imshow(sensitivity, cmap="viridis")
    plt.title(
        f"Swap sensitivity | {cfg['model']['variant']} | {cfg['dataset']['regime']} | {ckpt_tag}"
    )
    plt.xlabel("swap position b")
    plt.ylabel("swap position a")
    plt.colorbar(label="mean fraction changed")
    save_figure(
        probe_dir / "figures" / "sensitivity_heatmap.png",
        caption="Mean fraction of output tokens changed when swapping positions.",
    )

    plt.figure()
    plt.hist(change_counts, bins=20)
    plt.title(
        f"Histogram of output token changes | {cfg['model']['variant']} | {cfg['dataset']['regime']} | {ckpt_tag}"
    )
    plt.xlabel("# changed tokens")
    plt.ylabel("count")
    save_figure(
        probe_dir / "figures" / "change_histogram.png",
        caption="Distribution of output token changes across swaps.",
    )

    print(f"Saved counterfactual swap probe to {probe_dir}")


if __name__ == "__main__":
    main()
