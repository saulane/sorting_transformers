import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / "src"))

import numpy as np
import torch

from sorting_transformers.models.variants import build_model
from sorting_transformers.utils.checkpoint import load_config_from_checkpoint, load_checkpoint
from sorting_transformers.utils.io import ensure_dir, save_csv, save_json
from sorting_transformers.utils.plotting import save_figure, set_style
from sorting_transformers.utils.probes import batch_from_values
from sorting_transformers.utils.seed import resolve_device


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe attention heads as comparators.")
    parser.add_argument("--ckpt", required=True, help="Path to checkpoint.")
    parser.add_argument("--config", default=None, help="Optional config YAML.")
    parser.add_argument("--length", type=int, default=None, help="Sequence length to probe.")
    parser.add_argument("--pos_i", type=int, default=0, help="First value position index.")
    parser.add_argument("--pos_j", type=int, default=1, help="Second value position index.")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for probe.")
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
    if args.pos_i >= length or args.pos_j >= length:
        raise ValueError("pos_i and pos_j must be within sequence length.")
    value_min = cfg["dataset"]["train_value_min"]
    value_max = cfg["dataset"]["train_value_max"]
    fixed_value = (value_min + value_max) // 2

    samples = []
    deltas = []
    indicators = []
    pairs = []
    for v_i in range(value_min, value_max + 1):
        for v_j in range(value_min, value_max + 1):
            values = [fixed_value] * length
            values[args.pos_i] = v_i
            values[args.pos_j] = v_j
            delta = v_i - v_j
            samples.append(values)
            deltas.append(delta)
            indicators.append(
                [float(delta > 0), float(delta < 0), float(delta == 0)]
            )
            pairs.append((v_i, v_j))

    deltas_np = np.array(deltas, dtype=np.float32)
    indicators_np = np.array(indicators, dtype=np.float32)

    n_layers = len(model.encoder_layers)
    n_heads = model.encoder_layers[0].self_attn.n_heads
    weights = np.zeros((len(samples), n_layers, n_heads), dtype=np.float32)

    query_index = args.pos_i + 1
    key_index = args.pos_j + 1

    with torch.no_grad():
        for start in range(0, len(samples), args.batch_size):
            batch_values = samples[start : start + args.batch_size]
            input_ids, attention_mask = batch_from_values(batch_values, cfg, device)
            src_key_padding_mask = ~attention_mask

            _, extras = model.encode(
                input_ids,
                src_key_padding_mask=src_key_padding_mask,
                return_attn=True,
            )
            attn_weights = extras["attn_weights"]
            for layer_idx, layer_weights in enumerate(attn_weights):
                layer_np = layer_weights[:, :, query_index, key_index].detach().cpu().numpy()
                weights[start : start + layer_np.shape[0], layer_idx, :] = layer_np

    results = []
    for layer_idx in range(n_layers):
        for head_idx in range(n_heads):
            head_weights = weights[:, layer_idx, head_idx]
            corr = np.corrcoef(head_weights, deltas_np)[0, 1]
            corr_gt = np.corrcoef(head_weights, indicators_np[:, 0])[0, 1]
            corr_lt = np.corrcoef(head_weights, indicators_np[:, 1])[0, 1]
            corr_eq = np.corrcoef(head_weights, indicators_np[:, 2])[0, 1]
            corr = float(np.nan_to_num(corr))
            corr_gt = float(np.nan_to_num(corr_gt))
            corr_lt = float(np.nan_to_num(corr_lt))
            corr_eq = float(np.nan_to_num(corr_eq))
            results.append(
                {
                    "layer": layer_idx,
                    "head": head_idx,
                    "corr_delta": corr,
                    "corr_gt": corr_gt,
                    "corr_lt": corr_lt,
                    "corr_eq": corr_eq,
                }
            )

    results_sorted = sorted(results, key=lambda x: abs(x["corr_delta"]), reverse=True)

    run_dir = ckpt_path.parent.parent
    probe_dir = ensure_dir(run_dir / "probes" / "attention_comparison")
    save_json(results_sorted, probe_dir / "probe_results.json")
    save_csv(results_sorted, probe_dir / "probe_results.csv", [
        "layer",
        "head",
        "corr_delta",
        "corr_gt",
        "corr_lt",
        "corr_eq",
    ])

    top = results_sorted[:4]
    set_style()
    value_range = list(range(value_min, value_max + 1))
    for idx, item in enumerate(top, start=1):
        layer = item["layer"]
        head = item["head"]
        head_weights = weights[:, layer, head]
        unique_deltas = sorted(set(deltas))
        mean_weights = []
        for delta in unique_deltas:
            mean_weights.append(head_weights[deltas_np == delta].mean())

        import matplotlib.pyplot as plt

        plt.figure()
        plt.scatter(deltas_np, head_weights, alpha=0.3, s=10, label="samples")
        plt.plot(unique_deltas, mean_weights, color="red", label="mean")
        plt.title(
            f"Attention vs delta | L{layer} H{head} | {cfg['model']['variant']} | {cfg['dataset']['regime']} | {ckpt_tag}"
        )
        plt.xlabel("value_i - value_j")
        plt.ylabel("attention weight")
        plt.legend()
        save_figure(
            probe_dir / "figures" / f"scatter_layer{layer}_head{head}.png",
            caption="Attention weight vs delta for selected head.",
        )

        plt.figure()
        plt.plot(unique_deltas, mean_weights, marker="o")
        plt.title(
            f"Mean attention vs delta | L{layer} H{head} | {cfg['model']['variant']} | {ckpt_tag}"
        )
        plt.xlabel("value_i - value_j")
        plt.ylabel("mean attention")
        save_figure(
            probe_dir / "figures" / f"mean_layer{layer}_head{head}.png",
            caption="Mean attention weight aggregated by delta.",
        )

        heat = np.zeros((len(value_range), len(value_range)), dtype=np.float32)
        counts = np.zeros_like(heat)
        for (v_i, v_j), w in zip(pairs, head_weights, strict=True):
            i_idx = v_i - value_min
            j_idx = v_j - value_min
            heat[i_idx, j_idx] += w
            counts[i_idx, j_idx] += 1
        heat = np.divide(heat, np.maximum(counts, 1))
        plt.figure()
        plt.imshow(heat, cmap="viridis")
        plt.title(f"Attention heatmap | L{layer} H{head} | {ckpt_tag}")
        plt.xlabel("value_j")
        plt.ylabel("value_i")
        plt.colorbar(label="attention weight")
        save_figure(
            probe_dir / "figures" / f"heatmap_layer{layer}_head{head}.png",
            caption="Attention weight heatmap over value pairs.",
        )

    print(f"Saved attention comparison probe to {probe_dir}")


if __name__ == "__main__":
    main()
