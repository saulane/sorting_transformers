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
from sorting_transformers.utils.io import ensure_dir, save_csv, save_json, save_tensor_npz
from sorting_transformers.utils.plotting import save_figure, set_style
from sorting_transformers.utils.probes import batch_from_values
from sorting_transformers.utils.seed import resolve_device


def greedy_predict(model, cfg, values_batch, device, length, overrides=None, head_overrides=None):
    input_ids, attention_mask = batch_from_values(values_batch, cfg, device)
    src_key_padding_mask = ~attention_mask
    if overrides or head_overrides:
        with torch.no_grad():
            memory, _ = model.encode(
                input_ids,
                src_key_padding_mask=src_key_padding_mask,
                layer_output_overrides=overrides,
                attn_head_overrides=head_overrides,
            )
            batch_size = input_ids.size(0)
            ys = torch.full(
                (batch_size, 1),
                cfg["bos_token_id"],
                device=device,
                dtype=torch.long,
            )
            for _ in range(length + 1):
                tgt_key_padding_mask = ys.eq(cfg["pad_token_id"])
                logits, _ = model.decode(
                    ys,
                    memory,
                    src_key_padding_mask=src_key_padding_mask,
                    tgt_key_padding_mask=tgt_key_padding_mask,
                )
                next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
                ys = torch.cat([ys, next_token], dim=1)
            preds = ys[:, 1:]
    else:
        with torch.no_grad():
            preds = model.greedy_decode(
                input_ids,
                src_key_padding_mask=src_key_padding_mask,
                max_len=length + 1,
                bos_token_id=cfg["bos_token_id"],
                pad_token_id=cfg["pad_token_id"],
            )
    return preds.detach().cpu().numpy()


def batch_metrics(preds, targets, cfg):
    exact = (preds == targets).all(axis=1).mean()
    taus = []
    for pred, target in zip(preds, targets, strict=True):
        pred_values = [
            token - cfg["dataset"]["vocab_offset"] for token in pred[:-1].tolist()
        ]
        target_values = [
            token - cfg["dataset"]["vocab_offset"] for token in target[:-1].tolist()
        ]
        taus.append(kendall_tau(pred_values, target_values))
    return float(exact), float(np.mean(taus))


def main() -> None:
    parser = argparse.ArgumentParser(description="Activation patching probe.")
    parser.add_argument("--ckpt", required=True, help="Path to checkpoint.")
    parser.add_argument("--config", default=None, help="Optional config YAML.")
    parser.add_argument("--length", type=int, default=None, help="Sequence length.")
    parser.add_argument("--n_samples", type=int, default=32, help="Number of samples.")
    parser.add_argument("--swap_a", type=int, default=0, help="Swap position a.")
    parser.add_argument("--swap_b", type=int, default=1, help="Swap position b.")
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
    if args.swap_a >= length or args.swap_b >= length:
        raise ValueError("swap positions must be within sequence length.")
    dataset = SortingDataset(
        size=args.n_samples,
        lengths=[length],
        value_min=cfg["dataset"]["train_value_min"],
        value_max=cfg["dataset"]["train_value_max"],
        allow_duplicates=cfg["dataset"]["allow_duplicates"],
        vocab_offset=cfg["dataset"]["vocab_offset"],
        bos_token_id=cfg["bos_token_id"],
        eos_token_id=cfg["eos_token_id"],
        seed=cfg["seed"] + 777,
    )

    clean_values = [dataset[i]["values"].tolist() for i in range(len(dataset))]
    corrupted_values = []
    for values in clean_values:
        swapped = list(values)
        swapped[args.swap_a], swapped[args.swap_b] = swapped[args.swap_b], swapped[args.swap_a]
        corrupted_values.append(swapped)

    target_tokens = [
        [cfg["dataset"]["vocab_offset"] + v for v in sorted(values)] + [cfg["eos_token_id"]]
        for values in clean_values
    ]
    target_tokens = np.array(target_tokens, dtype=np.int64)

    clean_preds = greedy_predict(model, cfg, clean_values, device, length)
    corrupted_preds = greedy_predict(model, cfg, corrupted_values, device, length)

    clean_exact, clean_tau = batch_metrics(clean_preds, target_tokens, cfg)
    corrupted_exact, corrupted_tau = batch_metrics(corrupted_preds, target_tokens, cfg)

    input_ids, attention_mask = batch_from_values(clean_values, cfg, device)
    src_key_padding_mask = ~attention_mask
    with torch.no_grad():
        _, clean_extras = model.encode(
            input_ids,
            src_key_padding_mask=src_key_padding_mask,
            return_hidden=True,
            return_head_output=True,
        )
    clean_hidden = clean_extras["hidden_states"]
    clean_head_outputs = clean_extras["head_outputs"]

    layer_recovery = []
    for layer_idx, layer_state in enumerate(clean_hidden):
        overrides = {layer_idx: layer_state}
        patched_preds = greedy_predict(
            model, cfg, corrupted_values, device, length, overrides=overrides
        )
        patched_exact, patched_tau = batch_metrics(patched_preds, target_tokens, cfg)
        layer_recovery.append(
            {
                "layer": layer_idx,
                "exact_recovery": patched_exact - corrupted_exact,
                "tau_recovery": patched_tau - corrupted_tau,
            }
        )

    head_recovery = np.zeros((len(clean_head_outputs), clean_head_outputs[0].size(1)))
    for layer_idx, head_outputs in enumerate(clean_head_outputs):
        for head_idx in range(head_outputs.size(1)):
            head_mask = torch.zeros(head_outputs.size(1), dtype=torch.bool, device=device)
            head_mask[head_idx] = True
            head_overrides = {
                layer_idx: {
                    "head_output": head_outputs,
                    "head_mask": head_mask,
                }
            }
            patched_preds = greedy_predict(
                model, cfg, corrupted_values, device, length, head_overrides=head_overrides
            )
            patched_exact, _ = batch_metrics(patched_preds, target_tokens, cfg)
            head_recovery[layer_idx, head_idx] = patched_exact - corrupted_exact

    run_dir = ckpt_path.parent.parent
    probe_dir = ensure_dir(run_dir / "probes" / "activation_patching")
    save_json(
        {
            "clean_exact": clean_exact,
            "clean_tau": clean_tau,
            "corrupted_exact": corrupted_exact,
            "corrupted_tau": corrupted_tau,
            "layer_recovery": layer_recovery,
            "head_recovery_path": "head_recovery.npz",
        },
        probe_dir / "probe_results.json",
    )
    save_tensor_npz(probe_dir / "head_recovery.npz", head_recovery=head_recovery)
    save_csv(
        layer_recovery,
        probe_dir / "probe_results.csv",
        ["layer", "exact_recovery", "tau_recovery"],
    )

    set_style()
    import matplotlib.pyplot as plt

    layers = [item["layer"] for item in layer_recovery]
    exact_rec = [item["exact_recovery"] for item in layer_recovery]
    plt.figure()
    plt.plot(layers, exact_rec, marker="o")
    plt.title(
        f"Activation patching recovery | {cfg['model']['variant']} | {cfg['dataset']['regime']} | {ckpt_tag}"
    )
    plt.xlabel("layer")
    plt.ylabel("exact match recovery")
    save_figure(
        probe_dir / "figures" / "layer_recovery.png",
        caption="Exact match recovery after patching layer outputs.",
    )

    plt.figure()
    plt.imshow(head_recovery, cmap="viridis")
    plt.title(
        f"Head-level recovery (exact match) | {cfg['model']['variant']} | {cfg['dataset']['regime']} | {ckpt_tag}"
    )
    plt.xlabel("head")
    plt.ylabel("layer")
    plt.colorbar(label="recovery")
    save_figure(
        probe_dir / "figures" / "head_recovery.png",
        caption="Head-level exact match recovery.",
    )

    print(f"Saved activation patching probe to {probe_dir}")


if __name__ == "__main__":
    main()
