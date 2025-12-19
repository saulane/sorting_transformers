import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / "src"))

import numpy as np
import torch
from torch.utils.data import DataLoader

from sorting_transformers.data.collate import collate_batch
from sorting_transformers.data.sorting_dataset import SortingDataset
from sorting_transformers.models.variants import build_model
from sorting_transformers.utils.checkpoint import load_config_from_checkpoint, load_checkpoint
from sorting_transformers.utils.io import ensure_dir, save_csv, save_json
from sorting_transformers.utils.plotting import save_figure, set_style
from sorting_transformers.utils.seed import resolve_device


def collect_attention_and_hidden(model, cfg, length, n_samples, batch_size, device):
    dataset = SortingDataset(
        size=n_samples,
        lengths=[length],
        value_min=cfg["dataset"]["train_value_min"],
        value_max=cfg["dataset"]["train_value_max"],
        allow_duplicates=cfg["dataset"]["allow_duplicates"],
        vocab_offset=cfg["dataset"]["vocab_offset"],
        bos_token_id=cfg["bos_token_id"],
        eos_token_id=cfg["eos_token_id"],
        seed=cfg["seed"] + length,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate_batch(batch, cfg["pad_token_id"]),
    )

    n_layers = len(model.encoder_layers)
    n_heads = model.encoder_layers[0].self_attn.n_heads
    attn_sum = [np.zeros((n_heads, length + 2, length + 2), dtype=np.float64) for _ in range(n_layers)]
    hidden_norm_sum = np.zeros((n_layers, length + 2), dtype=np.float64)
    total = 0

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        src_key_padding_mask = ~attention_mask
        with torch.no_grad():
            _, extras = model.encode(
                input_ids,
                src_key_padding_mask=src_key_padding_mask,
                return_hidden=True,
                return_attn=True,
            )
        attn_weights = extras["attn_weights"]
        hidden_states = extras["hidden_states"]

        for layer_idx, layer_weights in enumerate(attn_weights):
            attn_sum[layer_idx] += layer_weights.detach().cpu().numpy().mean(axis=0)
        for layer_idx, hidden in enumerate(hidden_states):
            norms = torch.norm(hidden, dim=-1).detach().cpu().numpy()
            hidden_norm_sum[layer_idx] += norms.mean(axis=0)

        total += 1

    attn_avg = [attn / max(1, total) for attn in attn_sum]
    hidden_norm_avg = hidden_norm_sum / max(1, total)
    return attn_avg, hidden_norm_avg


def relative_profile(attn: np.ndarray) -> np.ndarray:
    n_heads, seq_len, _ = attn.shape
    max_dist = seq_len - 1
    profiles = np.zeros((n_heads, max_dist + 1), dtype=np.float64)
    counts = np.zeros(max_dist + 1, dtype=np.float64)
    for i in range(seq_len):
        for j in range(seq_len):
            dist = abs(j - i)
            counts[dist] += 1
            profiles[:, dist] += attn[:, i, j]
    profiles /= counts
    return profiles


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def collect_probe_features(model, cfg, length, n_samples, batch_size, device):
    dataset = SortingDataset(
        size=n_samples,
        lengths=[length],
        value_min=cfg["dataset"]["train_value_min"],
        value_max=cfg["dataset"]["train_value_max"],
        allow_duplicates=cfg["dataset"]["allow_duplicates"],
        vocab_offset=cfg["dataset"]["vocab_offset"],
        bos_token_id=cfg["bos_token_id"],
        eos_token_id=cfg["eos_token_id"],
        seed=cfg["seed"] + 999 + length,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate_batch(batch, cfg["pad_token_id"]),
    )
    features = []
    labels = []

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        src_key_padding_mask = ~attention_mask
        with torch.no_grad():
            _, extras = model.encode(
                input_ids,
                src_key_padding_mask=src_key_padding_mask,
                return_hidden=True,
            )
        hidden_states = extras["hidden_states"][-1]
        values = input_ids[:, 1 : 1 + length].detach().cpu().numpy()
        values = values - cfg["dataset"]["vocab_offset"]
        ranks = []
        for seq in values:
            sorted_seq = sorted(seq.tolist())
            positions = {}
            for idx, val in enumerate(sorted_seq):
                positions.setdefault(val, []).append(idx)
            ranks.extend([positions[val].pop(0) for val in seq])
        labels.append(np.array(ranks))
        features.append(hidden_states[:, 1 : 1 + length, :].detach().cpu().numpy().reshape(-1, hidden_states.size(-1)))

    return np.concatenate(features, axis=0), np.concatenate(labels, axis=0)


def train_transfer_probe(train_features, train_labels, test_features, test_labels, num_classes, device):
    train_x = torch.tensor(train_features, dtype=torch.float32, device=device)
    train_y = torch.tensor(train_labels, dtype=torch.long, device=device)
    test_x = torch.tensor(test_features, dtype=torch.float32, device=device)
    test_y = torch.tensor(test_labels, dtype=torch.long, device=device)

    probe = torch.nn.Linear(train_x.size(1), num_classes).to(device)
    optimizer = torch.optim.SGD(probe.parameters(), lr=0.1)
    criterion = torch.nn.CrossEntropyLoss()

    for _ in range(15):
        logits = probe(train_x)
        loss = criterion(logits, train_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        preds = probe(test_x).argmax(dim=1)
        accuracy = (preds == test_y).float().mean().item()
    return accuracy


def main() -> None:
    parser = argparse.ArgumentParser(description="Length invariance probe.")
    parser.add_argument("--ckpt", required=True, help="Path to checkpoint.")
    parser.add_argument("--config", default=None, help="Optional config YAML.")
    parser.add_argument("--short_len", type=int, default=None, help="Short length.")
    parser.add_argument("--long_len", type=int, default=None, help="Long length.")
    parser.add_argument("--n_samples", type=int, default=256, help="Samples per length.")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size.")
    args = parser.parse_args()

    ckpt_path = Path(args.ckpt)
    cfg = load_config_from_checkpoint(ckpt_path, args.config)
    ckpt_tag = ckpt_path.name
    device = resolve_device(cfg["device"])

    model = build_model(cfg).to(device)
    checkpoint = load_checkpoint(ckpt_path, device="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    short_len = args.short_len or min(cfg["dataset"]["train_lengths"])
    long_len = args.long_len or max(cfg["dataset"]["test_lengths"])

    short_attn, short_norm = collect_attention_and_hidden(
        model, cfg, short_len, args.n_samples, args.batch_size, device
    )
    long_attn, long_norm = collect_attention_and_hidden(
        model, cfg, long_len, args.n_samples, args.batch_size, device
    )

    similarity = []
    for layer_idx, (attn_short, attn_long) in enumerate(zip(short_attn, long_attn, strict=True)):
        prof_short = relative_profile(attn_short)
        prof_long = relative_profile(attn_long)
        max_dist = min(prof_short.shape[1], prof_long.shape[1])
        scores = []
        for head_idx in range(prof_short.shape[0]):
            scores.append(
                cosine_similarity(
                    prof_short[head_idx, :max_dist],
                    prof_long[head_idx, :max_dist],
                )
            )
            similarity.append(
                {
                    "layer": layer_idx,
                    "head": head_idx,
                    "similarity": scores[-1],
                }
            )

    train_features, train_labels = collect_probe_features(
        model, cfg, short_len, args.n_samples, args.batch_size, device
    )
    test_features, test_labels = collect_probe_features(
        model, cfg, long_len, args.n_samples, args.batch_size, device
    )
    transfer_accuracy = train_transfer_probe(
        train_features,
        train_labels,
        test_features,
        test_labels,
        num_classes=max(short_len, long_len),
        device=device,
    )

    run_dir = ckpt_path.parent.parent
    probe_dir = ensure_dir(run_dir / "probes" / "length_invariance")
    save_json(
        {
            "short_len": short_len,
            "long_len": long_len,
            "similarity": similarity,
            "transfer_accuracy": transfer_accuracy,
        },
        probe_dir / "probe_results.json",
    )
    save_csv(similarity, probe_dir / "probe_results.csv", ["layer", "head", "similarity"])

    set_style()
    import matplotlib.pyplot as plt

    layers = sorted(set(item["layer"] for item in similarity))
    avg_sim = []
    for layer in layers:
        scores = [item["similarity"] for item in similarity if item["layer"] == layer]
        avg_sim.append(float(np.mean(scores)))
    plt.figure()
    plt.plot(layers, avg_sim, marker="o")
    plt.title(
        f"Attention similarity by layer | {cfg['model']['variant']} | {cfg['dataset']['regime']} | {ckpt_tag}"
    )
    plt.xlabel("layer")
    plt.ylabel("mean cosine similarity")
    save_figure(
        probe_dir / "figures" / "similarity_by_layer.png",
        caption="Mean cosine similarity of relative attention profiles.",
    )

    plt.figure()
    plt.plot(short_norm.mean(axis=1), label=f"short L={short_len}")
    plt.plot(long_norm.mean(axis=1), label=f"long L={long_len}")
    plt.title(f"Hidden norm statistics by layer | {ckpt_tag}")
    plt.xlabel("layer")
    plt.ylabel("mean norm")
    plt.legend()
    save_figure(
        probe_dir / "figures" / "hidden_norms.png",
        caption="Mean hidden-state norms by layer across lengths.",
    )

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].imshow(short_attn[-1][0], cmap="viridis")
    axes[0].set_title(
        f"Short L={short_len} (layer {len(short_attn)-1}, head 0) | {cfg['model']['variant']} | {cfg['dataset']['regime']} | {ckpt_tag}"
    )
    axes[0].set_xlabel("key position")
    axes[0].set_ylabel("query position")
    axes[1].imshow(long_attn[-1][0], cmap="viridis")
    axes[1].set_title(
        f"Long L={long_len} (layer {len(long_attn)-1}, head 0) | {cfg['model']['variant']} | {cfg['dataset']['regime']} | {ckpt_tag}"
    )
    axes[1].set_xlabel("key position")
    axes[1].set_ylabel("query position")
    save_figure(
        probe_dir / "figures" / "attention_heatmaps.png",
        caption="Representative attention heatmaps for short vs long lengths.",
    )

    print(f"Saved length invariance probe to {probe_dir}")


if __name__ == "__main__":
    main()
