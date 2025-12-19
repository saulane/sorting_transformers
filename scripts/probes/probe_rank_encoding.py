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


def stable_ranks(values: np.ndarray) -> np.ndarray:
    ranks = np.zeros_like(values)
    for i in range(values.shape[0]):
        seq = values[i].tolist()
        sorted_seq = sorted(seq)
        positions = {}
        for idx, val in enumerate(sorted_seq):
            positions.setdefault(val, []).append(idx)
        ranks[i] = np.array([positions[val].pop(0) for val in seq])
    return ranks


def collect_features(
    model,
    cfg: dict,
    length: int,
    n_samples: int,
    batch_size: int,
    device: torch.device,
):
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
    features = [list() for _ in range(n_layers)]
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
        hidden_states = extras["hidden_states"]

        values = input_ids[:, 1 : 1 + length].detach().cpu().numpy()
        values = values - cfg["dataset"]["vocab_offset"]
        batch_ranks = stable_ranks(values)
        labels.append(batch_ranks.reshape(-1))

        for layer_idx, hidden in enumerate(hidden_states):
            layer_hidden = hidden[:, 1 : 1 + length, :].detach().cpu().numpy()
            features[layer_idx].append(layer_hidden.reshape(-1, layer_hidden.shape[-1]))

    labels_np = np.concatenate(labels, axis=0)
    features_np = [np.concatenate(layer_feats, axis=0) for layer_feats in features]
    return features_np, labels_np


def train_linear_probe(
    features: np.ndarray,
    labels: np.ndarray,
    num_classes: int,
    device: torch.device,
    rng: np.random.Generator,
    epochs: int = 15,
    lr: float = 0.1,
):
    idx = rng.permutation(len(labels))
    split = int(0.8 * len(labels))
    train_idx = idx[:split]
    test_idx = idx[split:]

    x_train = torch.tensor(features[train_idx], dtype=torch.float32, device=device)
    y_train = torch.tensor(labels[train_idx], dtype=torch.long, device=device)
    x_test = torch.tensor(features[test_idx], dtype=torch.float32, device=device)
    y_test = torch.tensor(labels[test_idx], dtype=torch.long, device=device)

    probe = torch.nn.Linear(x_train.size(1), num_classes).to(device)
    optimizer = torch.optim.SGD(probe.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()

    for _ in range(epochs):
        logits = probe(x_train)
        loss = criterion(logits, y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        test_logits = probe(x_test)
        preds = test_logits.argmax(dim=1)
        accuracy = (preds == y_test).float().mean().item()

    confusion = None
    if num_classes <= 64:
        confusion = torch.zeros((num_classes, num_classes), dtype=torch.int64)
        for true, pred in zip(y_test.cpu(), preds.cpu(), strict=True):
            confusion[true, pred] += 1
        confusion = confusion.numpy()

    return accuracy, confusion


def main() -> None:
    parser = argparse.ArgumentParser(description="Rank encoding probe.")
    parser.add_argument("--ckpt", required=True, help="Path to checkpoint.")
    parser.add_argument("--config", default=None, help="Optional config YAML.")
    parser.add_argument("--lengths", default=None, help="Comma-separated lengths.")
    parser.add_argument("--n_samples", type=int, default=512, help="Samples per length.")
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

    if args.lengths:
        lengths = [int(x) for x in args.lengths.split(",") if x.strip()]
    else:
        lengths = list(cfg["dataset"]["train_lengths"])
        lengths.append(max(cfg["dataset"]["test_lengths"]))

    results = []
    confusion_matrices = {}

    for length in lengths:
        features, labels = collect_features(
            model,
            cfg,
            length=length,
            n_samples=args.n_samples,
            batch_size=args.batch_size,
            device=device,
        )
        for layer_idx, layer_features in enumerate(features):
            accuracy, confusion = train_linear_probe(
                layer_features,
                labels,
                num_classes=length,
                device=device,
                rng=np.random.default_rng(cfg["seed"] + length + layer_idx),
            )
            results.append(
                {
                    "length": length,
                    "layer": layer_idx,
                    "accuracy": accuracy,
                }
            )
            if confusion is not None and layer_idx == len(features) - 1:
                confusion_matrices[length] = confusion

    run_dir = ckpt_path.parent.parent
    probe_dir = ensure_dir(run_dir / "probes" / "rank_encoding")
    save_json({"results": results}, probe_dir / "probe_results.json")
    save_csv(results, probe_dir / "probe_results.csv", ["length", "layer", "accuracy"])

    set_style()
    import matplotlib.pyplot as plt

    for length in sorted(set(r["length"] for r in results)):
        subset = [r for r in results if r["length"] == length]
        subset = sorted(subset, key=lambda x: x["layer"])
        layers = [r["layer"] for r in subset]
        accs = [r["accuracy"] for r in subset]
        plt.plot(layers, accs, label=f"L={length}")
    plt.title(
        f"Rank probe accuracy by layer | {cfg['model']['variant']} | {cfg['dataset']['regime']} | {ckpt_tag}"
    )
    plt.xlabel("layer")
    plt.ylabel("accuracy")
    plt.legend()
    save_figure(
        probe_dir / "figures" / "rank_probe_accuracy.png",
        caption="Linear probe accuracy by layer across lengths.",
    )

    for length, matrix in confusion_matrices.items():
        plt.figure()
        plt.imshow(matrix, cmap="viridis")
        plt.title(
            f"Confusion matrix (last layer) | L={length} | {cfg['model']['variant']} | {cfg['dataset']['regime']} | {ckpt_tag}"
        )
        plt.xlabel("predicted rank")
        plt.ylabel("true rank")
        plt.colorbar(label="count")
        save_figure(
            probe_dir / "figures" / f"confusion_L{length}.png",
            caption="Confusion matrix for the last encoder layer probe.",
        )

    print(f"Saved rank encoding probe to {probe_dir}")


if __name__ == "__main__":
    main()
