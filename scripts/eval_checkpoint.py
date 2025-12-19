import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

import torch

from sorting_transformers.data.sorting_dataset import build_dataset_from_config
from sorting_transformers.models.variants import build_model
from sorting_transformers.train.evaluate import evaluate_lengths
from sorting_transformers.utils.config import load_yaml, save_yaml
from sorting_transformers.utils.io import ensure_dir, save_json, save_csv
from sorting_transformers.utils.plotting import lineplot, save_figure, set_style
from sorting_transformers.utils.seed import resolve_device


def _load_config(ckpt_path: Path, config_path: str | None) -> dict:
    if config_path:
        return load_yaml(config_path)
    resolved_path = ckpt_path.parent / "config_resolved.yaml"
    if resolved_path.exists():
        return load_yaml(resolved_path)
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    return checkpoint["config"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate checkpoint by length.")
    parser.add_argument("--ckpt", required=True, help="Path to checkpoint.")
    parser.add_argument("--config", default=None, help="Optional config YAML.")
    args = parser.parse_args()

    ckpt_path = Path(args.ckpt)
    cfg = _load_config(ckpt_path, args.config)

    device = resolve_device(cfg["device"])
    model = build_model(cfg).to(device)
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    base_test_dataset = build_dataset_from_config(cfg, split="test", seed_offset=20_000)
    lengths = cfg["dataset"]["test_lengths"]
    metrics_by_length = evaluate_lengths(
        model,
        lengths,
        base_dataset=base_test_dataset,
        batch_size=cfg["train"]["batch_size"],
        pad_token_id=cfg["pad_token_id"],
        bos_token_id=cfg["bos_token_id"],
        eos_token_id=cfg["eos_token_id"],
        device=device,
    )

    run_dir = ckpt_path.parent.parent
    eval_dir = ensure_dir(run_dir / "eval")
    save_yaml(cfg, eval_dir / "config_eval.yaml")

    save_json(metrics_by_length, eval_dir / "metrics_by_length.json")
    rows = [
        {"length": length, **metrics}
        for length in sorted(metrics_by_length.keys())
        for metrics in [metrics_by_length[length]]
    ]
    if rows:
        fieldnames = ["length"] + [key for key in rows[0] if key != "length"]
        save_csv(rows, eval_dir / "metrics_by_length.csv", fieldnames)

    set_style()
    lengths_sorted = sorted(metrics_by_length.keys())
    exact = [metrics_by_length[length]["exact_match"] for length in lengths_sorted]
    tau = [metrics_by_length[length]["kendall_tau"] for length in lengths_sorted]
    title = f"Eval by length | {cfg['model']['variant']} | {cfg['dataset']['regime']} | {ckpt_path.name}"
    lineplot(lengths_sorted, [exact, tau], ["exact_match", "kendall_tau"], title, "length", "score")
    save_figure(eval_dir / "figures" / "metrics_by_length.png", caption="Exact match and Kendall tau by length.")

    print(f"Saved evaluation to {eval_dir}")


if __name__ == "__main__":
    main()
