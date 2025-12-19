import argparse
from datetime import datetime
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from sorting_transformers.utils.config import finalize_config, load_yaml, resolve_config, save_yaml
from sorting_transformers.utils.seed import set_seed
from sorting_transformers.utils.io import ensure_dir
from sorting_transformers.train.trainer import Trainer


def _finalize_config(cfg: dict) -> dict:
    cfg = finalize_config(cfg)
    if cfg.get("run_name") in (None, ""):
        cfg["run_name"] = datetime.now().strftime("%Y%m%d_%H%M%S")
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a sorting transformer variant.")
    parser.add_argument(
        "--config",
        action="append",
        required=True,
        help="Path to a YAML config file (can be specified multiple times).",
    )
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Override config values with key=value (repeatable).",
    )
    parser.add_argument("--run_name", default=None, help="Override run name.")
    parser.add_argument("--output_dir", default=None, help="Override output directory.")
    parser.add_argument(
        "--resume",
        default=None,
        help="Path to checkpoint to resume from.",
    )
    args = parser.parse_args()

    configs = [load_yaml(path) for path in args.config]
    cfg = resolve_config(configs[0], configs[1:], args.override)
    if args.run_name:
        cfg["run_name"] = args.run_name
    if args.output_dir:
        cfg["output_dir"] = args.output_dir
    cfg = _finalize_config(cfg)

    set_seed(cfg["seed"])

    resume_ckpt = Path(args.resume) if args.resume else None
    if resume_ckpt and not args.run_name and not args.output_dir:
        run_dir = resume_ckpt.parent.parent
        cfg["run_name"] = run_dir.name
        cfg["output_dir"] = str(run_dir.parent)
    else:
        run_dir = Path(cfg["output_dir"]) / cfg["run_name"]
    ensure_dir(run_dir)
    save_yaml(cfg, run_dir / "config_resolved.yaml")
    trainer = Trainer(cfg, run_dir=run_dir, resume_ckpt=resume_ckpt)
    summary = trainer.train()
    print(f"Finished run {cfg['run_name']} | best exact match {summary['best_metric']:.4f}")


if __name__ == "__main__":
    main()
