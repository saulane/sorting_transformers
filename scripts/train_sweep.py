import argparse
from datetime import datetime
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from sorting_transformers.train.trainer import Trainer
from sorting_transformers.utils.config import finalize_config, load_yaml, resolve_config, save_yaml
from sorting_transformers.utils.seed import set_seed
from sorting_transformers.utils.io import ensure_dir


DEFAULT_REGIMES = ["unique", "dups", "value_shift", "len_extrap"]
DEFAULT_VARIANTS = [
    "baseline_absolute",
    "rel_bias",
    "rope",
    "alibi",
    "comparison_normalized",
    "tied_attention",
    "tiny",
    "wide_shallow",
]


def _load_optional(path: Path) -> dict:
    return load_yaml(path) if path.exists() else {}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a sweep of variants and regimes.")
    parser.add_argument(
        "--base_config",
        default="configs/base.yaml",
        help="Base config YAML.",
    )
    parser.add_argument(
        "--regimes",
        default=",".join(DEFAULT_REGIMES),
        help="Comma-separated regime names.",
    )
    parser.add_argument(
        "--variants",
        default=",".join(DEFAULT_VARIANTS),
        help="Comma-separated model variants.",
    )
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Global key=value overrides.",
    )
    args = parser.parse_args()

    base_cfg = load_yaml(args.base_config)
    regimes = [r.strip() for r in args.regimes.split(",") if r.strip()]
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]

    for regime in regimes:
        regime_cfg = load_yaml(Path("configs/regimes") / f"{regime}.yaml")
        for variant in variants:
            model_cfg = _load_optional(Path("configs/models") / f"{variant}.yaml")
            cfg = resolve_config(base_cfg, [regime_cfg, model_cfg], args.override)
            cfg["model"]["variant"] = variant
            run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
            cfg["run_name"] = run_name
            cfg["output_dir"] = str(Path("runs") / regime / variant)
            cfg = finalize_config(cfg)

            set_seed(cfg["seed"])
            run_dir = Path(cfg["output_dir"]) / cfg["run_name"]
            ensure_dir(run_dir)
            save_yaml(cfg, run_dir / "config_resolved.yaml")

            print(f"Training {variant} on {regime} -> {run_dir}")
            trainer = Trainer(cfg, run_dir=run_dir)
            trainer.train()


if __name__ == "__main__":
    main()
