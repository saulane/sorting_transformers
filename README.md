# Sorting Transformers Research Scaffold

Train a transformers model to sort integer sequences. This obviously does not works and is made only for fun and research. If anyone can make this work faster than any serious sorting algorithm, I'll make them chocolat chips cookies.

## Quickstart

Train a single run:

```bash
python scripts/train.py --config configs/base.yaml --override model.variant=baseline_absolute --run_name my_run
```

Train with layered configs (base + regime + model):

```bash
python scripts/train.py --config configs/base.yaml --config configs/regimes/unique.yaml --config configs/models/tiny.yaml
```

Run a sweep over regimes and variants:

```bash
python scripts/train_sweep.py --base_config configs/base.yaml
```

Evaluate a checkpoint by length:

```bash
python scripts/eval_checkpoint.py --ckpt runs/unique/baseline_absolute/<run_name>/checkpoints/best.ckpt
```

Run probes from a checkpoint:

```bash
python scripts/probes/probe_attention_comparison.py --ckpt runs/unique/baseline_absolute/<run_name>/checkpoints/best.ckpt
python scripts/probes/probe_rank_encoding.py --ckpt runs/unique/baseline_absolute/<run_name>/checkpoints/best.ckpt
python scripts/probes/probe_counterfactual_swap.py --ckpt runs/unique/baseline_absolute/<run_name>/checkpoints/best.ckpt
python scripts/probes/probe_activation_patching.py --ckpt runs/unique/baseline_absolute/<run_name>/checkpoints/best.ckpt
python scripts/probes/probe_length_invariance.py --ckpt runs/unique/baseline_absolute/<run_name>/checkpoints/best.ckpt
```

## Config system

- YAML configs live in `configs/`.
- Regime overrides are in `configs/regimes/`.
- Model overrides are in `configs/models/`.
- Use `--override key=value` to modify any field from the CLI.

Resolved configs are saved as `config_resolved.yaml` in each run directory for reproducibility.

## Outputs

Each run directory contains:

- `checkpoints/` with `last.ckpt` and `best.ckpt`
- `logs/` with `train.csv` and `val.csv`
- `eval/` with metrics and figures
- `probes/{probe_name}/` with JSON/CSV and figures

## Tests

Run the minimal sanity tests:

```bash
python -m unittest
```
