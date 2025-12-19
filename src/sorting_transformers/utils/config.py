import copy
from pathlib import Path
from typing import Any, Dict, Iterable

import yaml


def load_yaml(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"Config file must be a mapping: {path}")
    return data


def deep_update(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in updates.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def parse_override(override: str) -> tuple[list[str], Any]:
    if "=" not in override:
        raise ValueError(f"Override must be key=value: {override}")
    key, raw_value = override.split("=", 1)
    key_parts = [part for part in key.strip().split(".") if part]
    if not key_parts:
        raise ValueError(f"Override key cannot be empty: {override}")
    try:
        value = yaml.safe_load(raw_value)
    except yaml.YAMLError as exc:
        raise ValueError(f"Failed parsing override value: {override}") from exc
    return key_parts, value


def apply_overrides(config: Dict[str, Any], overrides: Iterable[str]) -> Dict[str, Any]:
    resolved = copy.deepcopy(config)
    for override in overrides:
        key_parts, value = parse_override(override)
        cursor = resolved
        for part in key_parts[:-1]:
            if part not in cursor or not isinstance(cursor[part], dict):
                cursor[part] = {}
            cursor = cursor[part]
        cursor[key_parts[-1]] = value
    return resolved


def save_yaml(config: Dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)


def resolve_config(
    base_config: Dict[str, Any],
    override_configs: Iterable[Dict[str, Any]] | None = None,
    overrides: Iterable[str] | None = None,
) -> Dict[str, Any]:
    resolved = copy.deepcopy(base_config)
    for cfg in override_configs or []:
        resolved = deep_update(resolved, cfg)
    if overrides:
        resolved = apply_overrides(resolved, overrides)
    return resolved


def finalize_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    if cfg.get("run_name") in (None, ""):
        cfg["run_name"] = ""
    if cfg.get("output_dir") in (None, ""):
        cfg["output_dir"] = "runs"
    dataset_cfg = cfg.get("dataset", {})
    if dataset_cfg.get("train_tokens_budget"):
        lengths = dataset_cfg["train_lengths"]
        avg_len = sum(lengths) / len(lengths)
        token_budget = dataset_cfg["train_tokens_budget"]
        tokens_per_example = avg_len + 2
        dataset_cfg["n_train_examples"] = max(
            1, int(token_budget // tokens_per_example)
        )
    train_cfg = cfg.get("train", {})
    if train_cfg.get("max_steps") is None and train_cfg.get("epochs") is None:
        train_cfg["max_steps"] = 1000
    return cfg
