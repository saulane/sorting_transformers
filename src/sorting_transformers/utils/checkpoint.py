from pathlib import Path
from typing import Any, Dict, Optional

import torch

from sorting_transformers.utils.config import load_yaml


def load_config_from_checkpoint(ckpt_path: Path, config_path: Optional[str] = None) -> Dict[str, Any]:
    if config_path:
        return load_yaml(config_path)
    resolved_path = ckpt_path.parent.parent / "config_resolved.yaml"
    if resolved_path.exists():
        return load_yaml(resolved_path)
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    return checkpoint["config"]


def load_checkpoint(ckpt_path: Path, device: str = "cpu") -> Dict[str, Any]:
    return torch.load(ckpt_path, map_location=device)
