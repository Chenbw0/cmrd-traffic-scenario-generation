from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import torch

from isgen import ensure_dir


def save_checkpoint(path: str | Path, payload: Dict[str, Any]) -> None:
    target = Path(path)
    ensure_dir(target.parent)
    torch.save(payload, target)


def load_checkpoint(path: str | Path) -> Dict[str, Any]:
    return torch.load(path, map_location="cpu")


def save_history(path: str | Path, history: list[Dict[str, Any]]) -> None:
    target = Path(path)
    ensure_dir(target.parent)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2)
