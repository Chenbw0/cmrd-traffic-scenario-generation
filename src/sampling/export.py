from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

import torch
from tqdm.auto import tqdm

from isgen import ensure_dir


def save_samples(records: List[Dict[str, Any]], output_dir: str | Path, metadata: Dict[str, Any]) -> None:
    root = ensure_dir(output_dir)
    torch.save(records, root / "samples.pt")
    serialized_rows = []
    for record in tqdm(records, desc="Writing samples.jsonl", unit="sample"):
        serializable = {}
        for key, value in record.items():
            if isinstance(value, torch.Tensor):
                serializable[key] = value.cpu().tolist()
            else:
                serializable[key] = value
        serialized_rows.append(serializable)
    for file_name in ("samples.jsonl", "sample_records.jsonl"):
        with (root / file_name).open("w", encoding="utf-8") as handle:
            for row in serialized_rows:
                handle.write(json.dumps(row) + "\n")
    with (root / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
