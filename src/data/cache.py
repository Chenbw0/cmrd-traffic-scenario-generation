from __future__ import annotations

import gc
from pathlib import Path
from typing import Any, Dict, List

import torch
from tqdm.auto import tqdm

from isgen import ensure_dir, load_json, save_json

CACHE_LAYOUT_VERSION = "2026-04-20-processing-profile-cache-v4"


def cache_split_path(cache_dir: str | Path, split: str) -> Path:
    return Path(cache_dir) / f"slices_{split}.pt"


def cache_split_shard_prefix(cache_dir: str | Path, split: str) -> Path:
    return Path(cache_dir) / f"slices_{split}"


def _cache_split_shard_paths(cache_dir: str | Path, split: str) -> List[Path]:
    root = Path(cache_dir)
    return sorted(root.glob(f"slices_{split}.part*.pt"))


def cache_ready(cache_dir: str | Path, expected_signature: Dict[str, Any] | None = None) -> bool:
    root = Path(cache_dir)
    split_ready = True
    for split in ("train", "val", "test"):
        split_ready = split_ready and (cache_split_path(root, split).exists() or len(_cache_split_shard_paths(root, split)) > 0)
    files_present = all(
        path.exists()
        for path in (
            root / "metadata.json",
            root / "difficulty_stats.json",
            root / "slice_index.pt",
        )
    ) and split_ready
    if not files_present:
        return False
    if expected_signature is None:
        return True
    try:
        metadata = load_cache_metadata(root)
    except FileNotFoundError:
        return False
    return metadata.get("cache_signature") == expected_signature


def save_slices_to_cache(
    cache_dir: str | Path,
    slices_by_split: Dict[str, List[Dict[str, Any]]],
    metadata: Dict[str, Any],
    shard_size: int | None = None,
    clear_after_save: bool = False,
) -> None:
    cache_root = ensure_dir(cache_dir)
    cache_layout: Dict[str, Dict[str, Any]] = {}
    for split, items in slices_by_split.items():
        single_path = cache_root / f"slices_{split}.pt"
        for stale_path in _cache_split_shard_paths(cache_root, split):
            stale_path.unlink(missing_ok=True)
        if shard_size is None or shard_size <= 0 or len(items) <= shard_size:
            torch.save(items, single_path)
            cache_layout[split] = {"sharded": False, "num_shards": 1, "num_items": len(items)}
            if clear_after_save:
                items.clear()
                gc.collect()
            continue
        single_path.unlink(missing_ok=True)
        num_shards = (len(items) + shard_size - 1) // shard_size
        for shard_idx in tqdm(range(num_shards), desc=f"Saving cache {split}", unit="shard", leave=False):
            start = shard_idx * shard_size
            end = min(len(items), (shard_idx + 1) * shard_size)
            torch.save(items[start:end], cache_root / f"slices_{split}.part{shard_idx:04d}.pt")
        cache_layout[split] = {
            "sharded": True,
            "num_shards": num_shards,
            "num_items": len(items),
            "shard_size": int(shard_size),
        }
        if clear_after_save:
            items.clear()
            gc.collect()
    metadata_to_save = dict(metadata)
    metadata_to_save["cache_layout"] = cache_layout
    save_json(metadata_to_save, cache_root / "metadata.json")


def _maybe_materialize_maps(cache_dir: str | Path, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not items or "map_polyline_indices" not in items[0]:
        return items
    from isgen.data.slice_builder import materialize_cached_map_fields

    metadata = load_cache_metadata(cache_dir)
    data_root = metadata.get("data_root")
    map_cfg = metadata.get("map_materialization", {})
    if not data_root:
        return items
    max_polyline_points = int(map_cfg.get("max_polyline_points", 20))
    return [materialize_cached_map_fields(item, data_root, max_polyline_points) for item in items]


def load_slices_from_cache(cache_dir: str | Path, split: str, materialize_maps: bool = False) -> List[Dict[str, Any]]:
    path = cache_split_path(cache_dir, split)
    if path.exists():
        items = torch.load(path, map_location="cpu", weights_only=False)
        return _maybe_materialize_maps(cache_dir, items) if materialize_maps else items
    shard_paths = _cache_split_shard_paths(cache_dir, split)
    if not shard_paths:
        raise FileNotFoundError(f"Missing cached slices file: {path}")
    items: List[Dict[str, Any]] = []
    for shard_path in tqdm(shard_paths, desc=f"Loading cache {split}", unit="shard", leave=False):
        items.extend(torch.load(shard_path, map_location="cpu", weights_only=False))
    return _maybe_materialize_maps(cache_dir, items) if materialize_maps else items


def load_cache_metadata(cache_dir: str | Path) -> Dict[str, Any]:
    path = Path(cache_dir) / "metadata.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing cache metadata: {path}")
    return load_json(path)


def load_slices_from_shards(
    cache_dir: str | Path,
    expected_shard_fields: Dict[str, Any] | None = None,
    source_root: str | Path | None = None,
) -> tuple[Dict[str, List[Dict[str, Any]]], List[Dict[str, Any]]]:
    shard_dir = Path(cache_dir) / "slice_shards"
    if not shard_dir.exists():
        raise FileNotFoundError(f"Missing slice shard directory: {shard_dir}")
    slices_by_split: Dict[str, List[Dict[str, Any]]] = {"train": [], "val": [], "test": []}
    build_profile_rows: List[Dict[str, Any]] = []
    shard_paths = sorted(shard_dir.glob("*.pt"))
    if not shard_paths:
        raise FileNotFoundError(f"No slice shard files found under: {shard_dir}")
    for shard_path in tqdm(shard_paths, desc="Loading slice shards", unit="shard", leave=False):
        payload = torch.load(shard_path, map_location="cpu", weights_only=False)
        shard_signature = dict(payload.get("shard_signature", {}))
        if source_root is not None:
            source_file = str(shard_signature.get("source_file", ""))
            if not source_file.startswith(str(Path(source_root))):
                continue
        if expected_shard_fields:
            if any(shard_signature.get(key) != value for key, value in expected_shard_fields.items()):
                continue
        profile = dict(payload.get("profile", {}))
        split = str(profile.get("split", "train"))
        if split not in slices_by_split:
            continue
        shard_slices = list(payload.get("slices", []))
        slices_by_split[split].extend(shard_slices)
        build_profile_rows.append(profile)
    return slices_by_split, build_profile_rows
