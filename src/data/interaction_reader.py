from __future__ import annotations

import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from isgen.data import DEFAULT_LENGTH_M, DEFAULT_WIDTH_M, InteractionFile

LOGGER = logging.getLogger(__name__)
INTERACTION_READER_VERSION = "2026-04-20-selective-read-v3"

TRACK_COLUMN_ALIASES = {
    "case_id": ["case_id", "scenario_id", "scene_id"],
    "track_id": ["track_id", "agent_id", "id"],
    "frame_id": ["frame_id", "frame", "timestep"],
    "timestamp_ms": ["timestamp_ms", "timestamp", "time_ms"],
    "agent_type": ["agent_type", "object_type", "type"],
    "x": ["x", "position_x", "pos_x"],
    "y": ["y", "position_y", "pos_y"],
    "vx": ["vx", "velocity_x", "vel_x"],
    "vy": ["vy", "velocity_y", "vel_y"],
    "psi_rad": ["psi_rad", "heading", "heading_rad", "yaw"],
    "length": ["length", "agent_length"],
    "width": ["width", "agent_width"],
}


@dataclass
class ScanResult:
    files: List[InteractionFile]
    warnings: List[str]


def _find_first_column(columns: Iterable[str], aliases: List[str]) -> str | None:
    lowered = {column.lower(): column for column in columns}
    for alias in aliases:
        if alias.lower() in lowered:
            return lowered[alias.lower()]
    return None


def _infer_location_id(path: Path) -> str:
    name = path.stem
    for suffix in ("_train", "_val", "_obs", "_test"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def _infer_split(path: Path) -> str:
    parent = path.parent.name.lower()
    if "train" in parent:
        return "train"
    if "val" in parent:
        return "val"
    if "test" in parent:
        return "test"
    name = path.stem.lower()
    if name.endswith("_train"):
        return "train"
    if name.endswith("_val"):
        return "val"
    if name.endswith("_obs") or name.endswith("_test"):
        return "test"
    return "unassigned"


def _resolve_preprocess_workers(value: object | None) -> int:
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered == "auto":
            return max(1, min(8, (os.cpu_count() or 1)))
        try:
            value = int(lowered)
        except ValueError:
            return 1
    if value is None:
        return 1
    try:
        return max(1, int(value))
    except (TypeError, ValueError):
        return 1


def _candidate_read_columns(columns: Iterable[str]) -> list[str]:
    selected: list[str] = []
    seen: set[str] = set()
    column_list = list(columns)
    for aliases in TRACK_COLUMN_ALIASES.values():
        found = _find_first_column(column_list, aliases)
        if found is not None and found not in seen:
            selected.append(found)
            seen.add(found)
    return selected


def _csv_dtype_map(columns: Iterable[str]) -> Dict[str, str]:
    numeric_float = {
        "x",
        "y",
        "vx",
        "vy",
        "psi_rad",
        "heading",
        "heading_rad",
        "yaw",
        "length",
        "width",
        "position_x",
        "position_y",
        "pos_x",
        "pos_y",
        "velocity_x",
        "velocity_y",
        "vel_x",
        "vel_y",
        "agent_length",
        "agent_width",
    }
    numeric_int = {"frame_id", "frame", "timestep", "timestamp_ms", "timestamp", "time_ms"}
    dtype_map: Dict[str, str] = {}
    for column in columns:
        lowered = column.lower()
        if lowered in numeric_float:
            dtype_map[column] = "float32"
        elif lowered in numeric_int:
            dtype_map[column] = "float64"
    return dtype_map


def _estimate_velocity(df: pd.DataFrame) -> pd.DataFrame:
    if "vx" in df.columns and "vy" in df.columns and df[["vx", "vy"]].notna().any().all():
        return df
    LOGGER.warning(
        "Velocity missing in %s; estimating causally with backward differences. "
        "Only first-frame edge cases use forward difference fallback.",
        df.attrs.get("source_path", "dataframe"),
    )
    df = df.sort_values(["track_id", "timestamp_ms"]).copy()
    grouped = df.groupby("track_id", sort=False)
    dt_prev = grouped["timestamp_ms"].diff() / 1000.0
    dx_prev = grouped["x"].diff()
    dy_prev = grouped["y"].diff()
    dt_next = (grouped["timestamp_ms"].shift(-1) - df["timestamp_ms"]) / 1000.0
    dx_next = grouped["x"].shift(-1) - df["x"]
    dy_next = grouped["y"].shift(-1) - df["y"]
    dt_prev = dt_prev.replace(0.0, np.nan)
    dt_next = dt_next.replace(0.0, np.nan)
    vx_prev = dx_prev / dt_prev
    vy_prev = dy_prev / dt_prev
    vx_next = dx_next / dt_next
    vy_next = dy_next / dt_next
    df["vx"] = vx_prev.where(vx_prev.notna(), vx_next).fillna(0.0)
    df["vy"] = vy_prev.where(vy_prev.notna(), vy_next).fillna(0.0)
    return df


def _estimate_heading(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    existing_heading = pd.to_numeric(df["heading"], errors="coerce") if "heading" in df.columns else pd.Series(np.nan, index=df.index, dtype=np.float32)
    if existing_heading.notna().all():
        df["heading"] = existing_heading.astype(np.float32)
        return df
    if existing_heading.notna().any():
        LOGGER.warning(
            "Heading partially missing in %s; filling missing values from velocity direction.",
            df.attrs.get("source_path", "dataframe"),
        )
    else:
        LOGGER.warning("Heading missing in %s; estimating from velocity direction.", df.attrs.get("source_path", "dataframe"))
    estimated_heading = pd.Series(np.arctan2(df["vy"].to_numpy(), df["vx"].to_numpy()), index=df.index, dtype=np.float32)
    heading = existing_heading.where(existing_heading.notna(), estimated_heading)
    heading = heading.groupby(df["track_id"], sort=False).ffill()
    heading = heading.groupby(df["track_id"], sort=False).bfill()
    df["heading"] = heading.fillna(0.0).astype(np.float32)
    return df


def _fill_size_defaults(df: pd.DataFrame, warnings: List[str], location_id: str) -> pd.DataFrame:
    df = df.copy()
    if "length" not in df.columns or df["length"].isna().all():
        warnings.append(f"{location_id}: length missing, defaulting to {DEFAULT_LENGTH_M}m")
        df["length"] = DEFAULT_LENGTH_M
    else:
        df["length"] = df["length"].fillna(DEFAULT_LENGTH_M)
    if "width" not in df.columns or df["width"].isna().all():
        warnings.append(f"{location_id}: width missing, defaulting to {DEFAULT_WIDTH_M}m")
        df["width"] = DEFAULT_WIDTH_M
    else:
        df["width"] = df["width"].fillna(DEFAULT_WIDTH_M)
    return df


def standardize_track_dataframe(df: pd.DataFrame, source_path: str, location_id: str) -> tuple[pd.DataFrame, List[str]]:
    rename_map: Dict[str, str] = {}
    warnings: List[str] = []
    for target, aliases in TRACK_COLUMN_ALIASES.items():
        found = _find_first_column(df.columns, aliases)
        if found is not None:
            rename_map[found] = target
    df = df.rename(columns=rename_map).copy()
    required = ["track_id", "x", "y"]
    for column in required:
        if column not in df.columns:
            raise ValueError(f"{source_path} missing required trajectory column '{column}'.")
    if "case_id" not in df.columns:
        warnings.append(f"{location_id}: case_id missing, assigning single case 0.")
        df["case_id"] = 0
    if "frame_id" not in df.columns:
        if "timestamp_ms" in df.columns:
            df["frame_id"] = df.groupby("track_id")["timestamp_ms"].rank(method="dense").astype(int)
        else:
            warnings.append(f"{location_id}: frame_id missing, assigning by row order per agent.")
            df["frame_id"] = df.groupby("track_id").cumcount() + 1
    if "timestamp_ms" not in df.columns:
        df["timestamp_ms"] = (df["frame_id"].astype(float) - 1.0) * 100.0
        warnings.append(f"{location_id}: timestamp missing, inferring from frame_id at 10Hz.")
    if "agent_type" not in df.columns:
        df["agent_type"] = "car"
        warnings.append(f"{location_id}: agent_type missing, defaulting to car.")
    df.attrs["source_path"] = source_path
    df = _estimate_velocity(df)
    if "psi_rad" in df.columns:
        df["heading"] = df["psi_rad"]
    df = _estimate_heading(df)
    df = _fill_size_defaults(df, warnings, location_id)
    standardized = df[
        [
            "case_id",
            "track_id",
            "frame_id",
            "timestamp_ms",
            "agent_type",
            "x",
            "y",
            "vx",
            "vy",
            "heading",
            "length",
            "width",
        ]
    ].copy()
    finite_mask = np.isfinite(standardized[["x", "y", "vx", "vy", "heading"]].to_numpy(dtype=np.float32)).all(axis=1)
    if not finite_mask.all():
        dropped = int((~finite_mask).sum())
        warnings.append(f"{location_id}: dropped {dropped} rows with non-finite x/y/vx/vy/heading after standardization.")
        standardized = standardized.loc[finite_mask].copy()
    standardized["case_id"] = standardized["case_id"].astype(str)
    standardized["track_id"] = standardized["track_id"].astype(int)
    standardized["agent_type"] = standardized["agent_type"].astype(str)
    standardized["scenario_id"] = standardized["case_id"].map(lambda case_id: f"{location_id}:{case_id}")
    standardized["location_id"] = location_id
    return standardized.sort_values(["case_id", "timestamp_ms", "track_id"]).reset_index(drop=True), warnings


def _read_raw_track_file(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        try:
            header = pd.read_csv(path, nrows=0)
        except pd.errors.EmptyDataError:
            return pd.DataFrame()
        usecols = _candidate_read_columns(header.columns)
        if not usecols:
            return pd.DataFrame()
        return pd.read_csv(
            path,
            usecols=usecols,
            dtype=_csv_dtype_map(usecols),
            memory_map=bool(path.stat().st_size > 0),
        )
    if path.suffix.lower() == ".parquet":
        try:
            preview = pd.read_parquet(path)
        except (ValueError, FileNotFoundError):
            raise
        if preview.empty:
            return preview
        usecols = _candidate_read_columns(preview.columns)
        if not usecols:
            return pd.DataFrame()
        if set(usecols) == set(preview.columns):
            return preview
        return pd.read_parquet(path, columns=usecols)
    raise ValueError(f"Unsupported trajectory file format: {path}")


def _read_single_track_file(path: Path) -> tuple[pd.DataFrame, List[str]]:
    try:
        df = _read_raw_track_file(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame(), [f"{path}: empty trajectory file skipped."]
    if df.empty:
        return pd.DataFrame(), [f"{path}: empty trajectory file skipped."]
    return standardize_track_dataframe(df, str(path), _infer_location_id(path))

def _scan_single_candidate(path: Path) -> InteractionFile | None:
    start_time = perf_counter()
    split = _infer_split(path)
    location_id = _infer_location_id(path)
    standardized_df, file_warnings = _read_single_track_file(path)
    if standardized_df.empty:
        return None
    has_future = not path.stem.endswith("_obs")
    return InteractionFile(
        split=split,
        location_id=location_id,
        file_path=str(path),
        dataframe=standardized_df,
        has_future=has_future,
        missing_field_warnings=file_warnings,
        read_time_sec=float(perf_counter() - start_time),
    )


def scan_interaction_directory(data_root: str | Path, preprocess_num_workers: object | None = None) -> ScanResult:
    root = Path(data_root)
    if not root.exists():
        raise FileNotFoundError(f"Data root '{root}' does not exist.")
    warnings: List[str] = []
    files: List[InteractionFile] = []
    candidates = sorted(path for path in [*root.rglob("*.csv"), *root.rglob("*.parquet")] if path.parent.name != "maps")
    if not candidates:
        raise FileNotFoundError(
            f"No trajectory csv/parquet files found under '{root}'. "
            "Expected INTERACTION-like files such as data/train/*.csv, data/val/*.csv or data/test_multi-agent/*.csv."
        )
    num_workers = _resolve_preprocess_workers(preprocess_num_workers)
    if num_workers <= 1 or len(candidates) <= 1:
        for path in tqdm(candidates, desc="Scanning trajectory files", unit="file"):
            interaction_file = _scan_single_candidate(path)
            if interaction_file is None:
                warnings.append(f"{path}: empty trajectory file skipped.")
                continue
            warnings.extend(interaction_file.missing_field_warnings)
            files.append(interaction_file)
        return ScanResult(files=files, warnings=warnings)

    completed: list[tuple[int, InteractionFile | None, Path]] = []
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        future_to_item = {
            executor.submit(_scan_single_candidate, path): (idx, path)
            for idx, path in enumerate(candidates)
        }
        progress = tqdm(total=len(candidates), desc="Scanning trajectory files", unit="file")
        for future in as_completed(future_to_item):
            idx, path = future_to_item[future]
            interaction_file = future.result()
            completed.append((idx, interaction_file, path))
            progress.update(1)
        progress.close()
    completed.sort(key=lambda item: item[0])
    for _, interaction_file, path in completed:
        if interaction_file is None:
            warnings.append(f"{path}: empty trajectory file skipped.")
            continue
        warnings.extend(interaction_file.missing_field_warnings)
        files.append(interaction_file)
    return ScanResult(files=files, warnings=warnings)
