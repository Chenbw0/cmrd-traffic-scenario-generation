from __future__ import annotations

from typing import Dict, Iterable, List

import numpy as np


def robust_stats(values: Iterable[float]) -> Dict[str, float]:
    array = np.asarray(list(values), dtype=np.float32)
    array = array[np.isfinite(array)]
    if len(array) == 0:
        return {"median": 0.0, "iqr": 1.0, "min": 0.0, "max": 1.0}
    q25, q50, q75 = np.quantile(array, [0.25, 0.5, 0.75])
    return {
        "median": float(q50),
        "iqr": float(max(q75 - q25, 1e-6)),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def robust_scale(value: float, stats: Dict[str, float]) -> float:
    return float((value - stats["median"]) / max(stats["iqr"], 1e-6))


def empirical_cdf_values(values: List[float]) -> Dict[str, List[float]]:
    sorted_values = np.asarray(values, dtype=np.float32)
    sorted_values = sorted_values[np.isfinite(sorted_values)]
    sorted_values = np.sort(sorted_values)
    return {"sorted_values": sorted_values.tolist()}


def empirical_cdf_transform(value: float, sorted_values: List[float]) -> float:
    if not sorted_values:
        return 0.5
    array = np.asarray(sorted_values, dtype=np.float32)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return 0.5
    rank = np.searchsorted(array, value, side="right")
    return float(rank / len(array))
