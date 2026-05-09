from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Literal

import numpy as np
from sklearn.isotonic import IsotonicRegression

from isgen import load_json, save_json


CalibrationMethod = Literal["linear", "isotonic"]


@dataclass
class BehaviorCalibrator:
    method: CalibrationMethod
    payload: Dict[str, object]

    def predict_sampled(self, target_values: np.ndarray) -> np.ndarray:
        target_values = np.asarray(target_values, dtype=np.float32)
        if self.method == "linear":
            slope = float(self.payload.get("slope", 1.0))
            intercept = float(self.payload.get("intercept", 0.0))
            return slope * target_values + intercept
        xs = np.asarray(self.payload.get("x", []), dtype=np.float32)
        ys = np.asarray(self.payload.get("y", []), dtype=np.float32)
        if xs.size == 0 or ys.size == 0:
            return target_values.copy()
        return np.interp(target_values, xs, ys, left=ys[0], right=ys[-1])

    def invert_target(self, desired_sampled_values: np.ndarray) -> np.ndarray:
        desired_sampled_values = np.asarray(desired_sampled_values, dtype=np.float32)
        if self.method == "linear":
            slope = float(self.payload.get("slope", 1.0))
            intercept = float(self.payload.get("intercept", 0.0))
            if abs(slope) < 1e-6:
                return desired_sampled_values.copy()
            return (desired_sampled_values - intercept) / slope
        xs = np.asarray(self.payload.get("x", []), dtype=np.float32)
        ys = np.asarray(self.payload.get("y", []), dtype=np.float32)
        if xs.size == 0 or ys.size == 0:
            return desired_sampled_values.copy()
        order = np.argsort(ys)
        ys_sorted = ys[order]
        xs_sorted = xs[order]
        ys_unique, unique_idx = np.unique(ys_sorted, return_index=True)
        xs_unique = xs_sorted[unique_idx]
        return np.interp(desired_sampled_values, ys_unique, xs_unique, left=xs_unique[0], right=xs_unique[-1])

    def to_dict(self) -> Dict[str, object]:
        return {"method": self.method, "payload": self.payload}


def fit_behavior_calibrator(
    target_values: Iterable[float],
    sampled_behavior_values: Iterable[float],
    method: CalibrationMethod = "linear",
) -> BehaviorCalibrator:
    x = np.asarray(list(target_values), dtype=np.float32)
    y = np.asarray(list(sampled_behavior_values), dtype=np.float32)
    if x.size == 0 or y.size == 0:
        return BehaviorCalibrator(method=method, payload={})
    if method == "linear":
        if x.size == 1 or float(np.std(x)) < 1e-8 or float(np.std(y)) < 1e-8:
            slope = 1.0
            intercept = float(np.mean(y - x))
        else:
            try:
                slope, intercept = np.polyfit(x, y, deg=1)
            except (np.linalg.LinAlgError, ValueError, FloatingPointError):
                slope = 1.0
                intercept = float(np.mean(y - x))
        return BehaviorCalibrator(method="linear", payload={"slope": float(slope), "intercept": float(intercept)})
    model = IsotonicRegression(out_of_bounds="clip", increasing=True)
    y_fit = model.fit_transform(x, y)
    order = np.argsort(x)
    return BehaviorCalibrator(
        method="isotonic",
        payload={
            "x": x[order].tolist(),
            "y": np.asarray(y_fit, dtype=np.float32)[order].tolist(),
        },
    )


def load_behavior_calibrator(path: str | Path) -> BehaviorCalibrator:
    payload = load_json(path)
    return BehaviorCalibrator(method=str(payload["method"]), payload=dict(payload["payload"]))


def save_behavior_calibrator(calibrator: BehaviorCalibrator, path: str | Path) -> None:
    save_json(calibrator.to_dict(), path)


def calibration_error_summary(target_values: Iterable[float], sampled_values: Iterable[float], calibrator: BehaviorCalibrator) -> Dict[str, float]:
    target = np.asarray(list(target_values), dtype=np.float32)
    sampled = np.asarray(list(sampled_values), dtype=np.float32)
    if target.size == 0 or sampled.size == 0:
        return {"before_mae": 0.0, "after_mae": 0.0}
    conditioned = calibrator.invert_target(target)
    predicted = calibrator.predict_sampled(conditioned)
    return {
        "before_mae": float(np.mean(np.abs(sampled - target))),
        "after_mae": float(np.mean(np.abs(predicted - target))),
    }
