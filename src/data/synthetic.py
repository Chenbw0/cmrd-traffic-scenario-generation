from __future__ import annotations

import uuid
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from isgen.data.map_reader import synthetic_map


def generate_synthetic_track_dataframe(num_cases: int = 12, future_only_test: bool = False) -> Dict[str, pd.DataFrame]:
    splits = {"train": [], "val": [], "test": []}
    for split_name, case_indices in {"train": range(0, num_cases), "val": range(num_cases, num_cases + 4), "test": range(num_cases + 4, num_cases + 8)}.items():
        rows: List[Dict] = []
        for case_offset, case_id in enumerate(case_indices):
            difficulty = 0.1 + 0.8 * ((case_id % 6) / 5.0)
            approach_speed = 4.0 + 8.0 * difficulty
            crossing_offset = (0.5 - difficulty) * 10.0
            timestamps = np.arange(0, 4000, 100)
            for timestamp_ms in timestamps:
                t = timestamp_ms / 1000.0
                rows.append(
                    {
                        "case_id": str(case_id),
                        "track_id": 1,
                        "frame_id": int(timestamp_ms / 100) + 1,
                        "timestamp_ms": int(timestamp_ms),
                        "agent_type": "car",
                        "x": -20.0 + approach_speed * t,
                        "y": 0.0,
                        "vx": approach_speed,
                        "vy": 0.0,
                        "heading": 0.0,
                        "length": 4.8,
                        "width": 2.0,
                    }
                )
                rows.append(
                    {
                        "case_id": str(case_id),
                        "track_id": 2,
                        "frame_id": int(timestamp_ms / 100) + 1,
                        "timestamp_ms": int(timestamp_ms),
                        "agent_type": "car",
                        "x": crossing_offset,
                        "y": 20.0 - approach_speed * t,
                        "vx": 0.0,
                        "vy": -approach_speed,
                        "heading": -np.pi / 2.0,
                        "length": 4.8,
                        "width": 2.0,
                    }
                )
                rows.append(
                    {
                        "case_id": str(case_id),
                        "track_id": 3,
                        "frame_id": int(timestamp_ms / 100) + 1,
                        "timestamp_ms": int(timestamp_ms),
                        "agent_type": "pedestrian/bicycle",
                        "x": -10.0,
                        "y": -12.0 + 1.5 * t,
                        "vx": 0.0,
                        "vy": 1.5,
                        "heading": np.pi / 2.0,
                        "length": 1.0,
                        "width": 0.8,
                    }
                )
        frame = pd.DataFrame(rows)
        if split_name == "test" and future_only_test:
            max_timestamp = frame["timestamp_ms"].max()
            frame = frame[frame["timestamp_ms"] <= max_timestamp / 2].copy()
        frame["scenario_id"] = frame["case_id"].map(lambda item: f"synthetic_intersection:{item}")
        frame["location_id"] = "synthetic_intersection"
        splits[split_name] = frame
    return splits


def create_synthetic_dataset_root(root: str | Path) -> None:
    root_path = Path(root)
    (root_path / "train").mkdir(parents=True, exist_ok=True)
    (root_path / "val").mkdir(parents=True, exist_ok=True)
    (root_path / "test_multi-agent").mkdir(parents=True, exist_ok=True)
    (root_path / "maps").mkdir(parents=True, exist_ok=True)
    splits = generate_synthetic_track_dataframe()
    splits["train"].to_csv(root_path / "train" / "synthetic_intersection_train.csv", index=False)
    splits["val"].to_csv(root_path / "val" / "synthetic_intersection_val.csv", index=False)
    splits["test"].to_csv(root_path / "test_multi-agent" / "synthetic_intersection_obs.csv", index=False)
    map_data = synthetic_map()
    osm_xy_path = root_path / "maps" / "synthetic_intersection.osm_xy"
    tmp_path = root_path / "maps" / f"synthetic_intersection.{uuid.uuid4().hex}.osm_xy.tmp"
    with tmp_path.open("w", encoding="utf-8") as handle:
        handle.write("<osm_xy>\n")
        node_id = 1000
        way_id = 2000
        for polyline in map_data.polyline_list:
            node_ids = []
            for point in polyline:
                handle.write(f'  <node id="{node_id}" x="{point[0]:.3f}" y="{point[1]:.3f}" />\n')
                node_ids.append(node_id)
                node_id += 1
            handle.write(f'  <way id="{way_id}">\n')
            for ref in node_ids:
                handle.write(f'    <nd ref="{ref}" />\n')
            handle.write('    <tag k="type" v="line_thin" />\n')
            handle.write("  </way>\n")
            way_id += 1
        handle.write("</osm_xy>\n")
    try:
        tmp_path.replace(osm_xy_path)
    except PermissionError:
        if osm_xy_path.exists() and osm_xy_path.stat().st_size > 0:
            tmp_path.unlink(missing_ok=True)
        else:
            raise
