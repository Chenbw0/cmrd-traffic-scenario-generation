from __future__ import annotations

import logging
import xml.etree.ElementTree as ET
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from isgen.data import MapData
from isgen.data.transforms import resample_polyline

LOGGER = logging.getLogger(__name__)
MAP_READER_VERSION = "2026-04-19-osm-cache-v1"

MAP_TYPE_TO_ID = {
    "centerline": 0,
    "curbstone": 1,
    "line_thin": 2,
    "line_thick": 3,
    "virtual": 4,
    "pedestrian_marking": 5,
    "unknown": 6,
}


def _parse_xml_map(path: Path) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, Dict[str, str]], List[Dict[str, str]]]:
    root = ET.parse(path).getroot()
    nodes: Dict[str, np.ndarray] = {}
    ways: Dict[str, np.ndarray] = {}
    way_types: Dict[str, Dict[str, str]] = {}
    relations: List[Dict[str, str]] = []
    for element in root:
        if element.tag == "node":
            node_id = element.attrib["id"]
            if "x" in element.attrib and "y" in element.attrib:
                nodes[node_id] = np.asarray([float(element.attrib["x"]), float(element.attrib["y"])], dtype=np.float32)
            elif "lon" in element.attrib and "lat" in element.attrib:
                nodes[node_id] = np.asarray([float(element.attrib["lon"]), float(element.attrib["lat"])], dtype=np.float32)
        elif element.tag == "way":
            way_id = element.attrib["id"]
            refs: List[np.ndarray] = []
            tags: Dict[str, str] = {}
            for child in element:
                if child.tag == "nd":
                    ref = child.attrib["ref"]
                    if ref in nodes:
                        refs.append(nodes[ref])
                elif child.tag == "tag":
                    tags[child.attrib.get("k", "")] = child.attrib.get("v", "")
            if refs:
                ways[way_id] = np.stack(refs, axis=0)
                way_types[way_id] = tags
        elif element.tag == "relation":
            left = None
            right = None
            tags: Dict[str, str] = {}
            for child in element:
                if child.tag == "member":
                    if child.attrib.get("role") == "left":
                        left = child.attrib.get("ref")
                    elif child.attrib.get("role") == "right":
                        right = child.attrib.get("ref")
                elif child.tag == "tag":
                    tags[child.attrib.get("k", "")] = child.attrib.get("v", "")
            if left is not None and right is not None:
                relations.append({"left": left, "right": right, **tags})
    return nodes, ways, way_types, relations


def _polyline_heading(points: np.ndarray) -> np.ndarray:
    if len(points) == 1:
        return np.zeros(1, dtype=np.float32)
    deltas = np.diff(points[:, 0:2], axis=0, prepend=points[:1, 0:2])
    return np.arctan2(deltas[:, 1], deltas[:, 0]).astype(np.float32)


def _build_centerline(left: np.ndarray, right: np.ndarray, target_points: int) -> np.ndarray:
    left_rs, _ = resample_polyline(left, target_points)
    right_rs, _ = resample_polyline(right, target_points)
    center_xy = 0.5 * (left_rs[:, 0:2] + right_rs[:, 0:2])
    heading = _polyline_heading(center_xy)
    polyline_type = np.full((target_points, 1), MAP_TYPE_TO_ID["centerline"], dtype=np.float32)
    return np.concatenate([center_xy, heading[:, None], polyline_type], axis=-1)


def _build_way_polyline(points: np.ndarray, tags: Dict[str, str], target_points: int) -> np.ndarray:
    rs, _ = resample_polyline(points[:, 0:2], target_points)
    heading = _polyline_heading(rs)
    type_id = MAP_TYPE_TO_ID.get(tags.get("type", "unknown"), MAP_TYPE_TO_ID["unknown"])
    polyline_type = np.full((target_points, 1), type_id, dtype=np.float32)
    return np.concatenate([rs[:, 0:2], heading[:, None], polyline_type], axis=-1)


@lru_cache(maxsize=64)
def _read_interaction_map_cached(location_id: str, data_root_str: str, target_points: int) -> MapData:
    data_root = Path(data_root_str)
    maps_root = Path(data_root) / "maps"
    xy_path = maps_root / f"{location_id}.osm_xy"
    osm_path = maps_root / f"{location_id}.osm"
    candidate = xy_path if xy_path.exists() else osm_path
    if not candidate.exists():
        LOGGER.warning("Map missing for %s. Falling back to trajectory-only mode.", location_id)
        return MapData(location_id=location_id, polyline_list=[], polyline_type_ids=[], source_path="", map_available=False)
    _, ways, way_types, relations = _parse_xml_map(candidate)
    polylines: List[np.ndarray] = []
    type_ids: List[int] = []
    if relations:
        for relation in relations:
            if relation.get("type") != "lanelet":
                continue
            left = ways.get(relation["left"])
            right = ways.get(relation["right"])
            if left is None or right is None:
                continue
            centerline = _build_centerline(left, right, target_points)
            polylines.append(centerline)
            type_ids.append(MAP_TYPE_TO_ID["centerline"])
    if not polylines:
        for way_id, points in ways.items():
            polylines.append(_build_way_polyline(points, way_types.get(way_id, {}), target_points))
            type_ids.append(int(polylines[-1][0, -1]))
    if not polylines:
        raise ValueError(
            f"Unable to parse map file '{candidate}'. Expected lanelet-style .osm_xy/.osm with nodes/ways/relations "
            "or at least way polylines."
        )
    return MapData(location_id=location_id, polyline_list=polylines, polyline_type_ids=type_ids, source_path=str(candidate))


def read_interaction_map(location_id: str, data_root: str | Path, target_points: int = 20) -> MapData:
    return _read_interaction_map_cached(location_id, str(Path(data_root).resolve()), target_points)


def synthetic_map(location_id: str = "synthetic_intersection", target_points: int = 20) -> MapData:
    polylines = []
    for offset in (-3.5, 0.0, 3.5):
        xs = np.linspace(-50.0, 50.0, target_points, dtype=np.float32)
        ys = np.full_like(xs, offset)
        heading = np.zeros_like(xs)
        polyline_type = np.full_like(xs, MAP_TYPE_TO_ID["centerline"], dtype=np.float32)
        polylines.append(np.stack([xs, ys, heading, polyline_type], axis=-1))
        ys2 = np.linspace(-50.0, 50.0, target_points, dtype=np.float32)
        xs2 = np.full_like(ys2, offset)
        heading2 = np.full_like(xs2, np.pi / 2.0)
        polylines.append(np.stack([xs2, ys2, heading2, polyline_type], axis=-1))
    return MapData(location_id=location_id, polyline_list=polylines, polyline_type_ids=[0] * len(polylines), source_path="synthetic")
