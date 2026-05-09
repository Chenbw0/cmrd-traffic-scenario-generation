from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

import numpy as np

DEFAULT_LENGTH_M = 4.8
DEFAULT_WIDTH_M = 2.0


@dataclass
class InteractionFile:
    split: str
    location_id: str
    file_path: str
    dataframe: Any
    has_future: bool = True
    missing_field_warnings: List[str] = field(default_factory=list)
    read_time_sec: float = 0.0


@dataclass
class MapData:
    location_id: str
    polyline_list: List[np.ndarray]
    polyline_type_ids: List[int]
    source_path: str
    map_available: bool = True


@dataclass
class ScenarioSlice:
    """Padded scene slice tensors.

    history_states: [A, H, 5]
    history_mask: [A, H]
    current_states: [A, 5]
    future_states: [A, T, 5]
    future_mask: [A, T]
    map_polylines: [P, K, C]
    map_point_mask: [P, K]
    map_polyline_mask: [P]
    agent_mask: [A]
    """

    slice_id: str
    split: str
    location_id: str
    scenario_id: str
    case_id: str
    center_timestamp: int
    focal_agent_id: int
    agent_ids: np.ndarray
    agent_types: np.ndarray
    agent_sizes: np.ndarray
    history_states: np.ndarray
    history_mask: np.ndarray
    current_states: np.ndarray
    future_states: np.ndarray
    future_mask: np.ndarray
    map_polylines: np.ndarray
    map_point_mask: np.ndarray
    map_polyline_mask: np.ndarray
    agent_mask: np.ndarray
    world_origin: np.ndarray
    world_heading: float
    local_coordinates: bool
    metadata: Dict[str, Any] = field(default_factory=dict)
    difficulty_features: Dict[str, float] = field(default_factory=dict)
    difficulty_score: float = 0.0
    difficulty_level: str = "unknown"
    retrieval_embedding: np.ndarray | None = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "slice_id": self.slice_id,
            "split": self.split,
            "location_id": self.location_id,
            "scenario_id": self.scenario_id,
            "case_id": self.case_id,
            "center_timestamp": self.center_timestamp,
            "focal_agent_id": self.focal_agent_id,
            "agent_ids": self.agent_ids,
            "agent_types": self.agent_types,
            "agent_sizes": self.agent_sizes,
            "history_states": self.history_states.astype(np.float32),
            "history_mask": self.history_mask.astype(bool),
            "current_states": self.current_states.astype(np.float32),
            "future_states": self.future_states.astype(np.float32),
            "future_mask": self.future_mask.astype(bool),
            "map_polylines": self.map_polylines.astype(np.float32),
            "map_point_mask": self.map_point_mask.astype(bool),
            "map_polyline_mask": self.map_polyline_mask.astype(bool),
            "agent_mask": self.agent_mask.astype(bool),
            "world_origin": self.world_origin.astype(np.float32),
            "world_heading": float(self.world_heading),
            "local_coordinates": bool(self.local_coordinates),
            "metadata": self.metadata,
            "difficulty_features": self.difficulty_features,
            "difficulty_score": float(self.difficulty_score),
            "difficulty_level": self.difficulty_level,
            "retrieval_embedding": None
            if self.retrieval_embedding is None
            else self.retrieval_embedding.astype(np.float32),
        }
