"""Shared types and constants for VLN path sampling.

NavPath, VLNResult, and semantic constants used by both
grid-based and NavMesh-based pipelines.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from syn_utils.models import ROOM_TYPES
from syn_utils.goal_region import GoalRegion

# NOTE: After move to common/, relative import paths stay the same
# since syn_utils is a sibling package at vln_synthesize/ level.

# ── Constants ─────────────────────────────────────────────────────────────────

ROOM_WEIGHT: dict[str, float] = {
    "living-room": 2.0, "kitchen": 1.5, "dining-room": 1.5, "bedroom": 1.2,
    "office": 1.2, "bathroom": 0.6, "closet": 0.4, "balcony": 0.6, "hallway": 1.0,
}
ROOM_SUFFIXES: tuple[str, ...] = ("_floor", "_wall", "_ceiling", "_exterior")

_SKIP_SEM = (
    ROOM_TYPES
    | {"room", "cutter", "door", "window", "entrance", "open",
       "root", "room-node", "room-contour",
       "ground", "second-floor", "third-floor",
       "exterior", "staircase", "visited", "new",
       "ceiling-light", "lighting", "wall-decoration"}
)
_META_SKIP = {
    "object", "real-placeholder", "oversize-placeholder",
    "asset-as-placeholder", "asset-placeholder-for-children",
    "placeholder-bbox", "single-generator",
    "no-rotation", "no-collision", "no-children",
    "access-top", "access-front", "access-any-side", "access-all-sides",
    "access-stand-near", "access-open-door", "access-with-hand",
}


# ── NavPath ───────────────────────────────────────────────────────────────────

@dataclass
class NavPath:
    """A navigation path: ordered sequence of waypoint indices."""
    indices: list[int]
    distance: float = 0.0
    rooms_visited: list[str] = field(default_factory=list)
    num_rooms: int = 0
    scale: str = "short"
    score: float = 0.0
    goal_region: Optional[GoalRegion] = None
    goal_label: str = ""

    @property
    def length(self) -> int:
        return len(self.indices)

    def reached_goal(self, waypoints: np.ndarray, threshold: int = -1) -> bool:
        if self.goal_region is None:
            return True
        if not self.indices:
            return False
        if threshold < 0:
            return self.goal_region.contains_3d(waypoints[self.indices[-1]])
        check = self.indices if threshold == 0 else self.indices[-threshold:]
        return any(self.goal_region.contains_3d(waypoints[i]) for i in check)

    def reversed(self) -> NavPath:
        return NavPath(
            indices=self.indices[::-1], distance=self.distance,
            rooms_visited=self.rooms_visited[::-1], num_rooms=self.num_rooms,
            scale=self.scale, score=self.score,
            goal_region=self.goal_region, goal_label=self.goal_label,
        )

    def to_dict(self) -> dict:
        d: dict = {
            "waypoint_indices": self.indices,
            "distance_m": round(self.distance, 3),
            "rooms": self.rooms_visited,
            "num_rooms": self.num_rooms,
            "scale": self.scale,
            "num_waypoints": self.length,
        }
        if self.goal_label:
            d["goal_label"] = self.goal_label
        if self.goal_region is not None:
            d["goal_region"] = self.goal_region.to_dict()
        return d


# ── VLNResult (grid backend) ─────────────────────────────────────────────────

@dataclass
class VLNResult:
    """Complete VLN data for one scene (grid backend)."""
    waypoints: np.ndarray
    meta: list[dict]
    paths: list[NavPath]
    connectivity: dict
    grid: object  # NavGrid (avoid circular import)
    floor_z: float
    regions: dict[str, GoalRegion] = field(default_factory=dict)

    def summary(self) -> dict:
        sc = Counter(p.scale for p in self.paths)
        rp = sum(1 for p in self.paths if p.goal_region is not None)
        g = self.grid
        return {
            "num_waypoints": len(self.waypoints),
            "num_paths": len(self.paths),
            "num_edges": sum(len(v) for v in self.connectivity.values()),
            "paths_short": sc.get("short", 0),
            "paths_medium": sc.get("medium", 0),
            "paths_long": sc.get("long", 0),
            "grid_free_m2": round(getattr(g, "free_area_m2", 0), 1),
            "grid_components": getattr(g, "num_components", 0),
            "num_regions": len(self.regions),
            "paths_with_goal_region": rp,
        }
