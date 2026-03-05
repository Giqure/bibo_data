"""GoalRegion VLN pipeline — data types, scene helpers, viewpoint merge.

Defines ContinuousPath (equally-spaced viewpoints along a NavMesh polyline)
and VLNResult (collection of paths + unified waypoint set).  Also provides
NavMesh baking, room data extraction, and viewpoint de-duplication.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Optional

import carb
import numpy as np
from pxr import Usd, UsdGeom

from syn_utils.models import ObjectState
from vln_synthesize.syn_utils.nav_mesh_wrap import NavMeshWrapper
from syn_utils.goal_region import GoalRegion
from common.scene_utils import _room_prims, _merged_bbox, _rtype


# ══════════════════════════════════════════════════════════════════════════════
# Data classes
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class ContinuousPath:
    """A VLN path: viewpoint positions sampled along a NavMesh polyline."""
    positions: np.ndarray          # (V, 3) viewpoint positions
    distance: float = 0.0         # total path distance in metres
    rooms_visited: list[str] = field(default_factory=list)
    num_rooms: int = 0
    scale: str = "short"
    score: float = 0.0
    goal_region: Optional[GoalRegion] = None
    goal_label: str = ""

    @property
    def length(self) -> int:
        return len(self.positions)

    def reversed(self) -> "ContinuousPath":
        return ContinuousPath(
            positions=self.positions[::-1].copy(), distance=self.distance,
            rooms_visited=self.rooms_visited[::-1], num_rooms=self.num_rooms,
            scale=self.scale, score=self.score,
            goal_region=self.goal_region, goal_label=self.goal_label,
        )

    def to_dict(self) -> dict:
        d: dict = {
            "viewpoints": self.positions.tolist(),
            "distance_m": round(self.distance, 3),
            "rooms": self.rooms_visited,
            "num_rooms": self.num_rooms,
            "scale": self.scale,
            "num_viewpoints": self.length,
        }
        if self.goal_label:
            d["goal_label"] = self.goal_label
        if self.goal_region is not None:
            d["goal_region"] = self.goal_region.to_dict()
        return d


@dataclass
class VLNResult:
    """Complete VLN data — viewpoints derived from paths, no pre-placed graph."""
    waypoints: np.ndarray          # (W, 3) de-duped viewpoints
    meta: list[dict]               # per-waypoint metadata
    paths: list[ContinuousPath]
    connectivity: dict             # adjacency built from paths
    navmesh: NavMeshWrapper
    floor_z: float
    regions: dict[str, GoalRegion] = field(default_factory=dict)

    def summary(self) -> dict:
        sc = Counter(p.scale for p in self.paths)
        rp = sum(1 for p in self.paths if p.goal_region is not None)
        return {
            "num_waypoints": len(self.waypoints), "num_paths": len(self.paths),
            "num_edges": sum(len(v) for v in self.connectivity.values()),
            "paths_short": sc.get("short", 0), "paths_medium": sc.get("medium", 0),
            "paths_long": sc.get("long", 0),
            "navmesh_area_m2": round(self.navmesh.navigable_area_m2(), 1),
            "backend": "navmesh-gr", "num_regions": len(self.regions),
            "paths_with_goal_region": rp,
        }


# ══════════════════════════════════════════════════════════════════════════════
# Geometry helpers
# ══════════════════════════════════════════════════════════════════════════════

def _polyline_distance(pts: list[np.ndarray]) -> float:
    """Total length of a 3-D polyline."""
    return sum(float(np.linalg.norm(pts[i + 1] - pts[i]))
               for i in range(len(pts) - 1))


def _resample_polyline(pts: list[np.ndarray], step: float) -> np.ndarray:
    """Resample a polyline at equal arc-length intervals.

    Always includes the first and last point.  Returns (N, 3).
    """
    if len(pts) < 2:
        return np.array(pts)
    total = _polyline_distance(pts)
    if total < 1e-6:
        return np.array([pts[0]])
    n_seg = max(1, int(round(total / step)))
    actual_step = total / n_seg
    samples: list[np.ndarray] = [pts[0].copy()]
    remaining = actual_step
    for i in range(len(pts) - 1):
        seg = pts[i + 1] - pts[i]
        seg_len = float(np.linalg.norm(seg))
        if seg_len < 1e-9:
            continue
        direction = seg / seg_len
        consumed = 0.0
        while consumed + remaining <= seg_len + 1e-9:
            consumed += remaining
            samples.append(pts[i] + direction * consumed)
            remaining = actual_step
        remaining -= (seg_len - consumed)
    # guarantee last point
    last = pts[-1]
    if np.linalg.norm(samples[-1] - last) > 0.01:
        samples.append(last.copy())
    return np.array(samples)


def _rooms_on_polyline(positions: np.ndarray,
                       room_data: list[dict]) -> list[str]:
    """Determine ordered list of rooms traversed by positions."""
    seen: set[str] = set()
    rooms: list[str] = []
    for pos in positions:
        for rd in room_data:
            mn, mx = rd["min"], rd["max"]
            if (mn[0] <= pos[0] <= mx[0] and mn[1] <= pos[1] <= mx[1]
                    and mn[2] - 0.5 <= pos[2] <= mx[2] + 1.0):
                key = rd["key"]
                if key not in seen:
                    seen.add(key); rooms.append(key)
                break
    return rooms


# ══════════════════════════════════════════════════════════════════════════════
# NavMesh bake + room data
# ══════════════════════════════════════════════════════════════════════════════

def bake_navmesh(
    stage: Usd.Stage,
    agent_radius: float = 0.25,
    agent_height: float = 1.8,
    seed: int = 42,
) -> tuple[NavMeshWrapper, np.ndarray, np.ndarray]:
    """Bake NavMesh for the entire stage.

    Returns (wrapper, scene_min, scene_max).
    """
    bbox_cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    root_bbox = bbox_cache.ComputeWorldBound(
        stage.GetPseudoRoot()).ComputeAlignedRange()
    scene_min = np.array(root_bbox.GetMin())
    scene_max = np.array(root_bbox.GetMax())

    NavMeshWrapper.setup_volume(
        stage, scene_min, scene_max,
        agent_radius=agent_radius, agent_height=agent_height)
    nm = NavMeshWrapper(seed=seed)
    success = nm.bake()
    if not success:
        carb.log_error("NavMesh bake failed")
    else:
        carb.log_info(
            f"NavMesh baked: {nm.navigable_area_m2():.1f} m² navigable")
    return nm, scene_min, scene_max


def _collect_rooms(stage: Usd.Stage, state: dict, *,
                   full_bbox: bool = False) -> list[dict]:
    """Shared room collector.  *full_bbox*=True keeps 3-D min/max."""
    objs = state.get("objs", {})
    bbox_cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    room_data: list[dict] = []
    for key, info in objs.items():
        obj = ObjectState.from_dict(info)
        if not obj.active or obj.category != "room":
            continue
        prims = _room_prims(stage, obj.obj_name or key)
        if not prims:
            continue
        bmin, bmax = _merged_bbox(bbox_cache, prims)
        area = float((bmax[0] - bmin[0]) * (bmax[1] - bmin[1]))
        if area < 0.5:
            continue
        d: dict = {"key": key, "type": _rtype(obj), "area": area}
        if full_bbox:
            d["min"] = bmin.copy(); d["max"] = bmax.copy()
        else:
            d["min_xy"] = bmin[:2].copy(); d["max_xy"] = bmax[:2].copy()
            d["floor_z"] = float(bmin[2])
        room_data.append(d)
    return room_data


def extract_room_data_for_regions(stage: Usd.Stage,
                                  state: dict) -> list[dict]:
    """Room data (2-D bbox) for build_region_catalog."""
    return _collect_rooms(stage, state, full_bbox=False)


def _room_data_full(stage: Usd.Stage, state: dict) -> list[dict]:
    """Room data with full 3-D bboxes (for room detection along paths)."""
    return _collect_rooms(stage, state, full_bbox=True)


# ══════════════════════════════════════════════════════════════════════════════
# Viewpoint merge + connectivity
# ══════════════════════════════════════════════════════════════════════════════

def _merge_viewpoints(paths: list[ContinuousPath],
                      min_dist: float = 0.5,
                      room_data: list[dict] | None = None,
                      ) -> tuple[np.ndarray, list[dict]]:
    """Collect all path viewpoints into a de-duplicated waypoint array.

    Returns (wp[W,3], meta[W]).
    """
    all_pts: list[np.ndarray] = []
    for p in paths:
        for v in p.positions:
            if all_pts and np.min(np.linalg.norm(
                    np.array(all_pts) - v, axis=1)) < min_dist:
                continue
            all_pts.append(v.copy())
    if not all_pts:
        return np.empty((0, 3)), []
    wp = np.array(all_pts)
    meta: list[dict] = []
    for v in wp:
        room_label = "unknown"
        if room_data:
            for rd in room_data:
                mn, mx = rd["min"], rd["max"]
                if mn[0] <= v[0] <= mx[0] and mn[1] <= v[1] <= mx[1]:
                    room_label = rd.get("type") or rd["key"]
                    break
        meta.append({"type": f"room:{room_label}", "room": room_label})
    return wp, meta


def _build_connectivity(wp: np.ndarray, nm: NavMeshWrapper,
                        max_dist: float = 5.0) -> dict[int, list[dict]]:
    """Build adjacency from merged waypoints using NavMesh reachability."""
    W = len(wp)
    conn: dict[int, list[dict]] = {i: [] for i in range(W)}
    for i in range(W):
        for j in range(i + 1, W):
            d = float(np.linalg.norm(wp[i] - wp[j]))
            if d > max_dist:
                continue
            if nm.reachable(wp[i], wp[j]):
                conn[i].append({"index": j, "distance": round(d, 4)})
                conn[j].append({"index": i, "distance": round(d, 4)})
    return conn
