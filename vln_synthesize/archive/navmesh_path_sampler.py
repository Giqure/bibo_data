"""GoalRegion-based VLN path sampler — direct NavMesh pathfinding.

No waypoint graph, no Floyd-Warshall.  Each path is a NavMesh shortest
path from a random start to a GoalRegion target, with viewpoints sampled
at equal intervals along the continuous 3-D polyline.

Flow: bake NavMesh → build GoalRegion catalog →
      random starts × region targets → NavMesh.find_path →
      resample viewpoints → score / classify / deduplicate.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Optional

import carb
import numpy as np
from pxr import Usd

from vln_synthesize.syn_utils.nav_mesh_wrap import NavMeshWrapper
from syn_utils.goal_region import GoalRegion
from common.scene_utils import build_region_catalog
from navmesh_pipeline.navmesh_placer import bake_navmesh, extract_room_data_for_regions


# ── ContinuousPath ────────────────────────────────────────────────────────────

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


# ── VLNResult (NavMesh GR backend) ────────────────────────────────────────────

@dataclass
class VLNResult:
    """Complete VLN data — viewpoints derived from paths, no pre-placed graph."""
    waypoints: np.ndarray          # (W, 3) union of all path viewpoints (de-duped)
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
# Helpers
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


def _rooms_on_polyline(positions: np.ndarray, room_data: list[dict]) -> list[str]:
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
# GRPathSampler
# ══════════════════════════════════════════════════════════════════════════════

class GRPathSampler:
    """Sample VLN paths as: random start → GoalRegion via NavMesh."""

    def __init__(self, nm: NavMeshWrapper,
                 regions: dict[str, GoalRegion],
                 room_data: list[dict],
                 camera_height: float = 1.5,
                 floor_z: float = 0.0,
                 rng: np.random.Generator | None = None):
        self.nm = nm
        self.regions = regions
        self.room_data = room_data
        self.cam_z = floor_z + camera_height
        self.rng = rng or np.random.default_rng(42)

    # ── Core: find one path ───────────────────────────────────────────────

    def _find_path_to_region(self, start: np.ndarray,
                             region: GoalRegion,
                             n_targets: int = 15) -> Optional[list[np.ndarray]]:
        """NavMesh path from *start* to nearest reachable point in *region*."""
        targets = self.nm.sample_in_region(region, n_targets)
        if len(targets) == 0:
            return None
        best_pts, best_d = None, float("inf")
        for t in targets:
            pts = self.nm.shortest_path(start, t)
            if pts is None:
                continue
            d = _polyline_distance(pts)
            if d < best_d:
                best_d, best_d = d, d   # keep linter happy
                best_pts = pts
                best_d = d
        return best_pts

    def _random_start(self) -> Optional[np.ndarray]:
        """Random navigable point, snapped to camera height."""
        for _ in range(200):
            p = self.nm.random_point()
            if p is not None:
                p[2] = self.cam_z
                return p
        return None

    # ── Generate all candidates ───────────────────────────────────────────

    def generate(self, num_starts: int = 60,
                 viewpoint_step: float = 1.5,
                 min_distance: float = 2.0) -> list[ContinuousPath]:
        """For each random start × each region, compute NavMesh path."""
        cands: list[ContinuousPath] = []
        starts: list[np.ndarray] = []
        for _ in range(num_starts):
            s = self._random_start()
            if s is not None:
                # enforce spread among starts
                if not starts or np.min(np.linalg.norm(
                        np.array(starts) - s, axis=1)) >= viewpoint_step * 0.5:
                    starts.append(s)

        labels = list(self.regions.keys())
        carb.log_info(f"GR sampler: {len(starts)} starts × {len(labels)} regions")

        for si, start in enumerate(starts):
            for label in labels:
                region = self.regions[label]
                # skip if start already inside goal
                if region.contains_3d(start):
                    continue
                raw = self._find_path_to_region(start, region)
                if raw is None or len(raw) < 2:
                    continue
                dist = _polyline_distance(raw)
                if dist < min_distance:
                    continue
                # set z to camera height
                for p in raw:
                    p[2] = self.cam_z
                vps = _resample_polyline(raw, viewpoint_step)
                rooms = _rooms_on_polyline(vps, self.room_data)
                n_rooms = len(rooms)
                score = len(vps) + n_rooms * 3 + dist * 0.1 + 5.0
                cands.append(ContinuousPath(
                    positions=vps, distance=dist,
                    rooms_visited=rooms, num_rooms=n_rooms, score=score,
                    goal_region=region, goal_label=label,
                ))
        carb.log_info(f"Generated {len(cands)} GR path candidates")
        return cands

    # ── Select diverse subset ─────────────────────────────────────────────

    def sample(self, num_starts: int = 60,
               viewpoint_step: float = 1.5,
               min_distance: float = 2.0,
               target_short: int = 30,
               target_medium: int = 40,
               target_long: int = 20,
               short_max_rooms: int = 1,
               medium_max_rooms: int = 3,
               min_novelty: float = 0.3) -> list[ContinuousPath]:
        cands = self.generate(num_starts, viewpoint_step, min_distance)
        short_c, med_c, long_c = [], [], []
        for p in cands:
            if p.num_rooms <= short_max_rooms:
                p.scale = "short"; short_c.append(p)
            elif p.num_rooms <= medium_max_rooms:
                p.scale = "medium"; med_c.append(p)
            else:
                p.scale = "long"; long_c.append(p)

        sel  = _diverse_select(short_c, target_short, min_novelty)
        sel += _diverse_select(med_c,   target_medium, min_novelty)
        sel += _diverse_select(long_c,  target_long,   min_novelty)

        # reverse augmentation
        aug: list[ContinuousPath] = []
        for p in sel:
            aug.append(p)
            if p.length >= 4:
                aug.append(p.reversed())

        carb.log_info(
            f"GR paths: s={len(short_c)}→{sum(1 for p in sel if p.scale=='short')}, "
            f"m={len(med_c)}→{sum(1 for p in sel if p.scale=='medium')}, "
            f"l={len(long_c)}→{sum(1 for p in sel if p.scale=='long')}  "
            f"total(+rev)={len(aug)}")
        return aug


def _diverse_select(cands: list[ContinuousPath], target: int,
                    min_novelty: float) -> list[ContinuousPath]:
    """Greedy diversity selection based on spatial coverage."""
    if not cands or target <= 0:
        return []
    cands.sort(key=lambda p: p.score, reverse=True)
    sel: list[ContinuousPath] = []
    covered: set[tuple[int, int]] = set()  # discretised 1m cells

    def _cells(p: ContinuousPath) -> set[tuple[int, int]]:
        return {(int(v[0]), int(v[1])) for v in p.positions}

    for p in cands:
        if len(sel) >= target:
            break
        cs = _cells(p)
        nov = len(cs - covered) / len(cs) if cs else 0.0
        if len(sel) < 5 or nov >= min_novelty:
            sel.append(p); covered.update(cs)
    return sel


# ══════════════════════════════════════════════════════════════════════════════
# Merge viewpoints from paths → unified waypoint set
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
            # skip near-duplicates
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
                if (mn[0] <= v[0] <= mx[0] and mn[1] <= v[1] <= mx[1]):
                    room_label = rd.get("type") or rd["key"]
                    break
        meta.append({"type": f"room:{room_label}", "room": room_label})
    return wp, meta


def _build_connectivity(wp: np.ndarray, nm: NavMeshWrapper,
                        max_dist: float = 5.0) -> dict[int, list[dict]]:
    """Build adjacency from the merged waypoint set using NavMesh reachability."""
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


# ══════════════════════════════════════════════════════════════════════════════
# Public API
# ══════════════════════════════════════════════════════════════════════════════

def sample_vln_paths_navmesh(
    stage: Usd.Stage, state: dict, *,
    camera_height: float = 1.5,
    viewpoint_step: float = 1.5,
    num_starts: int = 60,
    agent_radius: float = 0.25,
    agent_height: float = 1.8,
    target_short: int = 30,
    target_medium: int = 40,
    target_long: int = 20,
    object_goal_radius: float = 1.5,
    wall_margin: float = 0.3,
    max_connect_distance: float = 5.0,
    seed: int = 42,
    # legacy kwargs (ignored, for syner2 compat)
    room_wp_base: int = 5,
    min_wp_distance: float = 1.5,
    max_edge_distance: float = 8.0,
    **_kw,
) -> VLNResult:
    """GR-based pipeline: NavMesh → GoalRegions → direct pathfinding.

    No waypoint graph. No Floyd-Warshall.
    Paths go from random start → GoalRegion via NavMesh shortest path.
    Viewpoints are resampled at *viewpoint_step* intervals along each path.
    """
    # ① Bake NavMesh
    nm, scene_min, scene_max = bake_navmesh(
        stage, agent_radius=agent_radius, agent_height=agent_height, seed=seed)
    floor_z = float(scene_min[2])

    # ② Build GoalRegion catalog (rooms + objects)
    room_data_for_regions = extract_room_data_for_regions(stage, state)
    regions = build_region_catalog(
        state, room_data_for_regions, stage,
        wall_margin=wall_margin, object_radius=object_goal_radius)
    if not regions:
        carb.log_warn("No GoalRegions found")
        return VLNResult(np.empty((0, 3)), [], [], {}, nm, floor_z)

    # Build room_data with full bbox for room detection along paths
    from navmesh_pipeline.navmesh_placer import _room_data_full
    room_data = _room_data_full(stage, state)

    # ③ Sample paths: random start → region via NavMesh
    sampler = GRPathSampler(
        nm, regions, room_data,
        camera_height=camera_height, floor_z=floor_z,
        rng=np.random.default_rng(seed))
    paths = sampler.sample(
        num_starts=num_starts, viewpoint_step=viewpoint_step,
        target_short=target_short, target_medium=target_medium,
        target_long=target_long)

    if not paths:
        carb.log_warn("No paths generated")
        return VLNResult(np.empty((0, 3)), [], [], {}, nm, floor_z, regions)

    # ④ Merge viewpoints → de-duped waypoint array (for rendering)
    wp, meta = _merge_viewpoints(paths, min_dist=viewpoint_step * 0.4,
                                 room_data=room_data)

    # ⑤ Build connectivity (for R2R compat export)
    conn = _build_connectivity(wp, nm, max_dist=max_connect_distance)

    result = VLNResult(wp, meta, paths, conn, nm, floor_z, regions)
    carb.log_info(f"NavMesh-GR VLN result: {result.summary()}")
    return result
