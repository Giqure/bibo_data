"""GoalRegion VLN path sampler — random starts → GoalRegions via NavMesh.

No waypoint graph, no Floyd-Warshall.  Each path is a NavMesh shortest
path from a random navigable point to a GoalRegion target, with viewpoints
resampled at equal arc-length intervals along the continuous 3-D polyline.
"""
from __future__ import annotations

from typing import Optional

import carb
import numpy as np
from pxr import Usd

from vln_synthesize.syn_utils.nav_mesh_wrap import NavMeshWrapper
from syn_utils.goal_region import GoalRegion
from common.scene_utils import build_region_catalog

from .gr_types import (
    ContinuousPath, VLNResult,
    _polyline_distance, _resample_polyline, _rooms_on_polyline,
    bake_navmesh, extract_room_data_for_regions, _room_data_full,
    _merge_viewpoints, _build_connectivity,
)


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

    def _find_path_to_region(
        self, start: np.ndarray,
        region: GoalRegion,
        n_targets: int = 15,
    ) -> Optional[list[np.ndarray]]:
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
                if not starts or np.min(np.linalg.norm(
                        np.array(starts) - s, axis=1)) >= viewpoint_step * 0.5:
                    starts.append(s)

        labels = list(self.regions.keys())
        carb.log_info(
            f"GR sampler: {len(starts)} starts × {len(labels)} regions")

        for si, start in enumerate(starts):
            for label in labels:
                region = self.regions[label]
                if region.contains_3d(start):
                    continue
                raw = self._find_path_to_region(start, region)
                if raw is None or len(raw) < 2:
                    continue
                dist = _polyline_distance(raw)
                if dist < min_distance:
                    continue
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
        """Classify → diverse select → reverse augmentation."""
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
            f"GR paths: "
            f"s={len(short_c)}→{sum(1 for p in sel if p.scale=='short')}, "
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
    covered: set[tuple[int, int]] = set()

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
    room_wp_base: int = 5,          # noqa: ARG001
    min_wp_distance: float = 1.5,   # noqa: ARG001
    max_edge_distance: float = 8.0, # noqa: ARG001
    **_kw,
) -> VLNResult:
    """GR-based pipeline: NavMesh → GoalRegions → direct pathfinding.

    Paths go from random starts → GoalRegion targets via NavMesh shortest
    path.  Viewpoints are resampled along each path at *viewpoint_step*
    intervals.  No waypoint graph.  No Floyd-Warshall.
    """
    # ① Bake NavMesh
    nm, scene_min, scene_max = bake_navmesh(
        stage, agent_radius=agent_radius,
        agent_height=agent_height, seed=seed)
    floor_z = float(scene_min[2])

    # ② Build GoalRegion catalog (rooms + objects)
    room_data_for_regions = extract_room_data_for_regions(stage, state)
    regions = build_region_catalog(
        state, room_data_for_regions, stage,
        wall_margin=wall_margin, object_radius=object_goal_radius)
    if not regions:
        carb.log_warn("No GoalRegions found")
        return VLNResult(np.empty((0, 3)), [], [], {}, nm, floor_z)

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
    wp, meta = _merge_viewpoints(
        paths, min_dist=viewpoint_step * 0.4, room_data=room_data)

    # ⑤ Build connectivity (R2R compat)
    conn = _build_connectivity(wp, nm, max_dist=max_connect_distance)

    result = VLNResult(wp, meta, paths, conn, nm, floor_z, regions)
    carb.log_info(f"NavMesh-GR VLN result: {result.summary()}")
    return result
