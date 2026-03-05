"""Grid-based VLN path sampler — PathSampler + public API.

Pipeline orchestration: build grid → place waypoints → build region
catalog → build waypoint graph → Floyd-Warshall → sample diverse paths.
"""
from __future__ import annotations

import carb
import numpy as np
from pxr import Usd

from syn_utils.goal_region import GoalRegion
from common.vln_types import NavPath, VLNResult, ROOM_WEIGHT, _SKIP_SEM, _META_SKIP, ROOM_SUFFIXES
from common.scene_utils import (
    _base, _room_prims, _bbox, _merged_bbox, _rtype,
    resolveStatePath, loadStateJson, build_region_catalog,
)
from grid_pipeline.grid_builder import build_nav_grid, place_waypoints
from grid_pipeline.waypoint_graph import WaypointGraph


# ══════════════════════════════════════════════════════════════════════════════
# PathSampler
# ══════════════════════════════════════════════════════════════════════════════

class PathSampler:
    """Generate diverse paths from a WaypointGraph."""

    def __init__(self, graph: WaypointGraph,
                 regions: dict[str, GoalRegion] | None = None,
                 rng: np.random.Generator | None = None):
        self.g = graph
        self.regions = regions or {}
        self.rng = rng or np.random.default_rng(42)

    def sample(self, target_short=0, target_medium=0, target_long=0,
               min_path_len=3, short_max_rooms=1, medium_max_rooms=3,
               min_novelty=0.3) -> list[NavPath]:
        W = self.g.W
        if not target_short:  target_short  = max(20, int(0.5 * W))
        if not target_medium: target_medium = max(30, int(0.8 * W))
        if not target_long:   target_long   = max(15, int(0.3 * W))
        cands = self._generate_candidates(min_path_len)
        short_c, med_c, long_c = [], [], []
        for p in cands:
            if p.num_rooms <= short_max_rooms:     p.scale = "short";  short_c.append(p)
            elif p.num_rooms <= medium_max_rooms:  p.scale = "medium"; med_c.append(p)
            else:                                  p.scale = "long";   long_c.append(p)
        sel  = self._diverse_select(short_c, target_short, min_novelty)
        sel += self._diverse_select(med_c,   target_medium, min_novelty)
        sel += self._diverse_select(long_c,  target_long,   min_novelty)
        aug = []
        for p in sel:
            aug.append(p)
            if p.length >= 4:
                aug.append(p.reversed())
        carb.log_info(f"Paths: s={len(short_c)}→{sum(1 for p in sel if p.scale=='short')}, "
                      f"m={len(med_c)}→{sum(1 for p in sel if p.scale=='medium')}, "
                      f"l={len(long_c)}→{sum(1 for p in sel if p.scale=='long')}  "
                      f"total(+rev)={len(aug)}")
        return aug

    def _generate_candidates(self, min_len):
        g, W = self.g, self.g.W
        cands: list[NavPath] = []
        for i in range(W):
            for j in range(i + 1, W):
                d = g.shortest_distance(i, j)
                if not np.isfinite(d) or d < 1.0:
                    continue
                idx = g.shortest_path(i, j)
                if idx is None or len(idx) < min_len:
                    continue
                rooms = g.rooms_on_path(idx)
                oc = sum(1 for k in idx if g.meta[k].get("type","").startswith("object:"))
                cands.append(NavPath(indices=idx, distance=d, rooms_visited=rooms,
                                     num_rooms=len(rooms),
                                     score=len(idx) + len(rooms)*3 + oc*2 + d*0.1))
        cands += self._random_walk_paths(min_len, count=W * 2)
        if self.regions:
            cands += self._region_targeted_paths(min_len)
        carb.log_info(f"Generated {len(cands)} candidates ({len(self.regions)} regions)")
        return cands

    def _region_targeted_paths(self, min_len):
        g, W, paths = self.g, self.g.W, []
        if W == 0:
            return paths
        for label, region in self.regions.items():
            inside = set(g.waypoints_in_region(region))
            if not inside:
                continue
            for s in range(W):
                if s in inside:
                    continue
                p = g.path_to_region(s, region)
                if p is None or p.length < min_len:
                    continue
                p.goal_region, p.goal_label = region, label
                oc = sum(1 for k in p.indices if g.meta[k].get("type","").startswith("object:"))
                p.score = p.length + p.num_rooms*3 + oc*2 + p.distance*0.1 + 5.0
                paths.append(p)
        return paths

    def _random_walk_paths(self, min_len, count, max_steps=30):
        g, W = self.g, self.g.W
        if W == 0:
            return []
        paths = []
        for _ in range(count):
            start = int(self.rng.integers(0, W))
            vw, vr, path = {start}, set(), [start]
            r = g.get_room(start)
            if r: vr.add(r)
            for _ in range(max_steps):
                cur = path[-1]; prev = path[-2] if len(path) >= 2 else -1
                nb = [(j, d) for j, d in g.adj[cur] if j != prev]
                if not nb: break
                wt = np.array([0.1 if j in vw else (3.0 if (nr := g.get_room(j)) and nr not in vr else 1.0) for j, _ in nb])
                wt /= wt.sum()
                nxt, _ = nb[self.rng.choice(len(nb), p=wt)]
                path.append(nxt); vw.add(nxt)
                nr = g.get_room(nxt)
                if nr: vr.add(nr)
            if len(path) >= min_len:
                rooms = g.rooms_on_path(path)
                td = sum(d if np.isfinite(d := g.shortest_distance(path[k], path[k+1])) else 0 for k in range(len(path)-1))
                paths.append(NavPath(indices=path, distance=td, rooms_visited=rooms,
                                     num_rooms=len(rooms), score=len(path)*0.8+len(rooms)*3))
        return paths

    def _diverse_select(self, cands, target, min_novelty):
        if not cands or target <= 0:
            return []
        cands.sort(key=lambda p: p.score, reverse=True)
        sel, cov = [], set()
        for p in cands:
            if len(sel) >= target: break
            ws = set(p.indices)
            nov = len(ws - cov) / len(ws) if ws else 0.0
            if len(sel) < 5 or nov >= min_novelty:
                sel.append(p); cov.update(ws)
        return sel


# ══════════════════════════════════════════════════════════════════════════════
# Public API
# ══════════════════════════════════════════════════════════════════════════════

def sample_vln_paths(
    stage: Usd.Stage, state: dict, *,
    camera_height=1.5, room_wp_base=5, min_wp_distance=1.5,
    max_edge_distance=8.0, grid_resolution=0.15,
    target_short=0, target_medium=0, target_long=0,
    object_goal_radius=1.5, wall_margin=0.3, seed=42,
) -> VLNResult:
    grid, fz, rd = build_nav_grid(stage, state, resolution=grid_resolution)
    wp, meta = place_waypoints(grid, stage, state, rd, fz,
                               camera_height=camera_height, room_wp_base=room_wp_base,
                               min_distance=min_wp_distance)
    if len(wp) == 0:
        carb.log_warn("No waypoints"); return VLNResult(np.empty((0,3)), [], [], {}, grid, fz)
    regions = build_region_catalog(state, rd, stage, wall_margin=wall_margin,
                                   object_radius=object_goal_radius)
    g = WaypointGraph(wp, meta, grid, max_edge_dist=max_edge_distance)
    g.precompute_shortest_paths()
    paths = PathSampler(g, regions=regions, rng=np.random.default_rng(seed)).sample(
        target_short=target_short, target_medium=target_medium, target_long=target_long)
    conn = g.connectivity_r2r()
    result = VLNResult(wp, meta, paths, conn, grid, fz, regions)
    carb.log_info(f"VLN result: {result.summary()}")
    return result


def sampleSemanticWaypoints(stage, state, camera_height=1.5,
                            room_wp_count=5, min_distance=1.5, **kw):
    """Drop-in compat wrapper for waypoint_sampler.sampleSemanticWaypoints."""
    ch = kw.get("cameraHeight", camera_height)
    rwc = kw.get("roomWaypoints", room_wp_count)
    md = kw.get("minWpDistance", min_distance)
    r = sample_vln_paths(stage, state, camera_height=ch, room_wp_base=rwc, min_wp_distance=md)
    return r.waypoints, r.meta
