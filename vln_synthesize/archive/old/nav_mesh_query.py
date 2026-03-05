"""Sampling, region queries, and geometry export for NavMeshWrapper.

Methods are defined here and bound to NavMeshWrapper at import time.
"""
from __future__ import annotations

import carb
import numpy as np

from vln_synthesize.syn_utils.nav_mesh_wrap import NavMeshWrapper


# ── Sampling ──────────────────────────────────────────────────────────────────

def _sample_near(self: NavMeshWrapper, center: np.ndarray, n: int,
                 min_dist: float, max_radius: float = 5.0) -> np.ndarray:
    points: list[np.ndarray] = []
    for _ in range(n * 200):
        if len(points) >= n:
            break
        p = self.random_point()
        if p is None:
            continue
        if np.linalg.norm(p - center) > max_radius:
            continue
        if not points or np.min(np.linalg.norm(np.array(points) - p, axis=1)) >= min_dist:
            points.append(p)
    return np.array(points) if points else np.empty((0, 3))


# ── Region queries ────────────────────────────────────────────────────────────

def _sample_in_region(self: NavMeshWrapper, region, n: int,
                      min_dist: float = 0.5, max_attempts: int = 800) -> np.ndarray:
    points: list[np.ndarray] = []
    for _ in range(max_attempts):
        if len(points) >= n:
            break
        p = self.random_point()
        if p is None or not region.contains_3d(p):
            continue
        if not points or np.min(np.linalg.norm(np.array(points) - p, axis=1)) >= min_dist:
            points.append(p)
    return np.array(points) if points else np.empty((0, 3))


def _path_to_region(self: NavMeshWrapper, start, region, n_samples=20, **kw):
    targets = self.sample_in_region(region, n_samples)
    if len(targets) == 0:
        return float("inf"), None
    best_d, best_p = float("inf"), None
    for t in targets:
        pp = self.shortest_path(start, t, **kw)
        if pp is None:
            continue
        d = sum(float(np.linalg.norm(pp[k + 1] - pp[k])) for k in range(len(pp) - 1))
        if d < best_d:
            best_d, best_p = d, pp
    return best_d, best_p


def _path_distance_to_region(self: NavMeshWrapper, start, region,
                             n_samples=20, **kw) -> float:
    d, _ = self.path_to_region(start, region, n_samples, **kw)
    return d


def _nearest_in_region(self: NavMeshWrapper, xyz, region, n_samples=30):
    targets = self.sample_in_region(region, n_samples)
    if len(targets) == 0:
        return None
    return targets[int(np.argmin(np.linalg.norm(targets - xyz, axis=1)))].copy()


# ── Geometry ──────────────────────────────────────────────────────────────────

def _navigable_triangles(self: NavMeshWrapper, area_index: int = 0) -> np.ndarray:
    if self._navmesh is None:
        return np.empty((0, 3, 3))
    verts = self._navmesh.get_draw_triangles(area_index)
    if not verts:
        return np.empty((0, 3, 3))
    pts = np.array([[v.x, v.y, v.z] for v in verts])
    n_tris = len(pts) // 3
    return pts[:n_tris * 3].reshape(n_tris, 3, 3)


def _navigable_area_m2(self: NavMeshWrapper) -> float:
    tris = self.navigable_triangles()
    if len(tris) == 0:
        return 0.0
    e1 = tris[:, 1] - tris[:, 0]
    e2 = tris[:, 2] - tris[:, 0]
    return float(np.sum(0.5 * np.linalg.norm(np.cross(e1, e2), axis=1)))


# ── Bind to NavMeshWrapper ───────────────────────────────────────────────────

NavMeshWrapper.sample_near = _sample_near  # type: ignore[attr-defined]
NavMeshWrapper.sample_in_region = _sample_in_region  # type: ignore[attr-defined]
NavMeshWrapper.path_to_region = _path_to_region  # type: ignore[attr-defined]
NavMeshWrapper.path_distance_to_region = _path_distance_to_region  # type: ignore[attr-defined]
NavMeshWrapper.nearest_in_region = _nearest_in_region  # type: ignore[attr-defined]
NavMeshWrapper.navigable_triangles = _navigable_triangles  # type: ignore[attr-defined]
NavMeshWrapper.navigable_area_m2 = _navigable_area_m2  # type: ignore[attr-defined]
