"""Goal regions for VLN navigation — targets as areas, not points.

GoalRegion ABC + CircleRegion + RectRegion.
Extended types (Polygon, Composite, WaypointSet) live in goal_region_ext.py.
"""
from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from syn_utils.nav_grid import NavGrid
    from vln_synthesize.syn_utils.nav_mesh_wrap import NavMeshWrapper


class GoalRegion(ABC):
    """Abstract goal region — any area the agent should reach."""

    @abstractmethod
    def contains(self, xy: np.ndarray) -> bool: ...
    def contains_3d(self, xyz: np.ndarray) -> bool:
        return self.contains(xyz[:2])
    def contains_batch(self, xys: np.ndarray) -> np.ndarray:
        return np.array([self.contains(xy) for xy in xys], dtype=bool)
    @abstractmethod
    def area(self) -> float: ...
    @abstractmethod
    def bounding_box(self) -> tuple[np.ndarray, np.ndarray]: ...
    @abstractmethod
    def nearest_boundary(self, xy: np.ndarray) -> np.ndarray: ...

    def distance_to(self, xy: np.ndarray) -> float:
        """Signed distance: negative inside, positive outside."""
        d = float(np.linalg.norm(xy[:2] - self.nearest_boundary(xy)))
        return -d if self.contains(xy) else d

    # ── Grid integration ──────────────────────────────────────────────────

    def grid_mask(self, grid: NavGrid) -> np.ndarray:
        """Boolean mask (nx, ny): True for navigable cells inside region."""
        bb_min, bb_max = self.bounding_box()
        i0, j0 = grid.to_ij(bb_min)
        i1, j1 = grid.to_ij(bb_max)
        mask = np.zeros((grid.nx, grid.ny), dtype=bool)
        for i in range(max(0, i0), min(grid.nx, i1 + 1)):
            for j in range(max(0, j0), min(grid.ny, j1 + 1)):
                if grid.grid[i, j] == grid.FLOOR:
                    if self.contains(grid.to_xy((i, j))):
                        mask[i, j] = True
        return mask

    def grid_mask_vectorised(self, grid: NavGrid) -> np.ndarray:
        """Faster grid_mask using batch containment."""
        bb_min, bb_max = self.bounding_box()
        i0, j0 = grid.to_ij(bb_min)
        i1, j1 = grid.to_ij(bb_max)
        i0, i1 = max(0, i0), min(grid.nx - 1, i1)
        j0, j1 = max(0, j0), min(grid.ny - 1, j1)
        ii, jj = np.arange(i0, i1 + 1), np.arange(j0, j1 + 1)
        gi, gj = np.meshgrid(ii, jj, indexing="ij")
        flat_ij = np.column_stack([gi.ravel(), gj.ravel()])
        xys = grid.min_xy + (flat_ij.astype(float) + 0.5) * grid.res
        inside = self.contains_batch(xys)
        floor = np.array([grid.grid[ij[0], ij[1]] == grid.FLOOR for ij in flat_ij], dtype=bool)
        mask = np.zeros((grid.nx, grid.ny), dtype=bool)
        valid = inside & floor
        mask[flat_ij[valid, 0], flat_ij[valid, 1]] = True
        return mask

    # ── NavMesh integration ───────────────────────────────────────────────

    def sample_navigable(self, n: int, nm: NavMeshWrapper,
                         min_dist: float = 0.5, max_attempts: int = 500) -> np.ndarray:
        """Sample n navigable points inside the region using NavMesh."""
        points: list[np.ndarray] = []
        for _ in range(max_attempts):
            if len(points) >= n:
                break
            p = nm.random_point()
            if p is None:
                continue
            if self.contains_3d(p):
                if not points or np.min(np.linalg.norm(np.array(points) - p, axis=1)) >= min_dist:
                    points.append(p)
        return np.array(points) if points else np.empty((0, 3))

    @abstractmethod
    def to_dict(self) -> dict: ...


# ── CircleRegion ──────────────────────────────────────────────────────────────

class CircleRegion(GoalRegion):
    """Circular region — "near object X within R metres." """
    def __init__(self, center: np.ndarray, radius: float):
        self.center = np.asarray(center[:2], dtype=np.float64)
        self.radius = float(radius)

    def contains(self, xy: np.ndarray) -> bool:
        return float(np.linalg.norm(np.asarray(xy[:2]) - self.center)) <= self.radius
    def contains_batch(self, xys: np.ndarray) -> np.ndarray:
        return np.linalg.norm(xys[:, :2] - self.center, axis=1) <= self.radius
    def area(self) -> float:
        return math.pi * self.radius ** 2
    def bounding_box(self):
        r = np.array([self.radius, self.radius])
        return self.center - r, self.center + r
    def nearest_boundary(self, xy: np.ndarray) -> np.ndarray:
        d = np.asarray(xy[:2]) - self.center
        norm = float(np.linalg.norm(d))
        if norm < 1e-9:
            return self.center + np.array([self.radius, 0.0])
        return self.center + d / norm * self.radius
    def to_dict(self):
        return {"type": "circle", "center": self.center.tolist(), "radius": self.radius}
    def __repr__(self):
        return f"CircleRegion(c={self.center}, r={self.radius:.2f})"


# ── RectRegion ────────────────────────────────────────────────────────────────

class RectRegion(GoalRegion):
    """Axis-aligned rectangle — room bounding box."""
    def __init__(self, min_xy: np.ndarray, max_xy: np.ndarray, margin: float = 0.0):
        self.min_xy = np.asarray(min_xy[:2], dtype=np.float64) + margin
        self.max_xy = np.asarray(max_xy[:2], dtype=np.float64) - margin

    def contains(self, xy: np.ndarray) -> bool:
        p = np.asarray(xy[:2])
        return bool(np.all(p >= self.min_xy) and np.all(p <= self.max_xy))
    def contains_batch(self, xys: np.ndarray) -> np.ndarray:
        pts = xys[:, :2]
        return np.all(pts >= self.min_xy, axis=1) & np.all(pts <= self.max_xy, axis=1)
    def area(self) -> float:
        d = self.max_xy - self.min_xy
        return float(max(0.0, d[0]) * max(0.0, d[1]))
    def bounding_box(self):
        return self.min_xy.copy(), self.max_xy.copy()
    def nearest_boundary(self, xy: np.ndarray) -> np.ndarray:
        p = np.clip(np.asarray(xy[:2]), self.min_xy, self.max_xy)
        if self.contains(xy):
            dists = np.array([p[0] - self.min_xy[0], self.max_xy[0] - p[0],
                              p[1] - self.min_xy[1], self.max_xy[1] - p[1]])
            edge = int(np.argmin(dists))
            result = p.copy()
            if edge == 0:   result[0] = self.min_xy[0]
            elif edge == 1: result[0] = self.max_xy[0]
            elif edge == 2: result[1] = self.min_xy[1]
            else:           result[1] = self.max_xy[1]
            return result
        return p
    def to_dict(self):
        return {"type": "rect", "min_xy": self.min_xy.tolist(), "max_xy": self.max_xy.tolist()}
    def __repr__(self):
        return f"RectRegion({self.min_xy} → {self.max_xy})"
