"""A* pathfinding, Bresenham LOS, and Poisson-disk sampling for NavGrid.

Functions are defined as plain methods and then bound to NavGrid so that
existing code using ``grid.astar_distance(...)`` continues to work.

Import this module (or ``syn_utils``) to activate the bindings.
"""
from __future__ import annotations

import heapq
import math
from typing import Optional

import numpy as np

from syn_utils.nav_grid import NavGrid


# ── Bresenham line & LOS ──────────────────────────────────────────────────────

def _bresenham(i1: int, j1: int, i2: int, j2: int):
    di, dj = abs(i2 - i1), abs(j2 - j1)
    si = 1 if i1 < i2 else -1
    sj = 1 if j1 < j2 else -1
    err = di - dj
    while True:
        yield i1, j1
        if i1 == i2 and j1 == j2:
            break
        e2 = 2 * err
        if e2 > -dj:
            err -= dj; i1 += si
        if e2 < di:
            err += di; j1 += sj


def _line_of_sight(self: NavGrid, xy1, xy2) -> bool:
    i1, j1 = self.to_ij(xy1)
    i2, j2 = self.to_ij(xy2)
    for ci, cj in _bresenham(i1, j1, i2, j2):
        if self.grid[ci, cj] != self.FLOOR:
            return False
    return True


# ── Single-goal A* ───────────────────────────────────────────────────────────

def _astar_distance(self: NavGrid, start_xy, goal_xy) -> float:
    if not self.reachable(start_xy, goal_xy):
        return float("inf")
    si, sj = self.to_ij(start_xy)
    gi, gj = self.to_ij(goal_xy)
    if (si, sj) == (gi, gj):
        return 0.0
    SQRT2 = math.sqrt(2.0)
    def h(i, j):
        dx, dy = abs(i - gi), abs(j - gj)
        return (max(dx, dy) + (SQRT2 - 1) * min(dx, dy)) * self.res
    g_best = {(si, sj): 0.0}
    heap = [(h(si, sj), 0.0, si, sj)]
    while heap:
        _f, g, ci, cj = heapq.heappop(heap)
        if ci == gi and cj == gj:
            return g
        if g > g_best.get((ci, cj), float("inf")):
            continue
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                if di == 0 and dj == 0:
                    continue
                ni, nj = ci + di, cj + dj
                if not (0 <= ni < self.nx and 0 <= nj < self.ny):
                    continue
                if self.grid[ni, nj] != self.FLOOR:
                    continue
                step = SQRT2 if (di != 0 and dj != 0) else 1.0
                ng = g + step * self.res
                if ng < g_best.get((ni, nj), float("inf")):
                    g_best[(ni, nj)] = ng
                    heapq.heappush(heap, (ng + h(ni, nj), ng, ni, nj))
    return float("inf")


def _astar_path(self: NavGrid, start_xy, goal_xy) -> Optional[list[np.ndarray]]:
    if not self.reachable(start_xy, goal_xy):
        return None
    si, sj = self.to_ij(start_xy)
    gi, gj = self.to_ij(goal_xy)
    if (si, sj) == (gi, gj):
        return [self.to_xy((si, sj))]
    SQRT2 = math.sqrt(2.0)
    def h(i, j):
        dx, dy = abs(i - gi), abs(j - gj)
        return (max(dx, dy) + (SQRT2 - 1) * min(dx, dy)) * self.res
    came_from: dict[tuple, tuple | None] = {(si, sj): None}
    g_best = {(si, sj): 0.0}
    heap = [(h(si, sj), 0.0, si, sj)]
    while heap:
        _f, g, ci, cj = heapq.heappop(heap)
        if ci == gi and cj == gj:
            cells, node = [], (gi, gj)
            while node is not None:
                cells.append(node)
                node = came_from.get(node)
            cells.reverse()
            return [self.to_xy(c) for c in cells]
        if g > g_best.get((ci, cj), float("inf")):
            continue
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                if di == 0 and dj == 0:
                    continue
                ni, nj = ci + di, cj + dj
                if not (0 <= ni < self.nx and 0 <= nj < self.ny):
                    continue
                if self.grid[ni, nj] != self.FLOOR:
                    continue
                step = SQRT2 if (di != 0 and dj != 0) else 1.0
                ng = g + step * self.res
                if ng < g_best.get((ni, nj), float("inf")):
                    g_best[(ni, nj)] = ng
                    came_from[(ni, nj)] = (ci, cj)
                    heapq.heappush(heap, (ng + h(ni, nj), ng, ni, nj))
    return None


# ── Multi-goal A* (region) ────────────────────────────────────────────────────

def _astar_to_region(self: NavGrid, start_xy, goal_mask: np.ndarray):
    si, sj = self.to_ij(start_xy)
    if self.grid[si, sj] != self.FLOOR:
        return float("inf"), None
    if goal_mask[si, sj]:
        return 0.0, [self.to_xy((si, sj))]
    if not np.any(goal_mask):
        return float("inf"), None
    gc = np.argwhere(goal_mask)
    gi_min, gj_min = gc.min(axis=0)
    gi_max, gj_max = gc.max(axis=0)
    SQRT2 = math.sqrt(2.0)
    def h(i, j):
        di = max(0, gi_min - i, i - gi_max)
        dj = max(0, gj_min - j, j - gj_max)
        return (max(di, dj) + (SQRT2 - 1) * min(di, dj)) * self.res
    came_from: dict[tuple, tuple | None] = {(si, sj): None}
    g_best = {(si, sj): 0.0}
    heap = [(h(si, sj), 0.0, si, sj)]
    while heap:
        _f, g, ci, cj = heapq.heappop(heap)
        if goal_mask[ci, cj]:
            cells, node = [], (ci, cj)
            while node is not None:
                cells.append(node)
                node = came_from.get(node)
            cells.reverse()
            return g, [self.to_xy(c) for c in cells]
        if g > g_best.get((ci, cj), float("inf")):
            continue
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                if di == 0 and dj == 0:
                    continue
                ni, nj = ci + di, cj + dj
                if not (0 <= ni < self.nx and 0 <= nj < self.ny):
                    continue
                if self.grid[ni, nj] != self.FLOOR:
                    continue
                sc = SQRT2 if (di != 0 and dj != 0) else 1.0
                ng = g + sc * self.res
                if ng < g_best.get((ni, nj), float("inf")):
                    g_best[(ni, nj)] = ng
                    came_from[(ni, nj)] = (ci, cj)
                    heapq.heappush(heap, (ng + h(ni, nj), ng, ni, nj))
    return float("inf"), None


def _astar_distance_to_region(self: NavGrid, start_xy, goal_mask: np.ndarray) -> float:
    d, _ = _astar_to_region(self, start_xy, goal_mask)
    return d


# ── Sampling ──────────────────────────────────────────────────────────────────

def _sample_poisson(self: NavGrid, n: int, min_dist: float,
                    weight_map=None, component=None, seed=None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    mask = self.free_mask.copy()
    if component is not None:
        mask &= (self.labels == component)
    cells = np.argwhere(mask)
    if len(cells) == 0:
        return np.empty((0, 2))
    w = weight_map[cells[:, 0], cells[:, 1]].astype(float) if weight_map is not None else np.ones(len(cells))
    w = np.maximum(w, 1e-8)
    probs = w / w.sum()
    points: list[np.ndarray] = []
    for _ in range(n * 60):
        if len(points) >= n:
            break
        idx = rng.choice(len(cells), p=probs)
        xy = self.to_xy(cells[idx])
        if not points or np.min(np.linalg.norm(np.asarray(points) - xy, axis=1)) >= min_dist:
            points.append(xy)
    return np.array(points) if points else np.empty((0, 2))


def _sample_in_rect(self: NavGrid, n: int, min_dist: float,
                    rect_min: np.ndarray, rect_max: np.ndarray,
                    weight: float = 1.0, seed=None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    i0, j0 = self.to_ij(rect_min)
    i1, j1 = self.to_ij(rect_max)
    sub = np.zeros((self.nx, self.ny), dtype=bool)
    sub[i0:i1 + 1, j0:j1 + 1] = True
    sub &= self.free_mask
    cells = np.argwhere(sub)
    if len(cells) == 0:
        return np.empty((0, 2))
    points: list[np.ndarray] = []
    for idx in rng.permutation(len(cells)):
        if len(points) >= n:
            break
        xy = self.to_xy(cells[idx])
        if not points or np.min(np.linalg.norm(np.asarray(points) - xy, axis=1)) >= min_dist:
            points.append(xy)
    return np.array(points) if points else np.empty((0, 2))


# ── Bind to NavGrid ──────────────────────────────────────────────────────────

NavGrid._bresenham = staticmethod(_bresenham)  # type: ignore[attr-defined]
NavGrid.line_of_sight = _line_of_sight  # type: ignore[attr-defined]
NavGrid.astar_distance = _astar_distance  # type: ignore[attr-defined]
NavGrid.astar_path = _astar_path  # type: ignore[attr-defined]
NavGrid.astar_to_region = _astar_to_region  # type: ignore[attr-defined]
NavGrid.astar_distance_to_region = _astar_distance_to_region  # type: ignore[attr-defined]
NavGrid.sample_poisson = _sample_poisson  # type: ignore[attr-defined]
NavGrid.sample_in_rect = _sample_in_rect  # type: ignore[attr-defined]
