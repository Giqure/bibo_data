"""2D occupancy grid for indoor floor navigation.

Search/sampling extensions (A*, LOS, Poisson-disk) are in nav_grid_search.py
and are bound to NavGrid at import time via syn_utils.__init__.
"""
from __future__ import annotations

from collections import deque
from typing import Optional

import numpy as np


class NavGrid:
    """2D occupancy grid.  FLOOR=0  OBSTACLE=1  VOID=2."""

    FLOOR = 0
    OBSTACLE = 1
    VOID = 2

    def __init__(self, min_xy: np.ndarray, max_xy: np.ndarray,
                 resolution: float = 0.20):
        self.min_xy = np.asarray(min_xy, dtype=np.float64)
        self.max_xy = np.asarray(max_xy, dtype=np.float64)
        self.res = float(resolution)
        extent = self.max_xy - self.min_xy
        self.nx = max(1, int(np.ceil(extent[0] / self.res)))
        self.ny = max(1, int(np.ceil(extent[1] / self.res)))
        self.grid = np.full((self.nx, self.ny), self.VOID, dtype=np.uint8)
        self._labels: Optional[np.ndarray] = None
        self._n_components: int = 0

    # ── Coordinates ───────────────────────────────────────────────────────

    def to_ij(self, xy) -> tuple[int, int]:
        raw = ((np.asarray(xy, dtype=float) - self.min_xy) / self.res).astype(int)
        return int(np.clip(raw[0], 0, self.nx - 1)), int(np.clip(raw[1], 0, self.ny - 1))

    def to_xy(self, ij) -> np.ndarray:
        return self.min_xy + (np.array([ij[0], ij[1]], dtype=float) + 0.5) * self.res

    def _invalidate(self):
        self._labels = None

    # ── Building ──────────────────────────────────────────────────────────

    def mark_floor(self, floor_min: np.ndarray, floor_max: np.ndarray,
                   wall_margin: float = 0.15):
        fmin = np.asarray(floor_min) + wall_margin
        fmax = np.asarray(floor_max) - wall_margin
        if fmin[0] >= fmax[0] or fmin[1] >= fmax[1]:
            return
        i0, j0 = self.to_ij(fmin)
        i1, j1 = self.to_ij(fmax)
        region = self.grid[i0:i1 + 1, j0:j1 + 1]
        region[region == self.VOID] = self.FLOOR
        self._invalidate()

    def mark_obstacle(self, obs_min: np.ndarray, obs_max: np.ndarray,
                      inflation: float = 0.20):
        omin = np.asarray(obs_min) - inflation
        omax = np.asarray(obs_max) + inflation
        i0, j0 = self.to_ij(omin)
        i1, j1 = self.to_ij(omax)
        self.grid[i0:i1 + 1, j0:j1 + 1] = self.OBSTACLE
        self._invalidate()

    # ── Queries ───────────────────────────────────────────────────────────

    def is_free(self, xy) -> bool:
        i, j = self.to_ij(xy)
        return int(self.grid[i, j]) == self.FLOOR

    @property
    def free_mask(self) -> np.ndarray:
        return self.grid == self.FLOOR
    @property
    def free_count(self) -> int:
        return int(np.sum(self.free_mask))
    @property
    def free_area_m2(self) -> float:
        return self.free_count * self.res ** 2

    # ── Connected components (lazy BFS) ───────────────────────────────────

    def _compute_labels(self):
        self._labels = np.full((self.nx, self.ny), -1, dtype=np.int32)
        label = 0
        for i in range(self.nx):
            for j in range(self.ny):
                if self.grid[i, j] == self.FLOOR and self._labels[i, j] < 0:
                    q = deque([(i, j)])
                    self._labels[i, j] = label
                    while q:
                        ci, cj = q.popleft()
                        for di, dj in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                            ni, nj = ci + di, cj + dj
                            if (0 <= ni < self.nx and 0 <= nj < self.ny
                                    and self.grid[ni, nj] == self.FLOOR
                                    and self._labels[ni, nj] < 0):
                                self._labels[ni, nj] = label
                                q.append((ni, nj))
                    label += 1
        self._n_components = label

    @property
    def labels(self) -> np.ndarray:
        if self._labels is None:
            self._compute_labels()
        return self._labels  # type: ignore
    @property
    def num_components(self) -> int:
        _ = self.labels
        return self._n_components
    def component_of(self, xy) -> int:
        i, j = self.to_ij(xy)
        return int(self.labels[i, j])
    def reachable(self, xy1, xy2) -> bool:
        c1, c2 = self.component_of(xy1), self.component_of(xy2)
        return c1 >= 0 and c1 == c2
    def largest_component_id(self) -> int:
        lbl = self.labels
        valid = lbl[lbl >= 0]
        return int(np.argmax(np.bincount(valid))) if len(valid) else -1

    # ── Nearest navigable ─────────────────────────────────────────────────

    def nearest_free(self, xy, max_radius: float = 5.0) -> Optional[np.ndarray]:
        si, sj = self.to_ij(xy)
        if self.grid[si, sj] == self.FLOOR:
            return self.to_xy((si, sj))
        max_steps = int(max_radius / self.res)
        visited = {(si, sj)}
        q: deque[tuple[int, int, int]] = deque([(si, sj, 0)])
        while q:
            ci, cj, d = q.popleft()
            if d > max_steps:
                continue
            if self.grid[ci, cj] == self.FLOOR:
                return self.to_xy((ci, cj))
            for di, dj in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                ni, nj = ci + di, cj + dj
                if 0 <= ni < self.nx and 0 <= nj < self.ny and (ni, nj) not in visited:
                    visited.add((ni, nj))
                    q.append((ni, nj, d + 1))
        return None

    # ── Debug ─────────────────────────────────────────────────────────────

    def to_image(self) -> np.ndarray:
        img = np.zeros((self.nx, self.ny, 3), dtype=np.uint8)
        img[self.grid == self.FLOOR] = [0, 200, 0]
        img[self.grid == self.OBSTACLE] = [200, 0, 0]
        img[self.grid == self.VOID] = [40, 40, 40]
        return img
