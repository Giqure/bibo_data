"""Extended goal region types: Polygon, Composite, WaypointSet, factories."""
from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

from syn_utils.goal_region import GoalRegion, CircleRegion, RectRegion

if TYPE_CHECKING:
    from syn_utils.nav_grid import NavGrid


class PolygonRegion(GoalRegion):
    """Arbitrary 2D polygon (ray-casting containment)."""

    def __init__(self, vertices: np.ndarray):
        v = vertices[:, :2] if vertices.ndim == 2 and vertices.shape[1] > 2 else vertices
        self.verts = np.asarray(v, dtype=np.float64)
        assert self.verts.ndim == 2 and self.verts.shape[1] == 2
        self._n = len(self.verts)

    def contains(self, xy: np.ndarray) -> bool:
        x, y = float(xy[0]), float(xy[1])
        inside = False
        j = self._n - 1
        for i in range(self._n):
            xi, yi = self.verts[i]
            xj, yj = self.verts[j]
            if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
                inside = not inside
            j = i
        return inside

    def area(self) -> float:
        x, y = self.verts[:, 0], self.verts[:, 1]
        return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))

    def bounding_box(self) -> tuple[np.ndarray, np.ndarray]:
        return self.verts.min(axis=0).copy(), self.verts.max(axis=0).copy()

    def nearest_boundary(self, xy: np.ndarray) -> np.ndarray:
        p = np.asarray(xy[:2], dtype=np.float64)
        best_pt, best_d2 = self.verts[0].copy(), float("inf")
        for i in range(self._n):
            a, b = self.verts[i], self.verts[(i + 1) % self._n]
            ab = b - a
            t = float(np.clip(np.dot(p - a, ab) / max(np.dot(ab, ab), 1e-12), 0, 1))
            proj = a + t * ab
            d2 = float(np.sum((p - proj) ** 2))
            if d2 < best_d2:
                best_d2, best_pt = d2, proj
        return best_pt

    def to_dict(self) -> dict:
        return {"type": "polygon", "vertices": self.verts.tolist()}

    def __repr__(self) -> str:
        return f"PolygonRegion({self._n} vertices, area={self.area():.1f}m²)"


class CompositeRegion(GoalRegion):
    """Union of multiple goal regions."""

    def __init__(self, children: list[GoalRegion]):
        self.children = children

    def contains(self, xy: np.ndarray) -> bool:
        return any(c.contains(xy) for c in self.children)

    def contains_batch(self, xys: np.ndarray) -> np.ndarray:
        result = np.zeros(len(xys), dtype=bool)
        for c in self.children:
            result |= c.contains_batch(xys)
        return result

    def area(self) -> float:
        return sum(c.area() for c in self.children)

    def bounding_box(self) -> tuple[np.ndarray, np.ndarray]:
        bbs = [c.bounding_box() for c in self.children]
        return np.min([b[0] for b in bbs], axis=0), np.max([b[1] for b in bbs], axis=0)

    def nearest_boundary(self, xy: np.ndarray) -> np.ndarray:
        pts = [c.nearest_boundary(xy) for c in self.children]
        dists = [float(np.linalg.norm(p - xy[:2])) for p in pts]
        return pts[int(np.argmin(dists))]

    def grid_mask(self, grid: NavGrid) -> np.ndarray:
        mask = np.zeros((grid.nx, grid.ny), dtype=bool)
        for c in self.children:
            mask |= c.grid_mask(grid)
        return mask

    def to_dict(self) -> dict:
        return {"type": "composite", "children": [c.to_dict() for c in self.children]}

    def __repr__(self) -> str:
        return f"CompositeRegion({len(self.children)} children)"


class WaypointSetRegion(GoalRegion):
    """Union of circles around waypoint indices in a graph."""

    def __init__(self, indices: list[int], radius: float = 1.5):
        self.indices = list(indices)
        self.radius = float(radius)
        self._centers: np.ndarray | None = None

    def bind(self, waypoints: np.ndarray):
        self._centers = waypoints[self.indices, :2].copy()

    def _check(self):
        if self._centers is None:
            raise RuntimeError("WaypointSetRegion.bind() not called")

    def contains(self, xy: np.ndarray) -> bool:
        self._check()
        return bool(np.any(np.linalg.norm(self._centers - xy[:2], axis=1) <= self.radius))

    def contains_batch(self, xys: np.ndarray) -> np.ndarray:
        self._check()
        d = np.linalg.norm(xys[:, None, :2] - self._centers[None, :, :], axis=2)  # type: ignore
        return np.any(d <= self.radius, axis=1)

    def area(self) -> float:
        return len(self.indices) * math.pi * self.radius ** 2

    def bounding_box(self) -> tuple[np.ndarray, np.ndarray]:
        if self._centers is None:
            return np.zeros(2), np.zeros(2)
        r = np.array([self.radius, self.radius])
        return self._centers.min(axis=0) - r, self._centers.max(axis=0) + r

    def nearest_boundary(self, xy: np.ndarray) -> np.ndarray:
        if self._centers is None:
            return np.array(xy[:2])
        p = np.asarray(xy[:2])
        ci = int(np.argmin(np.linalg.norm(self._centers - p, axis=1)))
        c = self._centers[ci]
        d = p - c
        norm = float(np.linalg.norm(d))
        return c + (d / norm * self.radius if norm > 1e-9 else np.array([self.radius, 0.0]))

    def to_dict(self) -> dict:
        return {"type": "waypoint_set", "indices": self.indices, "radius": self.radius}

    def __repr__(self) -> str:
        return f"WaypointSetRegion({len(self.indices)} wp, r={self.radius:.1f})"


# ── Deserialisation ───────────────────────────────────────────────────────────

def deserialize_region(d: dict) -> GoalRegion:
    """Deserialise a GoalRegion from dict."""
    t = d["type"]
    if t == "circle":    return CircleRegion(np.array(d["center"]), d["radius"])
    if t == "rect":      return RectRegion(np.array(d["min_xy"]), np.array(d["max_xy"]))
    if t == "polygon":   return PolygonRegion(np.array(d["vertices"]))
    if t == "composite": return CompositeRegion([deserialize_region(c) for c in d["children"]])
    if t == "waypoint_set": return WaypointSetRegion(d["indices"], d["radius"])
    raise ValueError(f"Unknown GoalRegion type: {t}")


# ── Factory functions ─────────────────────────────────────────────────────────

def room_region(room_min: np.ndarray, room_max: np.ndarray, wall_margin: float = 0.3):
    return RectRegion(room_min, room_max, margin=wall_margin)

def object_region(obj_pos: np.ndarray, approach_radius: float = 1.5):
    return CircleRegion(obj_pos[:2], approach_radius)

def multi_room_region(room_boxes: list[tuple[np.ndarray, np.ndarray]], wall_margin: float = 0.3):
    return CompositeRegion([RectRegion(rmin, rmax, margin=wall_margin) for rmin, rmax in room_boxes])

def object_group_region(positions: list[np.ndarray], approach_radius: float = 1.5):
    return CompositeRegion([CircleRegion(p[:2], approach_radius) for p in positions])
