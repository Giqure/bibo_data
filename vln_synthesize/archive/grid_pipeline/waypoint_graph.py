"""Sparse waypoint graph with Floyd-Warshall and region-aware queries.

Edges are built via NavGrid.line_of_sight (Bresenham).
"""
from __future__ import annotations

import carb
import numpy as np

from syn_utils.goal_region import GoalRegion
from common.vln_types import NavPath


class WaypointGraph:
    """Waypoint adjacency graph with all-pairs shortest paths."""

    def __init__(self, waypoints: np.ndarray, meta: list[dict],
                 grid, max_edge_dist: float = 8.0):
        self.waypoints = waypoints
        self.meta = meta
        self.grid = grid
        self.W = len(waypoints)
        self.adj: list[list[tuple[int, float]]] = [[] for _ in range(self.W)]
        self._dist: np.ndarray | None = None
        self._pred: np.ndarray | None = None
        self._build_edges(max_edge_dist)

    def _build_edges(self, max_dist: float):
        pts2d = self.waypoints[:, :2]
        ne = 0
        for i in range(self.W):
            for j in range(i + 1, self.W):
                d = float(np.linalg.norm(pts2d[i] - pts2d[j]))
                if d > max_dist:
                    continue
                if self.grid.line_of_sight(pts2d[i], pts2d[j]):
                    self.adj[i].append((j, d))
                    self.adj[j].append((i, d))
                    ne += 1
        carb.log_info(f"WaypointGraph: {self.W} nodes, {ne} edges")

    # ── Floyd-Warshall ────────────────────────────────────────────────────

    def precompute_shortest_paths(self):
        W = self.W
        dist = np.full((W, W), np.inf)
        pred = np.full((W, W), -1, dtype=np.int32)
        np.fill_diagonal(dist, 0.0)
        for i, nbs in enumerate(self.adj):
            for j, d in nbs:
                if d < dist[i, j]:
                    dist[i, j] = d; pred[i, j] = i
        for k in range(W):
            via_k = dist[:, k:k + 1] + dist[k:k + 1, :]
            better = via_k < dist
            dist = np.where(better, via_k, dist)
            pred = np.where(better, np.broadcast_to(pred[k:k + 1, :], (W, W)), pred)
        self._dist, self._pred = dist, pred
        carb.log_info(f"Floyd-Warshall: {int(np.sum(np.isfinite(dist))) - W} reachable pairs")

    def shortest_distance(self, i: int, j: int) -> float:
        if self._dist is None:
            self.precompute_shortest_paths()
        return float(self._dist[i, j])  # type: ignore

    def shortest_path(self, start: int, goal: int) -> list[int] | None:
        if self._dist is None:
            self.precompute_shortest_paths()
        assert self._dist is not None and self._pred is not None
        if not np.isfinite(self._dist[start, goal]):
            return None
        if start == goal:
            return [start]
        path, cur, visited = [], goal, set()
        while cur != start:
            if cur < 0 or cur in visited:
                return None
            visited.add(cur); path.append(cur)
            cur = int(self._pred[start, cur])
        path.append(start)
        return path[::-1]

    def get_room(self, idx: int) -> str | None:
        return self.meta[idx].get("room")

    def rooms_on_path(self, indices: list[int]) -> list[str]:
        seen: set[str] = set(); rooms: list[str] = []
        for i in indices:
            r = self.get_room(i)
            if r and r not in seen:
                seen.add(r); rooms.append(r)
        return rooms

    def connectivity_r2r(self) -> dict[int, list[dict]]:
        return {i: [{"index": j, "distance": round(d, 4)} for j, d in nbs]
                for i, nbs in enumerate(self.adj)}

    # ── Region queries ────────────────────────────────────────────────────

    def waypoints_in_region(self, region: GoalRegion) -> list[int]:
        return [i for i in range(self.W) if region.contains_3d(self.waypoints[i])]

    def nearest_waypoint_to_region(self, start: int, region: GoalRegion) -> tuple[int, float]:
        if self._dist is None:
            self.precompute_shortest_paths()
        inside = self.waypoints_in_region(region)
        if not inside:
            return -1, float("inf")
        dists = self._dist[start, inside]  # type: ignore
        best = int(np.argmin(dists))
        return inside[best], float(dists[best])

    def path_to_region(self, start: int, region: GoalRegion) -> NavPath | None:
        goal, dist = self.nearest_waypoint_to_region(start, region)
        if goal < 0 or not np.isfinite(dist):
            return None
        indices = self.shortest_path(start, goal)
        if indices is None:
            return None
        rooms = self.rooms_on_path(indices)
        return NavPath(indices=indices, distance=dist,
                       rooms_visited=rooms, num_rooms=len(rooms), goal_region=region)
