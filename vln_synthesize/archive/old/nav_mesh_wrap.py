"""NavMesh wrapper — core queries (snap, path, reachable).

Sampling, region queries, and geometry export are in nav_mesh_query.py
and are bound to NavMeshWrapper at import time via syn_utils.__init__.
"""
from __future__ import annotations

import math
from typing import Optional

import carb
import numpy as np


def _acquire_nav():
    import omni.anim.navigation.core as nav
    return nav.acquire_interface()

def _float3(xyz) -> "carb.Float3":
    return carb.Float3(float(xyz[0]), float(xyz[1]), float(xyz[2]))

def _to_np(f3) -> np.ndarray:
    return np.array([f3.x, f3.y, f3.z], dtype=np.float64)


class NavMeshWrapper:
    """High-level NavMesh interface for VLN data synthesis."""

    def __init__(self, seed: int = 42):
        self._inav = _acquire_nav()
        self._navmesh = None
        self._seed = seed
        self._session = "vln_sampler"
        self._inav.set_random_seed(self._session, seed)

    # ── Setup & Bake ──────────────────────────────────────────────────────

    @staticmethod
    def setup_volume(stage, scene_min: np.ndarray, scene_max: np.ndarray,
                     agent_radius: float = 0.25, agent_height: float = 1.8,
                     max_step_height: float = 0.3, max_slope: float = 30.0):
        """Create NavMeshVolume prim + write bake settings."""
        import isaacsim.core.utils.prims as prims_utils

        vol_path = "/World/NavMeshVolume"
        if prims_utils.is_prim_path_valid(vol_path):
            carb.log_info(f"NavMesh volume already exists at {vol_path}")
        else:
            pad = 2.0
            extent = (scene_max - scene_min) + 2 * pad
            center = (scene_min + scene_max) / 2
            prims_utils.create_prim(
                prim_path=vol_path,
                prim_type="NavMeshVolume",
                position=center.tolist(),
                scale=extent.tolist(),
                attributes={"volumeType": 0},
            )
        settings = carb.settings.get_settings()
        pfx = "/persistent/exts/omni.anim.navigation.core/navMesh/config"
        settings.set(f"{pfx}/agentRadius", agent_radius)
        settings.set(f"{pfx}/agentHeight", agent_height)
        settings.set(f"{pfx}/agentMaxStepHeight", max_step_height)
        settings.set(f"{pfx}/agentMaxSlope", max_slope)
        settings.set(f"{pfx}/autoRebakeOnChanges", False)
        carb.log_info(f"NavMesh volume at {vol_path}, r={agent_radius} h={agent_height}")

    def bake(self):
        self._inav.start_navmesh_baking_and_wait()
        self._navmesh = self._inav.get_navmesh()
        if self._navmesh is None:
            carb.log_error("NavMesh bake failed"); return False
        carb.log_info(f"NavMesh baked: {self._navmesh.get_area_count()} area(s)")
        return True

    @property
    def mesh(self):
        return self._navmesh

    # ── Point queries ─────────────────────────────────────────────────────

    def is_navigable(self, xyz, tolerance: float = 0.5) -> bool:
        if self._navmesh is None:
            return False
        result = self._navmesh.query_closest_point(target=_float3(xyz))
        if result is None:
            return False
        closest, _ = result
        dist = math.sqrt(sum((a - b) ** 2 for a, b in zip(
            [closest.x, closest.y, closest.z],
            [float(xyz[0]), float(xyz[1]), float(xyz[2])])))
        return dist <= tolerance

    def snap(self, xyz, island_id: int = -1) -> Optional[np.ndarray]:
        if self._navmesh is None:
            return None
        result = self._navmesh.query_closest_point(target=_float3(xyz),
                                                   search_island_id=island_id)
        if result is None:
            return None
        return _to_np(result[0])

    def snap_with_island(self, xyz) -> tuple[Optional[np.ndarray], int]:
        if self._navmesh is None:
            return None, -1
        result = self._navmesh.query_closest_point(target=_float3(xyz))
        if result is None:
            return None, -1
        return _to_np(result[0]), int(result[1])

    def random_point(self) -> Optional[np.ndarray]:
        if self._navmesh is None:
            return None
        p = self._navmesh.query_random_point(self._session)
        return _to_np(p) if p is not None else None

    # ── Path queries ──────────────────────────────────────────────────────

    def shortest_path(self, start: np.ndarray, goal: np.ndarray,
                      agent_radius: float = 0.25, agent_height: float = 1.8,
                      straighten: bool = True) -> Optional[list[np.ndarray]]:
        if self._navmesh is None:
            return None
        po = self._navmesh.query_shortest_path(
            start_pos=_float3(start), end_pos=_float3(goal),
            agent_radius=agent_radius, agent_height=agent_height,
            straighten=straighten)
        if po is None or po.get_point_count() == 0:
            return None
        return [_to_np(p) for p in po.get_points()]

    def path_distance(self, start: np.ndarray, goal: np.ndarray, **kw) -> float:
        if self._navmesh is None:
            return float("inf")
        po = self._navmesh.query_shortest_path(
            start_pos=_float3(start), end_pos=_float3(goal), **kw)
        if po is None or po.get_point_count() == 0:
            return float("inf")
        return float(po.length())

    def reachable(self, p1: np.ndarray, p2: np.ndarray) -> bool:
        _, i1 = self.snap_with_island(p1)
        _, i2 = self.snap_with_island(p2)
        return i1 >= 0 and i1 == i2
