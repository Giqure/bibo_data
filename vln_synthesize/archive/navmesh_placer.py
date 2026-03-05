"""NavMesh baking + waypoint placement (NavMesh-snapped).

Handles: scene volume setup, Recast bake, room/door/object waypoints
with exact NavMesh snap (no grid approximation).
"""
from __future__ import annotations

import math

import carb
import numpy as np
from pxr import Usd, UsdGeom

from syn_utils.models import ObjectState
from vln_synthesize.syn_utils.nav_mesh_wrap import NavMeshWrapper
from common.vln_types import ROOM_WEIGHT, _SKIP_SEM, _META_SKIP
from common.scene_utils import _room_prims, _merged_bbox, _rtype


# ══════════════════════════════════════════════════════════════════════════════
# Bake
# ══════════════════════════════════════════════════════════════════════════════

def bake_navmesh(
    stage: Usd.Stage,
    agent_radius: float = 0.25,
    agent_height: float = 1.8,
    seed: int = 42,
) -> tuple[NavMeshWrapper, np.ndarray, np.ndarray]:
    """Bake NavMesh for the entire stage.  Returns (wrapper, scene_min, scene_max)."""
    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    root_bbox = bbox_cache.ComputeWorldBound(stage.GetPseudoRoot()).ComputeAlignedRange()
    scene_min = np.array(root_bbox.GetMin())
    scene_max = np.array(root_bbox.GetMax())

    NavMeshWrapper.setup_volume(stage, scene_min, scene_max,
                                agent_radius=agent_radius, agent_height=agent_height)
    nm = NavMeshWrapper(seed=seed)
    success = nm.bake()
    if not success:
        carb.log_error("NavMesh bake failed")
    else:
        carb.log_info(f"NavMesh baked: {nm.navigable_area_m2():.1f} m² navigable")
    return nm, scene_min, scene_max


# ══════════════════════════════════════════════════════════════════════════════
# Place waypoints
# ══════════════════════════════════════════════════════════════════════════════

def place_waypoints_navmesh(
    nm: NavMeshWrapper, stage: Usd.Stage, state: dict,
    camera_height: float = 1.5, room_wp_base: int = 5,
    min_distance: float = 1.5,
) -> tuple[np.ndarray, list[dict], list[dict]]:
    """Place semantic waypoints snapped to NavMesh.

    Returns (wp3d[W,3], meta[W], room_data).
    """
    objs = state.get("objs", {})
    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    pts: list[np.ndarray] = []
    meta: list[dict] = []

    def _accept(pos, min_d):
        if not pts:
            return True
        return float(np.min(np.linalg.norm(np.array(pts) - pos, axis=1))) >= min_d

    # ── Collect rooms ─────────────────────────────────────────────────────
    room_data: list[dict] = []
    floor_zs: list[float] = []
    for key, info in objs.items():
        obj = ObjectState.from_dict(info)
        if not obj.active or obj.category != "room":
            continue
        prims = _room_prims(stage, obj.obj_name or key)
        if not prims:
            continue
        bmin, bmax = _merged_bbox(bbox_cache, prims)
        area = float((bmax[0] - bmin[0]) * (bmax[1] - bmin[1]))
        if area < 0.5:
            continue
        floor_z = float(bmin[2])
        floor_zs.append(floor_z)
        room_data.append({"key": key, "type": _rtype(obj),
                          "min": bmin, "max": bmax, "area": area, "floor_z": floor_z})

    global_floor_z = float(min(floor_zs)) if floor_zs else 0.0
    cam_z = global_floor_z + camera_height

    # ── Room waypoints ────────────────────────────────────────────────────
    for rd in room_data:
        weight = ROOM_WEIGHT.get(rd["type"] or "", 1.0)
        n = max(1, int(room_wp_base * weight * max(1.0, math.sqrt(rd["area"]) / 2)))
        center = (rd["min"] + rd["max"]) / 2; center[2] = cam_z
        room_pts = nm.sample_near(center, n * 3, min_distance * 0.8,
                                  max_radius=float(np.linalg.norm(rd["max"] - rd["min"])))
        added = 0
        margin = 0.15
        for p in room_pts:
            if added >= n:
                break
            if (p[0] < rd["min"][0] + margin or p[0] > rd["max"][0] - margin or
                    p[1] < rd["min"][1] + margin or p[1] > rd["max"][1] - margin):
                continue
            p[2] = cam_z
            if _accept(p, min_distance):
                pts.append(p.copy()); added += 1
                meta.append({"type": f"room:{rd['type'] or '?'}", "room": rd["key"],
                             "room_type": rd["type"] or "unknown"})
        carb.log_info(f"Room '{rd['key']}'({rd['type']}): {rd['area']:.0f}m² → {added} wp")

    # ── Door / entrance waypoints ─────────────────────────────────────────
    for key, info in objs.items():
        obj = ObjectState.from_dict(info)
        if not obj.active or obj.category != "door":
            continue
        sems = obj.tags.get("Semantics", [])
        label = "entrance" if "entrance" in sems else "door"
        if len(obj.connected_rooms) < 2:
            continue
        centres: list[np.ndarray] = []
        for rname in obj.connected_rooms[:2]:
            rinfo = objs.get(rname, {})
            if not rinfo:
                continue
            robj = ObjectState.from_dict(rinfo)
            rprims = _room_prims(stage, robj.obj_name or rname)
            if rprims:
                rmin, rmax = _merged_bbox(bbox_cache, rprims)
                centres.append((rmin + rmax) / 2)
        if len(centres) == 2:
            mid = (centres[0] + centres[1]) / 2; mid[2] = cam_z
            snapped = nm.snap(mid)
            if snapped is not None:
                snapped[2] = cam_z
                if _accept(snapped, min_distance * 0.8):
                    pts.append(snapped)
                    meta.append({"type": f"transition:{label}", "room": key,
                                 "connects": obj.connected_rooms[:2]})

    # ── Object waypoints ──────────────────────────────────────────────────
    for key, info in objs.items():
        obj = ObjectState.from_dict(info)
        if not obj.active:
            continue
        sems = set(obj.tags.get("Semantics", []))
        if sems & _SKIP_SEM:
            continue
        dof = info.get("dof_matrix_translation")
        if not dof or len(dof) < 3:
            continue
        try:
            tx, ty, tz = float(dof[0][3]), float(dof[1][3]), float(dof[2][3])
        except (IndexError, TypeError, ValueError):
            continue
        sem_label = next((s for s in sems if s not in _META_SKIP), "object")
        snapped = nm.snap(np.array([tx, ty, cam_z]))
        if snapped is None:
            continue
        snapped[2] = cam_z
        if _accept(snapped, min_distance * 0.6):
            pts.append(snapped)
            meta.append({"type": f"object:{sem_label}", "key": key,
                         "obj": obj.obj_name or key,
                         "generator": info.get("generator", ""), "pos": [tx, ty, tz]})

    wp3d = np.array(pts) if pts else np.empty((0, 3))
    carb.log_info(f"Placed {len(wp3d)} waypoints (NavMesh-snapped)")
    return wp3d, meta, room_data


# ══════════════════════════════════════════════════════════════════════════════

def extract_room_data_for_regions(stage: Usd.Stage, state: dict) -> list[dict]:
    """Extract room_data list for build_region_catalog (same format as grid pipeline)."""
    objs = state.get("objs", {})
    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    room_data: list[dict] = []
    for key, info in objs.items():
        obj = ObjectState.from_dict(info)
        if not obj.active or obj.category != "room":
            continue
        prims = _room_prims(stage, obj.obj_name or key)
        if not prims:
            continue
        bmin, bmax = _merged_bbox(bbox_cache, prims)
        area = float((bmax[0] - bmin[0]) * (bmax[1] - bmin[1]))
        if area < 0.5:
            continue
        room_data.append({"key": key, "type": _rtype(obj),
                          "min_xy": bmin[:2].copy(), "max_xy": bmax[:2].copy(),
                          "floor_z": float(bmin[2]), "area": area})
    return room_data


def _room_data_full(stage: Usd.Stage, state: dict) -> list[dict]:
    """Room data with full 3-D bboxes (for room detection along paths)."""
    objs = state.get("objs", {})
    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    room_data: list[dict] = []
    for key, info in objs.items():
        obj = ObjectState.from_dict(info)
        if not obj.active or obj.category != "room":
            continue
        prims = _room_prims(stage, obj.obj_name or key)
        if not prims:
            continue
        bmin, bmax = _merged_bbox(bbox_cache, prims)
        area = float((bmax[0] - bmin[0]) * (bmax[1] - bmin[1]))
        if area < 0.5:
            continue
        room_data.append({"key": key, "type": _rtype(obj),
                          "min": bmin.copy(), "max": bmax.copy(), "area": area})
    return room_data
