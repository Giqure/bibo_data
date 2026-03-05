"""Build NavGrid from USD scene + place semantic waypoints on it."""
from __future__ import annotations

import math
from typing import Optional

import carb
import numpy as np
from pxr import Usd, UsdGeom

from syn_utils.models import ROOM_TYPES, ObjectState
from syn_utils.nav_grid import NavGrid
from common.vln_types import ROOM_WEIGHT, _SKIP_SEM, _META_SKIP
from common.scene_utils import _base, _room_prims, _bbox, _merged_bbox, _rtype


# ══════════════════════════════════════════════════════════════════════════════
# Build NavGrid
# ══════════════════════════════════════════════════════════════════════════════

def build_nav_grid(
    stage: Usd.Stage, state: dict, resolution: float = 0.15,
    wall_margin: float = 0.15, obstacle_inflation: float = 0.25,
) -> tuple[NavGrid, float, list[dict]]:
    """Build NavGrid from scene geometry.  Returns (grid, floor_z, room_data)."""
    objs = state.get("objs", {})
    bc = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    grid_min, grid_max = np.full(2, np.inf), np.full(2, -np.inf)
    room_data: list[dict] = []
    floor_zs: list[float] = []
    for key, info in objs.items():
        obj = ObjectState.from_dict(info)
        if not obj.active or obj.category != "room":
            continue
        prims = _room_prims(stage, obj.obj_name or key)
        if not prims:
            continue
        bmin, bmax = _merged_bbox(bc, prims)
        area = float((bmax[0] - bmin[0]) * (bmax[1] - bmin[1]))
        if area < 0.5:
            continue
        fz = float(bmin[2])
        floor_zs.append(fz)
        grid_min = np.minimum(grid_min, bmin[:2])
        grid_max = np.maximum(grid_max, bmax[:2])
        room_data.append({"key": key, "type": _rtype(obj),
                          "min_xy": bmin[:2].copy(), "max_xy": bmax[:2].copy(),
                          "floor_z": fz, "area": area})
    if not room_data:
        carb.log_warn("No valid rooms"); return NavGrid(np.zeros(2), np.ones(2), resolution), 0.0, []
    gfz = float(min(floor_zs))
    pad = 1.0
    grid = NavGrid(grid_min - pad, grid_max + pad, resolution)
    for rd in room_data:
        grid.mark_floor(rd["min_xy"], rd["max_xy"], wall_margin)
    for key, info in objs.items():
        obj = ObjectState.from_dict(info)
        if not obj.active:
            continue
        sems = set(obj.tags.get("Semantics", []))
        if sems & {"room", "door", "window", "entrance", "open",
                   "ceiling-light", "lighting", "wall-decoration"}:
            continue
        if sems & ROOM_TYPES:
            continue
        dof = info.get("dof_matrix_translation")
        if not dof or len(dof) < 3:
            continue
        try:
            tx, ty = float(dof[0][3]), float(dof[1][3])
        except (IndexError, TypeError, ValueError):
            continue
        pn = (obj.obj_name or key).replace(".", "_").replace("(", "_").replace(")", "_")
        prim = stage.GetPrimAtPath(f"/World/{pn}")
        if prim.IsValid():
            try:
                omin, omax = _bbox(bc, prim)
                if (omax[0] - omin[0]) * (omax[1] - omin[1]) > 0.04:
                    grid.mark_obstacle(omin[:2], omax[:2], obstacle_inflation); continue
            except Exception:
                pass
        grid.mark_obstacle(np.array([tx - 0.3, ty - 0.3]),
                           np.array([tx + 0.3, ty + 0.3]), obstacle_inflation)
    carb.log_info(f"NavGrid: {grid.nx}×{grid.ny}, {grid.free_area_m2:.1f}m², {grid.num_components} comp")
    return grid, gfz, room_data


# ══════════════════════════════════════════════════════════════════════════════
# Place waypoints
# ══════════════════════════════════════════════════════════════════════════════

def place_waypoints(
    grid: NavGrid, stage: Usd.Stage, state: dict, room_data: list[dict],
    global_floor_z: float, camera_height: float = 1.5,
    room_wp_base: int = 5, min_distance: float = 1.5,
) -> tuple[np.ndarray, list[dict]]:
    """Place semantic waypoints — every point guaranteed navigable."""
    objs = state.get("objs", {})
    bc = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    pts: list[np.ndarray] = []
    meta: list[dict] = []

    def _accept(xy, md):
        return not pts or float(np.min(np.linalg.norm(np.asarray(pts) - xy, axis=1))) >= md

    # Rooms
    for rd in room_data:
        w = ROOM_WEIGHT.get(rd["type"] or "", 1.0)
        n = max(1, int(room_wp_base * w * max(1.0, math.sqrt(rd["area"]) / 2)))
        for xy in grid.sample_in_rect(n, min_distance, rd["min_xy"], rd["max_xy"]):
            if _accept(xy, min_distance):
                pts.append(xy)
                meta.append({"type": f"room:{rd['type'] or '?'}", "room": rd["key"],
                             "room_type": rd["type"] or "unknown"})

    # Doors
    for key, info in objs.items():
        obj = ObjectState.from_dict(info)
        if not obj.active or obj.category != "door":
            continue
        sems = obj.tags.get("Semantics", [])
        label = "entrance" if "entrance" in sems else "door"
        if len(obj.connected_rooms) < 2:
            continue
        centres: list[np.ndarray] = []
        for rn in obj.connected_rooms[:2]:
            ri = objs.get(rn, {})
            if not ri:
                continue
            robj = ObjectState.from_dict(ri)
            rprims = _room_prims(stage, robj.obj_name or rn)
            if rprims:
                rmin, rmax = _merged_bbox(bc, rprims)
                centres.append((rmin + rmax) / 2)
        if len(centres) == 2:
            mid = (centres[0][:2] + centres[1][:2]) / 2
            snapped = grid.nearest_free(mid)
            if snapped is not None and _accept(snapped, min_distance * 0.8):
                pts.append(snapped)
                meta.append({"type": f"transition:{label}", "room": key,
                             "connects": obj.connected_rooms[:2]})

    # Objects
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
            tx, ty = float(dof[0][3]), float(dof[1][3])
            tz = float(dof[2][3])
        except (IndexError, TypeError, ValueError):
            continue
        sem_label = next((s for s in sems if s not in _META_SKIP), "object")
        snapped = grid.nearest_free(np.array([tx, ty]))
        if snapped is not None and _accept(snapped, min_distance * 0.6):
            pts.append(snapped)
            meta.append({"type": f"object:{sem_label}", "key": key,
                         "obj": obj.obj_name or key,
                         "generator": info.get("generator", ""), "pos": [tx, ty, tz]})

    if pts:
        arr = np.array(pts)
        wp3d = np.hstack([arr, np.full((len(arr), 1), global_floor_z + camera_height)])
    else:
        wp3d = np.empty((0, 3))
    carb.log_info(f"Placed {len(wp3d)} waypoints")
    return wp3d, meta
