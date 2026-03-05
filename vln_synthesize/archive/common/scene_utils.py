"""USD scene helpers and region catalog builder.

Shared between grid-based and NavMesh-based pipelines.
"""
from __future__ import annotations

import json
import os
from typing import Optional

import carb
import numpy as np
from pxr import Usd, UsdGeom

from syn_utils.models import ROOM_TYPES, ObjectState
from syn_utils.goal_region import CircleRegion, RectRegion, GoalRegion
from syn_utils.goal_region_ext import CompositeRegion
from common.vln_types import ROOM_SUFFIXES, _SKIP_SEM, _META_SKIP


# ── USD helpers ───────────────────────────────────────────────────────────────

def _base(obj_name: str) -> str:
    base = obj_name.replace("-", "_").replace("/", "_")
    return (base[:-7] if base.endswith(".meshed") else base).replace(".", "_")


def _room_prims(stage: Usd.Stage, obj_name: str) -> list[Usd.Prim]:
    base = _base(obj_name)
    return [
        prim
        for suffix in (*ROOM_SUFFIXES, "")
        if (prim := stage.GetPrimAtPath(f"/World/{base}{suffix}")).IsValid()
    ]


def _bbox(bbox_cache: UsdGeom.BBoxCache, prim: Usd.Prim):
    aligned = bbox_cache.ComputeWorldBound(prim).ComputeAlignedRange()
    return np.array(aligned.GetMin()), np.array(aligned.GetMax())


def _merged_bbox(bbox_cache: UsdGeom.BBoxCache, prims: list[Usd.Prim]):
    bmin, bmax = np.full(3, np.inf), np.full(3, -np.inf)
    for p in prims:
        lo, hi = _bbox(bbox_cache, p)
        bmin = np.minimum(bmin, lo)
        bmax = np.maximum(bmax, hi)
    return bmin, bmax


def _rtype(obj: ObjectState) -> Optional[str]:
    return next((v for v in obj.tags.get("Semantics", []) if v in ROOM_TYPES), None)


# ── State file helpers ────────────────────────────────────────────────────────

def resolveStatePath(usdc_path: str, cli_arg: str | None = None) -> str | None:
    if cli_arg and os.path.isfile(cli_arg):
        return os.path.abspath(cli_arg)
    usdc_dir = os.path.dirname(os.path.abspath(usdc_path))
    for rel in ["..", ".", "../.."]:
        cand = os.path.normpath(os.path.join(usdc_dir, rel, "solve_state.json"))
        if os.path.isfile(cand):
            return cand
    return None


def loadStateJson(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


# ── Region catalog ────────────────────────────────────────────────────────────

def build_region_catalog(
    state: dict,
    room_data: list[dict],
    stage: Usd.Stage,
    wall_margin: float = 0.3,
    object_radius: float = 1.5,
) -> dict[str, GoalRegion]:
    """Build label→GoalRegion from scene data (rooms + objects)."""
    from syn_utils.goal_region_ext import room_region, object_region

    objs = state.get("objs", {})
    regions: dict[str, GoalRegion] = {}

    # Rooms
    room_by_type: dict[str, list[GoalRegion]] = {}
    for rd in room_data:
        rtype = rd.get("type") or "unknown"
        r = room_region(rd["min_xy"], rd["max_xy"], wall_margin)
        regions[f"room:{rtype}/{rd['key']}"] = r
        room_by_type.setdefault(rtype, []).append(r)
    for rtype, rlist in room_by_type.items():
        regions[f"rooms:{rtype}"] = CompositeRegion(rlist) if len(rlist) > 1 else rlist[0]

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
        except (IndexError, TypeError, ValueError):
            continue
        sem_label = next((s for s in sems if s not in _META_SKIP), None)
        if sem_label is None:
            continue
        regions[f"object:{sem_label}/{key}"] = object_region(np.array([tx, ty]), object_radius)

    carb.log_info(f"Region catalog: {len(regions)} regions")
    return regions
