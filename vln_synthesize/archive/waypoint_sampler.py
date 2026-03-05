"""Semantic waypoint sampler – rooms · doors · objects.

Naming: "living-room_0/0.meshed" → USDC "living_room_0_0_{floor,wall,...}"
Doors:  Factory-named prims → use connected-room midpoints.
Objects: iterate /World children, skip room geometry, sample near bbox centre.

API: resolveStatePath, loadStateJson, sampleSemanticWaypoints
"""
from __future__ import annotations

import json, math, os, re
from typing import Optional

import carb
import numpy as np
from pxr import Usd, UsdGeom

from syn_utils.waypoint import samplePointWithPoissonDisk

# ── Constants ──────────────────────────────────────────────────────────────────
ROOM_TYPES = {
    "kitchen", "bedroom", "living-room", "closet", "hallway", "bathroom",
    "garage", "balcony", "dining-room", "utility", "staircase-room",
    "warehouse", "office", "meeting-room", "open-office", "break-room",
    "restroom", "factory-office",
}
ROOM_WEIGHT: dict[str, float] = {
    "living-room": 2.0, "kitchen": 1.5, "dining-room": 1.5, "bedroom": 1.2,
    "office": 1.2, "bathroom": 0.6, "closet": 0.4, "balcony": 0.6, "hallway": 1.0,
}
ROOM_SUFFIXES: tuple[str, ...] = ("_floor", "_wall", "_ceiling", "_exterior")
TAG_PATTERN: re.Pattern = re.compile(r"^(\w+)\((.+)\)$")


# ── Helpers ────────────────────────────────────────────────────────────────────
def _tags(tag_list: list[str]) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    for tag_str in tag_list:
        match = TAG_PATTERN.match(tag_str.strip())
        if match:
            result.setdefault(match.group(1), []).append(match.group(2))
    return result


def _rtype(parsed: dict[str, list[str]]) -> Optional[str]:
    return next((val for val in parsed.get("Semantics", []) if val in ROOM_TYPES), None)


def _base(obj_name: str) -> str:
    """'living-room_0/0.meshed' → 'living_room_0_0'"""
    base = obj_name.replace("-", "_").replace("/", "_")
    return (base[:-7] if base.endswith(".meshed") else base).replace(".", "_")


def _roomPrims(stage: Usd.Stage, obj_name: str) -> list[Usd.Prim]:
    base = _base(obj_name)
    return [prim for suffix in (*ROOM_SUFFIXES, "")
            if (prim := stage.GetPrimAtPath(f"/World/{base}{suffix}")).IsValid()]


def _bbox(bbox_cache: UsdGeom.BBoxCache, prim: Usd.Prim) -> tuple[np.ndarray, np.ndarray]:
    aligned = bbox_cache.ComputeWorldBound(prim).ComputeAlignedRange()
    return np.array(aligned.GetMin()), np.array(aligned.GetMax())


def _merged(bbox_cache: UsdGeom.BBoxCache, prims: list[Usd.Prim]) -> tuple[np.ndarray, np.ndarray]:
    bound_min, bound_max = np.full(3, np.inf), np.full(3, -np.inf)
    for prim in prims:
        prim_min, prim_max = _bbox(bbox_cache, prim)
        bound_min = np.minimum(bound_min, prim_min)
        bound_max = np.maximum(bound_max, prim_max)
    return bound_min, bound_max


def _far(existing: list[np.ndarray], candidate: np.ndarray, min_dist: float) -> bool:
    return not existing or np.min(np.linalg.norm(np.asarray(existing) - candidate, axis=1)) >= min_dist


def _obj_bbox_xy(bbox_cache: UsdGeom.BBoxCache, stage: Usd.Stage,
                 obj_name: str) -> tuple[float, np.ndarray, np.ndarray] | None:
    """Return (half_extent, bbox_min_xy, bbox_max_xy) for an object prim, or None."""
    prim_name = obj_name.replace(".", "_").replace("(", "_").replace(")", "_")
    prim = stage.GetPrimAtPath(f"/World/{prim_name}")
    if not prim.IsValid():
        return None
    try:
        omin, omax = _bbox(bbox_cache, prim)
        half = float(max((omax[0] - omin[0]) / 2, (omax[1] - omin[1]) / 2))
        return half, omin[:2], omax[:2]
    except Exception:
        return None


def _point_in_room(xy: np.ndarray, room_floors: list[dict]) -> bool:
    """Check if *xy* falls inside any room's XY floor footprint."""
    for rf in room_floors:
        if (rf["min"][0] <= xy[0] <= rf["max"][0] and
                rf["min"][1] <= xy[1] <= rf["max"][1]):
            return True
    return False


def _point_clear_of_obstacles(xy: np.ndarray, obstacle_bboxes: list[tuple[np.ndarray, np.ndarray]],
                              clearance: float = 0.3) -> bool:
    """Ensure *xy* does not fall inside any obstacle's XY bbox (with margin)."""
    for ob_min, ob_max in obstacle_bboxes:
        if (ob_min[0] - clearance <= xy[0] <= ob_max[0] + clearance and
                ob_min[1] - clearance <= xy[1] <= ob_max[1] + clearance):
            return False
    return True


def _find_nearby_floor(
    obj_centre_xy: np.ndarray,
    bbox_cache: UsdGeom.BBoxCache,
    stage: Usd.Stage,
    obj_name: str,
    room_floors: list[dict],
    obstacle_bboxes: list[tuple[np.ndarray, np.ndarray]],
    standoff: float = 0.6,
    num_angles: int = 16,
    max_rings: int = 4,
) -> np.ndarray | None:
    """Find a nearby open-floor position around the object.

    Strategy:
      1. Compute the object's XY half-extent from its USD bbox.
      2. Sample *num_angles* directions × *max_rings* distance rings outward.
      3. For each candidate check:
         a) inside a room floor footprint,
         b) not overlapping any obstacle bbox.
      4. Return the closest valid candidate, or None.
    """
    info = _obj_bbox_xy(bbox_cache, stage, obj_name)
    half_extent = info[0] if info else 0.4
    obj_bbox = (info[1], info[2]) if info else None

    best: np.ndarray | None = None
    best_dist = float("inf")
    angles = np.linspace(0, 2 * math.pi, num_angles, endpoint=False)
    # add small jitter so we don't always land on axis-aligned spots
    angles += np.random.uniform(0, 2 * math.pi / num_angles)

    for ring in range(1, max_rings + 1):
        dist = half_extent + standoff * ring
        for angle in angles:
            candidate = obj_centre_xy + dist * np.array([math.cos(angle), math.sin(angle)])
            if not _point_in_room(candidate, room_floors):
                continue
            if not _point_clear_of_obstacles(candidate, obstacle_bboxes):
                continue
            d = float(np.linalg.norm(candidate - obj_centre_xy))
            if d < best_dist:
                best_dist = d
                best = candidate
        if best is not None:
            return best  # return closest ring that has a valid candidate

    return best  # None if nothing found

# ── Public API ─────────────────────────────────────────────────────────────────
def resolveStatePath(usdc_path: str, cli_arg: Optional[str] = None) -> Optional[str]:
    """Find solve_state.json: CLI arg → usdc_dir/.. → usdc_dir"""
    if cli_arg and os.path.isfile(cli_arg):
        return os.path.abspath(cli_arg)
    usdc_dir = os.path.dirname(os.path.abspath(usdc_path))
    for rel in ["..", ".", "../.."]:  
        candidate = os.path.normpath(os.path.join(usdc_dir, rel, "solve_state.json"))
        if os.path.isfile(candidate):
            return candidate
    return None
def loadStateJson(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


# ── Main sampler ───────────────────────────────────────────────────────────────
def sampleSemanticWaypoints(
    stage: Usd.Stage,
    state: dict,
    camera_height: float = 1.5,
    room_wp_count: int = 5,
    min_distance: float = 1.5,
) -> tuple[np.ndarray, list[dict]]:
    """Sample waypoints by room semantics + door transitions + object prims.

    Returns (waypoints[N,3], meta[N]).
    """
    all_objs: dict = state.get("objs", {})
    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    waypoints: list[np.ndarray] = []
    meta_list: list[dict] = []
    room_bases: set[str] = set()
    floor_zs: list[float] = []
    room_floors: list[dict] = []          # collected in Phase 1 for Phase 3

    # ── Phase 1: rooms ─────────────────────────────────────────────────────
    for key, info in all_objs.items():
        if not info.get("active"):
            continue
        parsed = _tags(info.get("tags", []))
        if "room" not in parsed.get("Semantics", []):
            continue
        room_type = _rtype(parsed)
        obj_name = info.get("obj", key)
        prims = _roomPrims(stage, obj_name)
        if not prims:
            continue
        room_bases.add(_base(obj_name))
        bound_min, bound_max = _merged(bbox_cache, prims)
        area = float((bound_max[0] - bound_min[0]) * (bound_max[1] - bound_min[1]))
        if area < 0.5:
            continue
        floor_z = float(bound_min[2])
        floor_zs.append(floor_z)
        room_floors.append({"min": bound_min[:2].copy(),
                            "max": bound_max[:2].copy(),
                            "floor_z": floor_z,
                            "key": key})
        weight = ROOM_WEIGHT.get(room_type or "", 1.0)
        num_target = max(1, int(room_wp_count * weight * max(1.0, math.sqrt(area) / 2)))
        candidates = samplePointWithPoissonDisk(bound_min[:2], bound_max[:2], min_distance, num_target * 5)
        added = 0
        for point_xy in candidates:
            if added >= num_target:
                break
            wp = np.array([point_xy[0], point_xy[1], floor_z + camera_height])
            if _far(waypoints, wp, min_distance):
                waypoints.append(wp)
                meta_list.append({"type": f"room:{room_type or '?'}", "room": key,
                                  "room_type": room_type or "unknown"})
                added += 1
        carb.log_info(f"Room '{key}'({room_type}): {area:.0f}m² → {added}wp")

    # ── Phase 2: doors → connected-room midpoints ──────────────────────────
    for key, info in all_objs.items():
        if not info.get("active"):
            continue
        semantics = _tags(info.get("tags", [])).get("Semantics", [])
        if "door" not in semantics and "entrance" not in semantics:
            continue
        label = "entrance" if "entrance" in semantics else "door"
        connections = [rel["target_name"] for rel in info.get("relations", [])
                       if rel.get("relation", {}).get("relation_type") == "CutFrom"]
        if len(connections) < 2:
            continue
        centres: list[np.ndarray] = []
        for room_name in connections[:2]:
            room_prims = _roomPrims(stage, all_objs.get(room_name, {}).get("obj", room_name))
            if room_prims:
                room_min, room_max = _merged(bbox_cache, room_prims)
                centres.append((room_min + room_max) / 2)
        if len(centres) == 2:
            midpoint = (centres[0] + centres[1]) / 2
            wp = np.array([midpoint[0], midpoint[1], min(centres[0][2], centres[1][2]) + camera_height])
            if _far(waypoints, wp, min_distance * 0.8):
                waypoints.append(wp)
                meta_list.append({"type": f"transition:{label}", "room": key,
                                  "connects": connections})

    # ── Phase 3: objects from state JSON (non-room, non-structural) ───────
    global_floor_z = min(floor_zs) if floor_zs else 0.0
    # structural / room / non-navigable semantics → skip entirely
    _SKIP_SEM = (
        ROOM_TYPES |                                         # all room types
        {"room", "cutter",                                   # mesh-level
         "door", "window", "entrance", "open",               # openings
         "root", "room-node", "room-contour",                # graph nodes
         "ground", "second-floor", "third-floor",            # floor labels
         "exterior", "staircase", "visited", "new",          # structural
         "ceiling-light", "lighting",                        # lights (not navigable)
         "wall-decoration",                                  # wall-mounted
         }
    )
    # meta / solver flags – not meaningful for labelling, strip from sem_label
    _META_SKIP = {
        "object", "real-placeholder", "oversize-placeholder",
        "asset-as-placeholder", "asset-placeholder-for-children",
        "placeholder-bbox", "single-generator",
        "no-rotation", "no-collision", "no-children",
        "access-top", "access-front", "access-any-side", "access-all-sides",
        "access-stand-near", "access-open-door", "access-with-hand",
    }
    # Pre-collect obstacle bboxes from state JSON for clearance checking
    obstacle_bboxes: list[tuple[np.ndarray, np.ndarray]] = []
    for okey, oinfo in all_objs.items():
        if not oinfo.get("active"):
            continue
        oparsed = _tags(oinfo.get("tags", []))
        osem = set(oparsed.get("Semantics", []))
        # skip rooms / doors / windows – only keep physical objects
        if osem & {"room", "door", "window", "entrance", "open",
                   "ceiling-light", "lighting", "wall-decoration"}:
            continue
        if osem & ROOM_TYPES:
            continue
        odof = oinfo.get("dof_matrix_translation")
        if not odof or len(odof) < 3:
            continue
        try:
            ox, oy = float(odof[0][3]), float(odof[1][3])
        except (IndexError, TypeError, ValueError):
            continue
        # try to get real bbox from USD prim
        oname = oinfo.get("obj", okey)
        obox = _obj_bbox_xy(bbox_cache, stage, oname)
        if obox is not None:
            _, ob_min_xy, ob_max_xy = obox
            if float((ob_max_xy[0]-ob_min_xy[0]) * (ob_max_xy[1]-ob_min_xy[1])) > 0.04:
                obstacle_bboxes.append((ob_min_xy.copy(), ob_max_xy.copy()))
        else:
            # fallback: small bbox around dof position
            r = 0.3
            obstacle_bboxes.append((np.array([ox - r, oy - r]),
                                    np.array([ox + r, oy + r])))

    for key, info in all_objs.items():
        if not info.get("active"):
            continue
        parsed = _tags(info.get("tags", []))
        semantics = set(parsed.get("Semantics", []))
        if semantics & _SKIP_SEM:
            continue
        # position from dof_matrix_translation (3×4 matrix, last col = translation)
        dof = info.get("dof_matrix_translation")
        if not dof or len(dof) < 3:
            continue
        try:
            tx, ty, tz = float(dof[0][3]), float(dof[1][3]), float(dof[2][3])
        except (IndexError, TypeError, ValueError):
            continue
        sem_label = next((s for s in semantics if s not in _META_SKIP), "object")
        obj_name = info.get("obj", key)
        obj_xy = np.array([tx, ty])
        # find nearby open floor space
        floor_xy = _find_nearby_floor(
            obj_xy, bbox_cache, stage, obj_name,
            room_floors, obstacle_bboxes,
        )
        if floor_xy is None:
            carb.log_warn(f"Object '{key}': no open floor found, skipping")
            continue
        wp = np.array([floor_xy[0], floor_xy[1], global_floor_z + camera_height])
        if _far(waypoints, wp, min_distance * 0.6):
            waypoints.append(wp)
            meta_list.append({"type": f"object:{sem_label}", "key": key,
                              "obj": obj_name, "generator": info.get("generator", ""),
                              "pos": [tx, ty, tz]})

    carb.log_info(f"Total waypoints: {len(waypoints)}")
    return (np.array(waypoints), meta_list) if waypoints else (np.empty((0, 3)), [])
