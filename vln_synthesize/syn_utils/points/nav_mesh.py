from typing import cast

import omni.usd
import isaacsim.core.utils.prims as prims_utils   # type: ignore
import isaacsim.core.utils.bounds as bounds_utils # type: ignore
import omni.anim.navigation.core as nav           # type: ignore
from omni.isaac.core import World                 # type: ignore

import carb
import numpy as np
import argparse

from pxr import Gf, Sdf, Usd, UsdGeom
from syn_utils.models import SMALL_OBJECT_FUNCTIONS, CutterState, ReadonlyStates, RoomState, States, ObjectState, NavigationPointCollection

import omni.kit.commands                          # type: ignore
import NavSchema                                  # type: ignore

from .poisson import sampleWithPoissonDisk
import time

NAVMESH_VOLUME_NAME = "NavMeshVolume"
NAVMESH_VOLUME_INCLUDE = 0
NAVMESH_VOLUME_EXCLUDE = 1

HALF_EXTENT = 0.5
INCLUDE_SCALE = 10.0
EXCLUDE_SCALE = 2.0

SMALL_OBJECT_THRESHOLD = 0.125  # 50cm cube

def CreateNavMeshVolume(
        position: tuple[float, float, float] | None = None,
        size: tuple[float, float, float] | None = None,
        volume_type: int = NAVMESH_VOLUME_INCLUDE,
        usd_context_name: str = "",
        layer: Sdf.Layer | None = None,
    ) -> Sdf.Path:
        # 对 CreateNavMeshVolumeCommand 的改造，接受 size 并返回 prim_path. 
        # CreateNavMeshVolumeCommand is from omni.anim.navigation.core-107.3.8+107.3.3.lx64.r.cp311.u353
        world = World.instance()
        if world:
            world.stop()
        _usd_context = omni.usd.get_context(usd_context_name)
        _selection = _usd_context.get_selection()
        stage = _usd_context.get_stage()
        meters_per_unit = UsdGeom.GetStageMetersPerUnit(stage)
        if layer is None:
            layer = stage.GetEditTarget().GetLayer()
        
        _prim_path = Sdf.Path(omni.usd.get_stage_next_free_path(stage, f"/{NAVMESH_VOLUME_NAME}", True))

        volume = NavSchema.NavMeshVolume.Define(stage, _prim_path)
        volumeTypeAttr = volume.GetNavVolumeTypeAttr()
        if volumeTypeAttr:
            volumeTypeAttr.Set("Include" if volume_type == NAVMESH_VOLUME_INCLUDE else "Exclude")

        omni.kit.commands.execute(
            "ApplyNavMeshAPICommand", prim_path=_prim_path, api=NavSchema.NavMeshAreaAPI
        )

        # Geometry setup
        if size is not None:
            size = Gf.Vec3d(*size)
        if position is not None:
            position = Gf.Vec3d(*position)

        # set the default boundable extent
        prim = stage.GetPrimAtPath(_prim_path)
        boundable = UsdGeom.Boundable(prim)
        extentAttr = boundable.GetExtentAttr()
        if extentAttr:
            # if size is not None:
            #     halfExtent = size / 2.0
            #     extentAttr.Set([-halfExtent, halfExtent])
            # else:
            #     # the unit of the defined extent is meter, should convert to the target unit
            #     halfExtent = HALF_EXTENT / meters_per_unit
            #     extentAttr.Set([(-halfExtent, -halfExtent, -halfExtent), (halfExtent, halfExtent, halfExtent)])
            halfExtent = HALF_EXTENT / meters_per_unit
            extentAttr.Set([(-halfExtent, -halfExtent, -halfExtent), (halfExtent, halfExtent, halfExtent)])

        _selection.set_prim_path_selected(_prim_path.pathString, True, True, True, True)

        scale = Gf.Matrix4d(1.0)
        if size is not None:
            scale.SetScale(size)
        else:
            scale.SetScale(INCLUDE_SCALE if volume_type == NAVMESH_VOLUME_INCLUDE else EXCLUDE_SCALE)
        xform = scale
        if position is not None:
            xform = xform * Gf.Matrix4d(1.0).SetTranslate(position)

        # transform the navmesh volume
        omni.kit.commands.execute("TransformPrim", path=_prim_path, new_transform_matrix=xform)
        if world:
            world.play()
        return _prim_path


def sampleWithNavMesh(states: States, args: argparse.Namespace):
    # get stage information
    bbox_cache = bounds_utils.create_bbox_cache()
    world_aabb = bounds_utils.compute_aabb(bbox_cache, prim_path="/World")
    world_min = world_aabb[0:3]
    world_max = world_aabb[3:6]

    # setup NavMesh volume
    pad = 1.0, 1.0, 0.02
    extent = (world_max - world_min) + np.array(pad)
    center = (world_min + world_max) / 2 - np.array([0, 0, pad[2]])
    navmeshvolume_prim_path = CreateNavMeshVolume(
        position=center, 
        size=extent, 
        volume_type=NAVMESH_VOLUME_INCLUDE,
    )
    carb.log_info(f"NavMesh volume created at {navmeshvolume_prim_path}.")

    settings = carb.settings.get_settings()
    from omni.anim.navigation.core import NavMeshSettings   # type: ignore
    settings.set(NavMeshSettings.AGENT_MIN_HEIGHT_SETTING_PATH,args.agent_height)
    settings.set(NavMeshSettings.AGENT_MIN_RADIUS_SETTING_PATH,args.agent_radius)
    settings.set(NavMeshSettings.AGENT_MAX_STEP_HEIGHT_SETTING_PATH,0)
    settings.set(NavMeshSettings.AGENT_MAX_FLOOR_SLOPE_SETTING_PATH,args.max_slope)
    carb.log_info(f"NavMesh persistent configured: r={args.agent_radius} h={args.agent_height} step={args.max_step_height} slope={args.max_slope}")

    i_nav = nav.acquire_interface()
    session = 'random_points'
    i_nav.set_random_seed(session, 12345)

    # bake NavMesh
    x = i_nav.start_navmesh_baking_and_wait()
    navmesh = i_nav.get_navmesh()
    if navmesh is None:
        carb.log_error("NavMesh bake failed")
    else:
        carb.log_info(f"NavMesh baked: {i_nav.get_area_count()} area(s)")


    cutters = states.get("cutter", {})
    cutters = cast(dict[str, CutterState], cutters)
    rooms = states.get("room", {})
    rooms = cast(dict[str, RoomState], rooms)
    objects = states.get("object", {})
    objects = cast(dict[str, ObjectState], objects)

    # compute Prims
    stage = omni.usd.get_context().get_stage()
    for state_dict in states.values():
        for state in state_dict.values():
            state.computePrims(stage)

    # compute object-room relations
    for obj in objects.values():
        obj.computeRoom(states)

    # set navigation goal
    # Cutters
    for cutter_id, cutter_state in cutters.items():
        cutter_type = cutter_state.cutter_type
        # 除 门、窗 外不处理
        if cutter_type not in {"door", "window", "entrance"}:
            continue
        # Find rooms cut by door
        if cutter_type in {"door", "entrance"}:
            for rel in cutter_state.relations:
                if not rel.relation.relation_type == "CutFrom":
                    continue
                room_id = rel.target_name
                room_state = rooms.get(room_id)
                if not room_state:
                    continue

                room_state.navigation_access.add(cutter_id)
                room_state.navigation_points.extend(computeRoomNavigationPointByCutter(cutter_state, room_state))        

        # Navigation points for cutter itself
        cutter_state.navigation_points = computeCutterNavigationPoint(cutter_state)
    # Objects
    for object_id, object_state in objects.items():
        object_state.navigation_points = computeObjectNavigationPoint(
            object_state, {"rooms": rooms, "objects": objects}
        )
    # Rooms
    for room_id, room_state in rooms.items():
        # room_state.navigation_points = computeRoomNavigationPoint(
        #     room_state, 
        # )
        ...

    # random points
    random_points = [navmesh.query_random_point(session) for _ in range(100)]

    # compute path
    # start_t = time.perf_counter()
    path_states = []

    for obj in objects.values():
        for random_point in random_points:
            paths = [
                (p, pt)
                for pt in obj.navigation_points
                if (p := navmesh.query_shortest_path(pt, random_point)) is not None
            ]
            if not paths:
                continue
            nav_path, nav_point = min(paths, key=lambda x: x[0].length())
            if nav_path.length() < 0.5:
                continue
            path_states.append({
                "from": {"id": "random_point", "type": "random_point", "position": random_point},
                "to": {"id": obj.id, "type": "object", "position": nav_point},
                "length": nav_path.length(),
                "navmesh_path": nav_path,
            })

        for objB in objects.values():
            paths = [
                (p, pt, ptB)
                for pt in obj.navigation_points
                for ptB in objB.navigation_points
                if (p := navmesh.query_shortest_path(pt, ptB)) is not None
            ]
            if not paths:
                continue
            nav_path, nav_point, nav_pointB = min(paths, key=lambda x: x[0].length())
            if nav_path.length() < 0.5:
                continue
            path_states.append({
                "from": {"id": obj.id, "type": "object", "position": nav_point},
                "to": {"id": objB.id, "type": "object", "position": nav_pointB},
                "length": nav_path.length(),
                "navmesh_path": nav_path,
            })
    # elapsed_t = time.perf_counter() - start_t
    # print(f"compute path time: {elapsed_t:.4f}s")
    
    return path_states


def computeRoomNavigationPointByCutter(cutter: CutterState, room: RoomState) -> NavigationPointCollection:
    # TODO: When the postion of cutter computable
    return []

def computeCutterNavigationPoint(cutter: CutterState) -> NavigationPointCollection:
    # TODO: Door may be rotated
    return []

def computeObjectNavigationPoint(obj_state: ObjectState, states: ReadonlyStates) -> NavigationPointCollection:
    obj_prim = obj_state.prims.first_prim
    if not obj_prim or not obj_prim.IsValid():
        return []
    
    rooms = states.get("rooms", {})
    rooms = cast(dict[str, RoomState], rooms)
    objects = states.get("objects", {})
    objects = cast(dict[str, ObjectState], objects)

    # Small objects
    # 在原地生成一个导航点
    obj_bbox = np.array(obj_state.prims.bbox)
    if len(obj_bbox) < 6:
        carb.log_warn(f"{obj_bbox=} is invalid. No navigation point created.")
        return []
    min_xyz = obj_bbox[0:3]
    max_xyz = obj_bbox[3:6]
    obj_volume = np.prod(max_xyz - min_xyz)
    center = (min_xyz + max_xyz) / 2
    if set(obj_state.semantic_tags) & SMALL_OBJECT_FUNCTIONS or obj_volume < SMALL_OBJECT_THRESHOLD:
        return [center.tolist()]

    # Furnitures
    # 在周围生成一圈点
    # TODO: 区分出非矩形家具，例如圆形. 当前在bbox采柏松盘点
    if "furniture" in obj_state.semantic_tags:
        points_2d = sampleWithPoissonDisk(min_xyz[0:2], max_xyz[0:2], min_dist=0.2, num_target=8)
        points_3d = np.column_stack([points_2d, np.full(len(points_2d), 0.02)])
        return points_3d.tolist()

    # wall decorations
    # 在前面，距墙面0.2m 处采集一排点
    room_state = obj_state.room
    if not room_state:
        carb.log_warn(f"The room of {obj_state.id=} was not found.")
    if "wall-decoration" in obj_state.semantic_tags:
        if not obj_state.prims.obb:
            carb.log_warn(f"OBB of {obj_state.id=} is not available. Navigation point creates at object center.")
            return [center.tolist()]
        centroid, axes, half_extent = obj_state.prims.obb
        short_axe_index = half_extent.argmin()
        long_axe_index = half_extent.argmax()
        if axes[long_axe_index][2] > 0.9:
            long_axe_index = 3 - short_axe_index - long_axe_index
        short_axe = axes[short_axe_index]
        long_axe = axes[long_axe_index]

        if room_state and room_state.prims.obb and np.dot(centroid - room_state.prims.obb[0], short_axe) > 0:
            short_axe = -short_axe

        navigation_points = [centroid + short_axe * 0.1]
        navigation_points.extend(
            navigation_points[0] + np.outer(
                np.arange(-half_extent[long_axe_index], half_extent[long_axe_index], 0.1),  # 0.1m 间隔
                long_axe
            )
        )
        return navigation_points
    return []

# def computeRoomNavigationPoint(room_state: RoomState) -> NavigationPointCollection:


