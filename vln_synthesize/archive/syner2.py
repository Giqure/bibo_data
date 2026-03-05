"""
VLN data synthesis using pure Isaac Sim (no IsaacLab dependency).
Loads USDC scenes, samples waypoints, renders panoramas, exports connectivity.

Usage:
    ~/.local/share/ov/pkg/isaac-sim-4.5.0/python.sh synthesize_vln.py \
        --usdc_path /path/to/scene.usdc \
        --output_dir /path/to/output \
        --scan_id scene_001 \
        --num_waypoints 50 \
        --headless
"""

import argparse
import json
import math
import os
import sys
from typing import Optional

import numpy as np

# ── 1. Parse args BEFORE importing Omniverse (SimulationApp needs config at init) ──

parser = argparse.ArgumentParser(description="VLN data synthesis with pure Isaac Sim")
parser.add_argument("--usdc_path", type=str, required=True, help="Path to the USDC scene file")
parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
parser.add_argument("--scan_id", type=str, default="scene_001", help="Scene identifier")
parser.add_argument("--num_waypoints", type=int, default=50, help="Number of waypoints to sample")
parser.add_argument("--camera_height", type=float, default=1.5, help="Camera height above floor (meters)")
parser.add_argument("--min_wp_distance", type=float, default=1.5, help="Minimum distance between waypoints (meters)")
parser.add_argument("--max_connect_distance", type=float, default=5.0, help="Max distance for connectivity edges")
parser.add_argument("--image_width", type=int, default=640, help="Rendered image width")
parser.add_argument("--image_height", type=int, default=480, help="Rendered image height")
parser.add_argument("--headless", action="store_true", help="Run in headless mode")
parser.add_argument("--solve_state", type=str, default=None,
                    help="Path to solve_state.json (default: <usdc_dir>/../solve_state.json)")
parser.add_argument("--room_waypoints", type=int, default=5,
                    help="Base number of waypoints to sample per room")
parser.add_argument("--no_semantic", action="store_true",
                    help="Disable semantic waypoint generation, use Poisson-disk only")
parser.add_argument("--backend", type=str, default="grid", choices=["grid", "navmesh"],
                    help="Navigation backend: 'grid' (2D occupancy) or 'navmesh' (Recast/Detour)")
args = parser.parse_args()

# ── 2. Launch Isaac Sim ──

from isaacsim import SimulationApp

sim_config = {
    "headless": args.headless,
    "width": args.image_width,
    "height": args.image_height,
    "anti_aliasing": 3,  # DLAA
}
simulation_app = SimulationApp(sim_config)

# Now we can import Omniverse modules
import carb
import omni.replicator.core as rep
import omni.usd
from omni.isaac.core import World
import isaacsim.core.utils.stage as stage_utils
from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics


# ── 3. Helper functions ──

def getBound(stage: Usd.Stage):
    """Compute the axis-aligned bounding box of all meshes in the stage."""
    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    root = stage.GetPseudoRoot()
    bbox = bbox_cache.ComputeWorldBound(root)
    bbox_range = bbox.ComputeAlignedRange()
    min_pt = np.array(bbox_range.GetMin())
    max_pt = np.array(bbox_range.GetMax())
    return min_pt, max_pt


def add_collision_to_meshes(stage: Usd.Stage):
    """Add collision APIs to all mesh prims for raycasting."""
    count = 0
    for prim in stage.Traverse():
        if prim.IsA(UsdGeom.Mesh):
            if not prim.HasAPI(UsdPhysics.CollisionAPI):
                UsdPhysics.CollisionAPI.Apply(prim)
                UsdPhysics.MeshCollisionAPI.Apply(prim)
                count += 1
    carb.log_info(f"Added collision to {count} meshes")
    return count


def setPhysicsScene(stage: Usd.Stage):
    """Create a PhysicsScene if none exists."""
    for prim in stage.Traverse():
        if prim.IsA(UsdPhysics.Scene):
            carb.log_info(f"Physics scene already exists at {prim.GetPath()}")
            return
    scene_prim = stage.DefinePrim("/PhysicsScene", "PhysicsScene")
    UsdPhysics.Scene.Define(stage, "/PhysicsScene")
    physx_api = PhysxSchema.PhysxSceneAPI.Apply(scene_prim)
    physx_api.CreateEnableCCDAttr(True)
    physx_api.CreateEnableGPUDynamicsAttr(False)
    carb.log_info("Created /PhysicsScene")


def sampleFloorZ(stage, minPt, maxPt, numSamples=100):
    """Estimate floor height by finding the lowest z where geometry exists."""
    return float(minPt[2])


# Import VLN pipeline
from common import resolveStatePath, loadStateJson  # noqa: E402
from grid_pipeline import sample_vln_paths  # noqa: E402


def poissonDiskSample2D(minXY, maxXY, minDist, numTarget, maxAttempts=30):
    """Simple Poisson-disk sampling in 2D."""
    points = []
    rng = np.random.default_rng(42)
    for _ in range(numTarget * maxAttempts):
        if len(points) >= numTarget:
            break
        candidate = rng.uniform(minXY, maxXY)
        if len(points) == 0:
            points.append(candidate)
            continue
        dists = np.linalg.norm(np.array(points) - candidate, axis=1)
        if np.min(dists) >= minDist:
            points.append(candidate)
    return np.array(points)


def raycastDown(world, origin3d, maxDist=20.0):
    """Cast a ray downward from *origin3d*; returns hit dict."""
    from omni.physx import get_physx_scene_query_interface

    origin = carb.Float3(float(origin3d[0]), float(origin3d[1]), float(origin3d[2]))
    direction = carb.Float3(0.0, 0.0, -1.0)
    return get_physx_scene_query_interface().raycast_closest(origin, direction, maxDist)


def raycastBetween(world, p1, p2):
    """Check if there's a clear line of sight between two 3D points."""
    from omni.physx import get_physx_scene_query_interface

    direction = p2 - p1
    dist = float(np.linalg.norm(direction))
    if dist < 1e-6:
        return True
    direction = direction / dist

    hit_info = {"blocked": False}

    def report_hit(hit):
        if hit.distance < dist - 0.1:
            hit_info["blocked"] = True
        return True

    origin = carb.Float3(float(p1[0]), float(p1[1]), float(p1[2]))
    dir3 = carb.Float3(float(direction[0]), float(direction[1]), float(direction[2]))
    get_physx_scene_query_interface().raycast_closest(origin, dir3, dist + 0.5)
    return not hit_info["blocked"]


# ── 3b. Waypoint visualisation helpers ──

def createWaypointMarkers(stage, waypoints: np.ndarray, radius: float = 0.15):
    """Create coloured sphere markers at each waypoint position."""
    from pxr import UsdShade

    root_path = "/World/WaypointMarkers"
    stage.DefinePrim(root_path, "Xform")

    # Material – bright green
    mat_path = f"{root_path}/WaypointMat"
    mat = UsdShade.Material.Define(stage, mat_path)
    shader = UsdShade.Shader.Define(stage, f"{mat_path}/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.1, 0.9, 0.2))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.4)
    shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(0.85)
    mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")

    for i, wp in enumerate(waypoints):
        sphere_path = f"{root_path}/wp_{i:04d}"
        sphere = UsdGeom.Sphere.Define(stage, sphere_path)
        sphere.GetRadiusAttr().Set(radius)

        xf = UsdGeom.Xformable(sphere.GetPrim())
        xf.AddTranslateOp().Set(Gf.Vec3d(float(wp[0]), float(wp[1]), float(wp[2])))

        UsdShade.MaterialBindingAPI.Apply(sphere.GetPrim()).Bind(mat)

    carb.log_info(f"Created {len(waypoints)} waypoint markers under {root_path}")
    return root_path


def createConnectivityLines(stage, waypoints: np.ndarray, connectivity: dict,
                            tubeRadius: float = 0.03):
    """Draw thin cylinders between connected waypoints."""
    from pxr import UsdShade

    root_path = "/World/ConnectivityEdges"
    stage.DefinePrim(root_path, "Xform")

    # Material – orange
    mat_path = f"{root_path}/EdgeMat"
    mat = UsdShade.Material.Define(stage, mat_path)
    shader = UsdShade.Shader.Define(stage, f"{mat_path}/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(1.0, 0.5, 0.0))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.5)
    shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(0.7)
    mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")

    edge_count = 0
    drawn = set()
    for i, neighbors in connectivity.items():
        for nb in neighbors:
            j = nb["index"]
            key = (min(i, j), max(i, j))
            if key in drawn:
                continue
            drawn.add(key)

            p1 = waypoints[i]
            p2 = waypoints[j]
            mid = (p1 + p2) / 2.0
            diff = p2 - p1
            length = float(np.linalg.norm(diff))
            if length < 1e-6:
                continue

            cyl_path = f"{root_path}/edge_{i:04d}_{j:04d}"
            cyl = UsdGeom.Cylinder.Define(stage, cyl_path)
            cyl.GetRadiusAttr().Set(tubeRadius)
            cyl.GetHeightAttr().Set(length)
            cyl.GetAxisAttr().Set("Z")

            # Orient cylinder from p1 to p2
            direction = diff / length
            up = np.array([0.0, 0.0, 1.0])
            dot = np.dot(up, direction)

            xf = UsdGeom.Xformable(cyl.GetPrim())
            xf.AddTranslateOp().Set(Gf.Vec3d(float(mid[0]), float(mid[1]), float(mid[2])))

            if abs(dot) < 0.9999:
                axis = np.cross(up, direction)
                axis = axis / np.linalg.norm(axis)
                angle_deg = float(np.degrees(np.arccos(np.clip(dot, -1, 1))))
                rot_op = xf.AddRotateXYZOp()
                # Rodrigues via quaternion → Euler is complex; use orient op instead
                xf.ClearXformOpOrder()
                xf.AddTranslateOp().Set(Gf.Vec3d(float(mid[0]), float(mid[1]), float(mid[2])))
                quat = _axis_angle_to_quat(axis, math.radians(angle_deg))
                xf.AddOrientOp().Set(Gf.Quatf(quat))

            UsdShade.MaterialBindingAPI.Apply(cyl.GetPrim()).Bind(mat)
            edge_count += 1

    carb.log_info(f"Created {edge_count} connectivity edges under {root_path}")
    return root_path


def _axis_angle_to_quat(axis: np.ndarray, angle_rad: float) -> Gf.Quatd:
    """Convert axis-angle to Gf.Quatd."""
    half = angle_rad / 2.0
    s = math.sin(half)
    return Gf.Quatd(math.cos(half), Gf.Vec3d(float(axis[0]*s), float(axis[1]*s), float(axis[2]*s)))


def removeMarkers(stage):
    """Remove waypoint / edge marker prims so they don't pollute renders."""
    for path in ["/World/WaypointMarkers", "/World/ConnectivityEdges"]:
        prim = stage.GetPrimAtPath(path)
        if prim.IsValid():
            stage.RemovePrim(path)


# ── 4. Panorama rendering with Replicator ──

def setupCamera(stage, width: int, height: int):
    """Create a render product for panorama capture."""
    cam_prim = stage.DefinePrim("/World/VLNCamera", "Camera")
    camera = UsdGeom.Camera.Define(stage, "/World/VLNCamera")
    camera.GetClippingRangeAttr().Set(Gf.Vec2f(0.01, 1000.0))

    # Horizontal FOV ~ 60 degrees (for 12 heading angles covering 360)
    focal_length = width / (2.0 * math.tan(math.radians(30)))
    camera.GetFocalLengthAttr().Set(float(focal_length))

    rp = rep.create.render_product("/World/VLNCamera", (width, height))
    return camera, rp


def setCameraPose(cameraPrim, position, headingDeg, elevationDeg):
    """Set camera position and orientation."""
    xformable = UsdGeom.Xformable(cameraPrim.GetPrim())
    xformable.ClearXformOpOrder()

    translateOp = xformable.AddTranslateOp()
    translateOp.Set(Gf.Vec3d(*position.tolist()))

    # Rotation: first heading (yaw around Z), then elevation (pitch around X)
    rotateOp = xformable.AddRotateXYZOp()
    rotateOp.Set(Gf.Vec3f(float(elevationDeg), 0.0, float(headingDeg)))


def capturePanorama(world, camera, renderProduct, position: np.ndarray,
                    waypointIdx: int, outputDir: str,
                    headings: int = 12, elevations: list = [-30, 0, 30]):
    """Capture panorama images at *position* (12 headings × 3 elevations)."""
    wpDir = os.path.join(outputDir, f"viewpoint_{waypointIdx:04d}")
    os.makedirs(wpDir, exist_ok=True)

    writer = rep.WriterRegistry.get("BasicWriter")
    writer.initialize(output_dir=wpDir, rgb=True, distance_to_image_plane=True)
    writer.attach([renderProduct])

    imagePaths = []
    for eIdx, elev in enumerate(elevations):
        for hIdx in range(headings):
            heading = hIdx * (360.0 / headings)
            setCameraPose(camera, position, heading, elev)
            for _ in range(3):
                world.step(render=True)
            rep.orchestrator.step()
            imagePaths.append({
                "heading": heading, "elevation": elev,
                "heading_idx": hIdx, "elevation_idx": eIdx,
            })

    writer.detach()
    return imagePaths


# ── 5. Main pipeline ──

def main():
    carb.log_info(f"Loading USDC: {args.usdc_path}")

    # Open the stage
    stage_utils.open_stage(args.usdc_path)
    stage = omni.usd.get_context().get_stage()

    if stage is None:
        carb.log_error("Failed to open stage!")
        simulation_app.close()
        sys.exit(1)

    # Setup physics
    setPhysicsScene(stage)
    add_collision_to_meshes(stage)

    # Create World and reset
    world = World(stage_units_in_meters=1.0)
    world.reset()
    for _ in range(10):
        world.step(render=True)

    # Get scene bounds
    min_pt, max_pt = getBound(stage)
    carb.log_info(f"Scene bounds: min={min_pt}, max={max_pt}")

    floor_z = sampleFloorZ(stage, min_pt, max_pt)
    carb.log_info(f"Estimated floor z: {floor_z}")

    # ── 5a. Sample waypoints ──
    vln_result = None        # set when full pipeline runs
    paths_data = []          # NavPath list from pipeline
    solveStatePath = resolveStatePath(args.usdc_path, args.solve_state)

    if solveStatePath and not args.no_semantic:
        carb.log_info(f"Loading solve_state: {solveStatePath}")
        solveState = loadStateJson(solveStatePath)

        if args.backend == "navmesh":
            from navmesh_pipeline import sample_vln_paths_navmesh
            vln_result = sample_vln_paths_navmesh(
                stage, solveState,
                camera_height=args.camera_height,
                room_wp_base=args.room_waypoints,
                min_wp_distance=args.min_wp_distance,
                max_edge_distance=args.max_connect_distance,
            )
        else:
            vln_result = sample_vln_paths(
                stage, solveState,
                camera_height=args.camera_height,
                room_wp_base=args.room_waypoints,
                min_wp_distance=args.min_wp_distance,
                max_edge_distance=args.max_connect_distance,
                grid_resolution=0.15,
            )

        waypoints = vln_result.waypoints
        waypoint_meta = vln_result.meta
        connectivity = vln_result.connectivity
        paths_data = vln_result.paths
        carb.log_info(f"Pipeline ({args.backend}): {vln_result.summary()}")
    else:
        # ── Fallback: Poisson-disk + raycast (original method) ──
        carb.log_info("Sampling waypoints via Poisson-disk...")
        candidates_2d = poissonDiskSample2D(
            min_pt[:2], max_pt[:2],
            args.min_wp_distance,
            args.num_waypoints * 3,
        )

        waypoints = []
        for xy in candidates_2d:
            if len(waypoints) >= args.num_waypoints:
                break
            ray_origin = np.array([xy[0], xy[1], max_pt[2] - 0.5])
            hit = raycastDown(world, ray_origin)
            if hit['hit']:
                wp = np.array(ray_origin)
                waypoints.append(wp)

        waypoints = np.array(waypoints)
        waypoint_meta = [{"type": "random", "tags": []} for _ in range(len(waypoints))]
        carb.log_info(f"Sampled {len(waypoints)} valid waypoints")

        # Build connectivity via raycasting (fallback only)
        connectivity = {}
        for i in range(len(waypoints)):
            neighbors = []
            for j in range(len(waypoints)):
                if i == j:
                    continue
                dist = float(np.linalg.norm(waypoints[i] - waypoints[j]))
                if dist <= args.max_connect_distance:
                    if raycastBetween(world, waypoints[i], waypoints[j]):
                        neighbors.append({"index": j, "distance": round(dist, 4)})
            connectivity[i] = neighbors

    if len(waypoints) == 0:
        carb.log_error("No valid waypoints found! Check your scene geometry.")
        simulation_app.close()
        sys.exit(1)

    # ── 5c. Visualise waypoints & connectivity ──
    carb.log_info("Rendering waypoint visualisation...")
    markers_root = createWaypointMarkers(stage, waypoints)
    edges_root = createConnectivityLines(stage, waypoints, connectivity)

    # Let the renderer catch up so markers are visible
    for _ in range(20):
        world.step(render=True)

    # ── 5d. Setup camera and render panoramas ──
    carb.log_info("Setting up camera for panorama rendering...")
    output_dir = os.path.join(args.output_dir, args.scan_id)
    os.makedirs(output_dir, exist_ok=True)

    camera, rp = setupCamera(stage, args.image_width, args.image_height)

    all_viewpoint_meta = []
    for wp_idx, wp_pos in enumerate(waypoints):
        carb.log_info(f"Rendering panorama {wp_idx + 1}/{len(waypoints)} at {wp_pos}")
        img_meta = capturePanorama(
            world=world,
            camera=camera,
            renderProduct=rp,
            position=wp_pos,
            waypointIdx=wp_idx,
            outputDir=output_dir,
        )
        all_viewpoint_meta.append({
            "viewpoint_index": wp_idx,
            "position": wp_pos.tolist(),
            "images": img_meta,
        })

    # Remove markers before final metadata (optional: keep for debug)
    # removeMarkers(stage)

    # ── 5e. Export metadata ──
    carb.log_info("Exporting metadata...")

    # Matterport3D-style connectivity (with semantic annotations)
    connectivity_out = []
    for i in range(len(waypoints)):
        node = {
            "image_id": f"{args.scan_id}_{i:04d}",
            "pose": waypoints[i].tolist(),
            "visible": [n["index"] for n in connectivity[i]],
            "unobstructed": [n["index"] for n in connectivity[i]],
            "height": float(waypoints[i][2]),
        }
        # Add semantic info if available
        if i < len(waypoint_meta):
            node["semantic"] = waypoint_meta[i]
        connectivity_out.append(node)

    conn_path = os.path.join(output_dir, f"{args.scan_id}_connectivity.json")
    with open(conn_path, "w") as f:
        json.dump(connectivity_out, f, indent=2)
    carb.log_info(f"Saved connectivity: {conn_path}")

    # Viewpoint metadata (now includes semantic tags)
    for i, vp_meta in enumerate(all_viewpoint_meta):
        if i < len(waypoint_meta):
            vp_meta["semantic"] = waypoint_meta[i]

    meta_path = os.path.join(output_dir, f"{args.scan_id}_viewpoints.json")
    with open(meta_path, "w") as f:
        json.dump(all_viewpoint_meta, f, indent=2)
    carb.log_info(f"Saved viewpoint metadata: {meta_path}")

    # Summary
    from collections import Counter
    type_counts = dict(Counter(m.get("type", "unknown") for m in waypoint_meta))
    summary = {
        "scan_id": args.scan_id,
        "num_waypoints": len(waypoints),
        "num_edges": sum(len(v) for v in connectivity.values()),
        "num_paths": len(paths_data),
        "waypoint_types": type_counts,
        "backend": args.backend if (solveStatePath and not args.no_semantic) else "poisson",
        "scene_bounds_min": min_pt.tolist(),
        "scene_bounds_max": max_pt.tolist(),
        "camera_height": args.camera_height,
        "usdc_path": os.path.abspath(args.usdc_path),
        "solve_state_path": os.path.abspath(solveStatePath) if solveStatePath else None,
    }
    if vln_result is not None:
        summary["pipeline_summary"] = vln_result.summary()
    summary_path = os.path.join(output_dir, f"{args.scan_id}_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    carb.log_info(f"Saved summary: {summary_path}")

    # Export VLN paths (navigation instructions data)
    if paths_data:
        paths_out = [p.to_dict() for p in paths_data]
        paths_path = os.path.join(output_dir, f"{args.scan_id}_paths.json")
        with open(paths_path, "w") as f:
            json.dump(paths_out, f, indent=2)
        carb.log_info(f"Saved {len(paths_out)} paths: {paths_path}")

    carb.log_info("=== VLN data synthesis complete ===")


if __name__ == "__main__":
    main()
    simulation_app.close()