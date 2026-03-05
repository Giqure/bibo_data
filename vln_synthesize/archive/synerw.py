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
from isaacsim.core.api import World
import isaacsim.core.utils.stage as stage_utils
from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics

from syn_utils.waypoint import samplePointWithPoissonDisk

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


def raycast_down(world, origin_3d, max_dist=20.0):
    """
    Cast a ray downward from origin_3d to find the floor hit point.
    Returns hit_position (np.array) or None.
    """
    from omni.physx import get_physx_scene_query_interface

    hit_pos = None

    def report_hit(hit):
        nonlocal hit_pos
        hit_pos = np.array([hit.position.x, hit.position.y, hit.position.z])
        return True  # stop at first hit

    origin = carb.Float3(float(origin_3d[0]), float(origin_3d[1]), float(origin_3d[2]))
    direction = carb.Float3(0.0, 0.0, -1.0)

    hit = get_physx_scene_query_interface().raycast_closest(origin, direction, max_dist)
    return hit


def raycast_between(world, p1, p2):
    """Check if there's a clear line of sight between two 3D points."""
    from omni.physx import get_physx_scene_query_interface

    direction = p2 - p1
    dist = float(np.linalg.norm(direction))
    if dist < 1e-6:
        return True
    direction = direction / dist

    hit_info = {"blocked": False}

    def report_hit(hit):
        hit_dist = hit.distance
        if hit_dist < dist - 0.1:  # hit something before reaching p2
            hit_info["blocked"] = True
        return True

    origin = carb.Float3(float(p1[0]), float(p1[1]), float(p1[2]))
    dir3 = carb.Float3(float(direction[0]), float(direction[1]), float(direction[2]))

    get_physx_scene_query_interface().raycast_closest(origin, dir3, dist + 0.5)
    return not hit_info["blocked"]


# ── 3b. Waypoint visualisation helpers ──

def create_waypoint_markers(stage: Usd.Stage, waypoints: np.ndarray, radius: float = 0.15):
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


def create_connectivity_lines(
    stage: Usd.Stage,
    waypoints: np.ndarray,
    connectivity: dict,
    tube_radius: float = 0.03,
):
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
            cyl.GetRadiusAttr().Set(tube_radius)
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


def remove_markers(stage: Usd.Stage):
    """Remove waypoint / edge marker prims so they don't pollute renders."""
    for path in ["/World/WaypointMarkers", "/World/ConnectivityEdges"]:
        prim = stage.GetPrimAtPath(path)
        if prim.IsValid():
            stage.RemovePrim(path)


# ── 4. Panorama rendering with Replicator ──

def setup_camera(stage: Usd.Stage, width: int, height: int):
    """Create a render product for panorama capture."""
    cam_prim = stage.DefinePrim("/World/VLNCamera", "Camera")
    camera = UsdGeom.Camera.Define(stage, "/World/VLNCamera")
    camera.GetClippingRangeAttr().Set(Gf.Vec2f(0.01, 1000.0))

    # Horizontal FOV ~ 60 degrees (for 12 heading angles covering 360)
    focal_length = width / (2.0 * math.tan(math.radians(30)))
    camera.GetFocalLengthAttr().Set(float(focal_length))

    rp = rep.create.render_product("/World/VLNCamera", (width, height))
    return camera, rp


def set_camera_pose(camera_prim, position, heading_deg, elevation_deg):
    """Set camera position and orientation."""
    xformable = UsdGeom.Xformable(camera_prim.GetPrim())
    xformable.ClearXformOpOrder()

    translate_op = xformable.AddTranslateOp()
    translate_op.Set(Gf.Vec3d(*position.tolist()))

    # Rotation: first heading (yaw around Z), then elevation (pitch around X)
    rotate_op = xformable.AddRotateXYZOp()
    # In USD camera convention: looking down -Z, up is +Y
    # We rotate: Z-axis yaw, then X-axis pitch
    rotate_op.Set(Gf.Vec3f(float(elevation_deg), 0.0, float(heading_deg)))


def capture_panorama(
    world: World,
    camera,
    render_product,
    position: np.ndarray,
    waypoint_idx: int,
    output_dir: str,
    headings: int = 12,
    elevations: list = [-30, 0, 30],
):
    """
    Capture a set of images forming a panorama at the given position.
    Headings: 12 x 30° = 360° (like Matterport3D)
    Elevations: 3 levels (-30°, 0°, +30°)
    """
    wp_dir = os.path.join(output_dir, f"viewpoint_{waypoint_idx:04d}")
    os.makedirs(wp_dir, exist_ok=True)

    writer = rep.WriterRegistry.get("BasicWriter")
    writer.initialize(
        output_dir=wp_dir,
        rgb=True,
        distance_to_image_plane=True,  # depth
    )
    writer.attach([render_product])

    image_paths = []
    for e_idx, elev in enumerate(elevations):
        for h_idx in range(headings):
            heading = h_idx * (360.0 / headings)
            set_camera_pose(camera, position, heading, elev)

            # Step simulation to render
            for _ in range(3):  # a few frames for rendering to converge
                world.step(render=True)

            rep.orchestrator.step()

            image_paths.append({
                "heading": heading,
                "elevation": elev,
                "heading_idx": h_idx,
                "elevation_idx": e_idx,
            })

    writer.detach()
    return image_paths


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

    world = World(stage_units_in_meters=1.0)

    # Setup physics
    from syn_utils.simulation import setPhysicsScene, setAllMeshCollision
    setPhysicsScene(stage)
    setAllMeshCollision(stage)
    # Reset world
    world.reset()
    for _ in range(10):
        world.step(render=True)

    # Get scene bounds
    min_pt, max_pt = getBound(stage)
    carb.log_info(f"Scene bounds: min={min_pt}, max={max_pt}")

    # ── 5a. Sample waypoints via Poisson-disk + raycast ──
    carb.log_info("Sampling waypoints...")
    candidates_2d = samplePointWithPoissonDisk(
        min_xy=min_pt[:2],
        max_xy=max_pt[:2],
        min_dist=args.min_wp_distance,
        num_target=args.num_waypoints * 3,  # oversample
    )

    waypoints = []
    for xy in candidates_2d:
        if len(waypoints) >= args.num_waypoints:
            break
        # Cast ray downward from above to find actual floor
        ray_origin = np.array([xy[0], xy[1], max_pt[2] - 0.5])
        hit = raycast_down(world, ray_origin)
        if hit['hit']:
            # Place camera at hit point + camera_height
            wp = np.array(ray_origin)
            waypoints.append(wp)

    waypoints = np.array(waypoints)
    carb.log_info(f"Sampled {len(waypoints)} valid waypoints")

    if len(waypoints) == 0:
        carb.log_error("No valid waypoints found! Check your scene geometry.")
        simulation_app.close()
        sys.exit(1)

    # ── 5b. Build connectivity graph ──
    carb.log_info("Building connectivity graph...")
    connectivity = {}
    for i in range(len(waypoints)):
        neighbors = []
        for j in range(len(waypoints)):
            if i == j:
                continue
            dist = float(np.linalg.norm(waypoints[i] - waypoints[j]))
            if dist <= args.max_connect_distance:
                if raycast_between(world, waypoints[i], waypoints[j]):
                    neighbors.append({"index": j, "distance": round(dist, 4)})
        connectivity[i] = neighbors

    # ── 5c. Visualise waypoints & connectivity ──
    carb.log_info("Rendering waypoint visualisation...")
    markers_root = create_waypoint_markers(stage, waypoints)
    edges_root = create_connectivity_lines(stage, waypoints, connectivity)

    # Let the renderer catch up so markers are visible
    for _ in range(20):
        world.step(render=True)

    # ── 5d. Setup camera and render panoramas ──
    carb.log_info("Setting up camera for panorama rendering...")
    output_dir = os.path.join(args.output_dir, args.scan_id)
    os.makedirs(output_dir, exist_ok=True)

    camera, rp = setup_camera(stage, args.image_width, args.image_height)

    all_viewpoint_meta = []
    for wp_idx, wp_pos in enumerate(waypoints):
        carb.log_info(f"Rendering panorama {wp_idx + 1}/{len(waypoints)} at {wp_pos}")
        img_meta = capture_panorama(
            world=world,
            camera=camera,
            render_product=rp,
            position=wp_pos,
            waypoint_idx=wp_idx,
            output_dir=output_dir,
        )
        all_viewpoint_meta.append({
            "viewpoint_index": wp_idx,
            "position": wp_pos.tolist(),
            "images": img_meta,
        })

    # Remove markers before final metadata (optional: keep for debug)
    # remove_markers(stage)

    # ── 5e. Export metadata ──
    carb.log_info("Exporting metadata...")

    # Matterport3D-style connectivity
    connectivity_out = []
    for i in range(len(waypoints)):
        node = {
            "image_id": f"{args.scan_id}_{i:04d}",
            "pose": waypoints[i].tolist(),
            "visible": [n["index"] for n in connectivity[i]],
            "unobstructed": [n["index"] for n in connectivity[i]],
            "height": float(waypoints[i][2]),
        }
        connectivity_out.append(node)

    conn_path = os.path.join(output_dir, f"{args.scan_id}_connectivity.json")
    with open(conn_path, "w") as f:
        json.dump(connectivity_out, f, indent=2)
    carb.log_info(f"Saved connectivity: {conn_path}")

    # Viewpoint metadata
    meta_path = os.path.join(output_dir, f"{args.scan_id}_viewpoints.json")
    with open(meta_path, "w") as f:
        json.dump(all_viewpoint_meta, f, indent=2)
    carb.log_info(f"Saved viewpoint metadata: {meta_path}")

    # Summary
    summary = {
        "scan_id": args.scan_id,
        "num_waypoints": len(waypoints),
        "num_edges": sum(len(v) for v in connectivity.values()),
        "scene_bounds_min": min_pt.tolist(),
        "scene_bounds_max": max_pt.tolist(),
        "camera_height": args.camera_height,
        "usdc_path": os.path.abspath(args.usdc_path),
    }
    summary_path = os.path.join(output_dir, f"{args.scan_id}_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    carb.log_info(f"Saved summary: {summary_path}")

    carb.log_info("=== VLN data synthesis complete ===")


if __name__ == "__main__":
    main()
    simulation_app.close()