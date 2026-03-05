"""
NavMesh visualization using pure Isaac Sim (no IsaacLab dependency).
Loads a USDC scene, bakes navmesh from solve_state.json, and visualizes it.

Usage:
    ~/.local/share/ov/pkg/isaac-sim-4.5.0/python.sh vln_synthesize/syner.py \
        --usdc_path /path/to/scene.usdc \
        --output_dir /path/to/output \
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
parser.add_argument("--image_width", type=int, default=640, help="Rendered image width")
parser.add_argument("--image_height", type=int, default=480, help="Rendered image height")
parser.add_argument("--headless", action="store_true", help="Run in headless mode")
parser.add_argument("--solve_state", type=str, default=None,
                    help="Path to solve_state.json (default: <usdc_dir>/../solve_state.json)")
parser.add_argument("--agent_radius", type=float, default=25.0, help="NavMesh agent radius cm")
parser.add_argument("--agent_height", type=float, default=80.0, help="NavMesh agent height cm")
parser.add_argument("--max_step_height", type=float, default=5.0, help="NavMesh max step height cm")
parser.add_argument("--max_slope", type=float, default=30.0, help="NavMesh max walkable slope")
parser.add_argument("--eye_height", type=float, default=1.5, help="Camera height above nav point (meters)")
parser.add_argument("--headings", type=int, default=12, help="Number of heading angles per panorama")
parser.add_argument("--elevations", type=str, default="-30,0,30", help="Comma-separated elevation angles")
parser.add_argument("--settle_frames", type=int, default=5, help="Frames to wait for rendering convergence")
parser.add_argument("--vp_min_distance", type=float, default=0.3, help="Min distance between viewpoints (meters)")
parser.add_argument("--path_sample_interval", type=float, default=1.0, help="Sample interval along paths (meters)")
args = parser.parse_args()
args.elevations = [float(e) for e in args.elevations.split(",")]

# ── 2. Launch Isaac Sim ──

from isaacsim import SimulationApp

sim_config = {
    "headless": args.headless,
    "width": args.image_width,
    "height": args.image_height,
    "anti_aliasing": 3,  # DLAA
}
simulation_app = SimulationApp(sim_config)
from isaacsim.core.utils.extensions import enable_extension  # type: ignore
enable_extension("omni.anim.navigation.bundle")
simulation_app.update()

# Now we can import Omniverse modules
import carb
import omni.replicator.core as rep
import omni.usd
from isaacsim.core.api import World
import isaacsim.core.utils.stage as stage_utils
import isaacsim.core.utils.prims as prims_utils
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

from syn_utils.json import readSolveStateJson
from syn_utils.points.nav_mesh import sampleWithNavMesh
from syn_utils.models import States


# ── 2b. preprocess ──

def preprocessStage(stage: Usd.Stage):
    doors = []
    for prim in stage.Traverse():
        if "door" in prim.GetName().lower():
            doors.append(prim)

    for door in doors:
        prims_utils.delete_prim(door.GetPath())

# ── 3. Visualize ──

def create_navmesh_visual(stage: Usd.Stage, i_nav, area_index: int = 0):
    """Visualize baked navmesh triangles as a USD mesh prim."""
    navmesh = i_nav.get_navmesh()
    verts = navmesh.get_draw_triangles(area_index)
    if not verts:
        carb.log_warn(f"No navmesh triangles available in area {area_index}")
        return None

    points = [Gf.Vec3f(float(v.x), float(v.y), float(v.z)) for v in verts]
    tri_count = len(points) // 3
    points = points[: tri_count * 3]

    mesh_path = f"/World/NavMeshVis/area_{area_index:02d}"
    stage.DefinePrim("/World/NavMeshVis", "Xform")
    mesh = UsdGeom.Mesh.Define(stage, mesh_path)
    mesh.GetPointsAttr().Set(points)
    mesh.GetFaceVertexCountsAttr().Set([3] * tri_count)
    mesh.GetFaceVertexIndicesAttr().Set(list(range(tri_count * 3)))
    mesh.GetSubdivisionSchemeAttr().Set("none")

    from pxr import UsdShade

    mat_path = "/World/NavMeshVis/NavMeshMat"
    mat = UsdShade.Material.Define(stage, mat_path)
    shader = UsdShade.Shader.Define(stage, f"{mat_path}/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.1, 0.6, 1.0))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.4)
    shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(0.35)
    mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(mat)

    carb.log_info(f"Created navmesh visualization mesh: {mesh_path}, triangles={tri_count}")
    return mesh_path

def navigation_points_visual(stage: Usd.Stage, states: States):
    from isaacsim.core.api.objects import VisualCone
    point_i = 0
    for state_dict in states.values():
        for state in state_dict.values():
            for navigation_point in state.navigation_points:
                import numpy as np

                # create a red visual cone at the given path
                prim = VisualCone(
                    prim_path=f"/World/Xform/Cone{point_i}",
                    position=navigation_point,
                    radius=0.005,
                    height=0.1,
                    color=np.array([1.0, 0.0, 0.0])
                )
                point_i += 1

def navigation_paths_visual(stage: Usd.Stage, path_states):
    from isaacsim.core.api.objects import VisualCone
    path_i = 0
    indices = np.random.permutation(len(path_states))[:10]
    for idx in indices:
        path_state = path_states[idx]
        point = 0
        for path_point in path_state["navmesh_path"].get_points():
            # create a blue visual line at the given path
            prim = VisualCone(
                prim_path=f"/World/Xform/Path{path_i}/Point{point}",
                position=np.array(path_point) + np.array([0.0, 0.0, 0.05]),  # lift it up a bit for better visibility
                radius=0.005,
                height=0.1,
                color=np.array([0.9 - path_i / len(path_states) * 0.8, 1.0, path_i / len(path_states) * 0.8 + 0.1])
            )
            point += 1
        path_i += 1

def setVisualThings(stage: Usd.Stage):
    for prim in stage.Traverse():
        if {"ceiling", "exterior"} & set(str(prim.GetPath()).split("_")):
            prim.GetAttribute("visibility").Set("invisible")

# ── 4. Panorama rendering with Replicator ──

def setup_camera(stage: Usd.Stage, width: int, height: int, num_headings: int = 12):
    """Create camera, render product, and annotators for panorama capture.

    Uses annotators directly instead of BasicWriter for precise control
    over file naming and output formats.
    """
    camera = UsdGeom.Camera.Define(stage, "/World/VLNCamera")
    camera.GetClippingRangeAttr().Set(Gf.Vec2f(0.01, 1000.0))

    # Horizontal FOV = 360 / num_headings (e.g. 30° for 12 headings)
    half_fov = 180.0 / num_headings
    focal_length = width / (2.0 * math.tan(math.radians(half_fov)))
    camera.GetFocalLengthAttr().Set(float(focal_length))

    rp = rep.create.render_product("/World/VLNCamera", (width, height))

    # Annotators: direct access to rendered buffers
    rgb_annot = rep.AnnotatorRegistry.get_annotator("rgb", device="cpu")
    rgb_annot.attach(rp)
    depth_annot = rep.AnnotatorRegistry.get_annotator("distance_to_image_plane", device="cpu")
    depth_annot.attach(rp)

    return camera, rp, {"rgb": rgb_annot, "depth": depth_annot}


def set_camera_pose(camera_prim, position: np.ndarray, heading_deg: float, elevation_deg: float):
    """Set camera position and orientation.

    Args:
        camera_prim: USD camera prim.
        position: 3D position [x, y, z] (eye‑height already added).
        heading_deg: Yaw in degrees (0 = +X, 90 = +Y, CCW).
        elevation_deg: Pitch in degrees (positive = look up).
    """
    xformable = UsdGeom.Xformable(camera_prim.GetPrim())
    xformable.ClearXformOpOrder()

    translate_op = xformable.AddTranslateOp()
    translate_op.Set(Gf.Vec3d(*position.tolist()))

    # RotateXYZ: applied as Rz(heading) · Ry(0) · Rx(elevation)
    rotate_op = xformable.AddRotateXYZOp()
    rotate_op.Set(Gf.Vec3f(float(elevation_deg), 0.0, float(heading_deg)))


def _save_annotator_data(
    annotators: dict,
    wp_dir: str,
    h_idx: int,
    e_idx: int,
):
    """Read annotator buffers and save to disk with structured filenames.

    File naming:  rgb_e{elevation_idx}_h{heading_idx}.png / .npy
    """
    from PIL import Image  # lightweight; imported lazily

    rgb_data = annotators["rgb"].get_data()        # (H, W, 4) uint8 RGBA
    depth_data = annotators["depth"].get_data()     # (H, W) float32

    tag = f"e{e_idx:01d}_h{h_idx:02d}"

    # RGB → save as PNG (drop alpha)
    rgb_path = os.path.join(wp_dir, f"rgb_{tag}.png")
    Image.fromarray(rgb_data[:, :, :3]).save(rgb_path)

    # Depth → save as compressed .npz
    depth_path = os.path.join(wp_dir, f"depth_{tag}.npz")
    np.savez_compressed(depth_path, depth=depth_data)

    return rgb_path, depth_path


def capture_panorama(
    world: World,
    camera,
    annotators: dict,
    position: np.ndarray,
    vp_id: str,
    output_dir: str,
    headings: int = 12,
    elevations: list[float] | None = None,
    settle_frames: int = 5,
) -> dict:
    """Capture a full panorama at *position* and write images to disk.

    Returns a metadata dict describing every captured image.
    """
    if elevations is None:
        elevations = [-30.0, 0.0, 30.0]

    wp_dir = os.path.join(output_dir, vp_id)
    os.makedirs(wp_dir, exist_ok=True)

    images_meta: list[dict] = []
    for e_idx, elev in enumerate(elevations):
        for h_idx in range(headings):
            heading = h_idx * (360.0 / headings)
            set_camera_pose(camera, position, heading, elev)

            # Let the renderer converge (path tracing / DLAA)
            for _ in range(settle_frames):
                world.step(render=False)
            world.render()  # single render pass to update annotator buffers

            rgb_path, depth_path = _save_annotator_data(
                annotators, wp_dir, h_idx, e_idx,
            )

            images_meta.append({
                "heading_deg": heading,
                "elevation_deg": elev,
                "heading_idx": h_idx,
                "elevation_idx": e_idx,
                "rgb": os.path.relpath(rgb_path, output_dir),
                "depth": os.path.relpath(depth_path, output_dir),
            })

    return {"viewpoint_id": vp_id, "position": position.tolist(), "images": images_meta}


# ── 4b. Viewpoint collection & graph ──

def _sample_path_points(path_points: list, interval: float) -> list[np.ndarray]:
    """Sub‑sample a polyline at fixed *interval* distance."""
    pts = [np.array(p) for p in path_points]
    if len(pts) < 2:
        return pts
    sampled = [pts[0]]
    accum = 0.0
    for i in range(1, len(pts)):
        seg = np.linalg.norm(pts[i] - pts[i - 1])
        accum += seg
        if accum >= interval:
            sampled.append(pts[i])
            accum = 0.0
    # always include the last point
    if np.linalg.norm(sampled[-1] - pts[-1]) > 0.01:
        sampled.append(pts[-1])
    return sampled


def collect_viewpoints(
    path_states: list[dict],
    states: States,
    min_distance: float = 0.3,
    path_sample_interval: float = 1.0,
) -> list[dict]:
    """Gather unique viewpoints from navigation points and path waypoints.

    Each viewpoint: {"id": str, "position": np.ndarray, "source": str}
    Points closer than *min_distance* are merged.
    """
    raw_points: list[dict] = []

    # 1) Navigation points from all states (objects, rooms, cutters)
    for state_type, state_dict in states.items():
        for state_id, state in state_dict.items():
            for pt in state.navigation_points:
                raw_points.append({
                    "position": np.array(pt, dtype=np.float64),
                    "source": f"{state_type}/{state_id}",
                })

    # 2) Sub‑sampled points along each NavMesh path
    for ps in path_states:
        nav_path = ps.get("navmesh_path")
        if nav_path is None:
            continue
        for pt in _sample_path_points(nav_path.get_points(), path_sample_interval):
            raw_points.append({
                "position": np.array(pt, dtype=np.float64),
                "source": "path",
            })

    # 3) Deduplicate by minimum distance (greedy)
    unique: list[dict] = []
    for rp in raw_points:
        pos = rp["position"]
        if any(np.linalg.norm(pos - u["position"]) < min_distance for u in unique):
            continue
        rp["id"] = f"vp_{len(unique):04d}"
        unique.append(rp)

    carb.log_info(f"Collected {len(unique)} unique viewpoints from {len(raw_points)} candidates")
    return unique


def build_viewpoint_graph(
    viewpoints: list[dict],
    path_states: list[dict],
    max_edge_distance: float = 5.0,
) -> dict[str, list[str]]:
    """Build an adjacency graph: two viewpoints are connected if a NavMesh
    path exists between them with length ≤ *max_edge_distance*.

    Falls back to Euclidean proximity if path info is insufficient.
    """
    positions = np.array([vp["position"] for vp in viewpoints])  # (N, 3)
    n = len(viewpoints)
    adjacency: dict[str, set[str]] = {vp["id"]: set() for vp in viewpoints}

    # Euclidean neighbor candidates (fast filter)
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.linalg.norm(positions[i] - positions[j])
            if dist <= max_edge_distance:
                adjacency[viewpoints[i]["id"]].add(viewpoints[j]["id"])
                adjacency[viewpoints[j]["id"]].add(viewpoints[i]["id"])

    return {k: sorted(v) for k, v in adjacency.items()}


# ── 4c. Capture pipeline orchestrator ──

def run_capture_pipeline(
    world: World,
    stage: Usd.Stage,
    path_states: list[dict],
    states: States,
    args: argparse.Namespace,
):
    """End‑to‑end image capture: collect viewpoints → capture panoramas → save metadata."""
    output_dir = os.path.join(args.output_dir, "panoramas")
    os.makedirs(output_dir, exist_ok=True)

    # Hide ceilings / exteriors for unobstructed shots
    setVisualThings(stage)

    # Camera & annotators
    camera, rp, annotators = setup_camera(
        stage, args.image_width, args.image_height, num_headings=args.headings,
    )

    # Viewpoints
    viewpoints = collect_viewpoints(
        path_states, states,
        min_distance=args.vp_min_distance,
        path_sample_interval=args.path_sample_interval,
    )
    if not viewpoints:
        carb.log_warn("No viewpoints collected — skipping capture.")
        return

    # Graph
    graph = build_viewpoint_graph(viewpoints, path_states)

    # Warm‑up renderer
    carb.log_info("Warming up renderer …")
    for _ in range(10):
        world.step(render=True)

    # Capture loop
    all_vp_meta: list[dict] = []
    total = len(viewpoints)
    for idx, vp in enumerate(viewpoints):
        pos_with_eye = vp["position"].copy()
        pos_with_eye[2] += args.eye_height  # lift camera to eye level

        carb.log_info(f"Capturing viewpoint {idx + 1}/{total}: {vp['id']}")
        vp_meta = capture_panorama(
            world, camera, annotators,
            position=pos_with_eye,
            vp_id=vp["id"],
            output_dir=output_dir,
            headings=args.headings,
            elevations=args.elevations,
            settle_frames=args.settle_frames,
        )
        vp_meta["source"] = vp["source"]
        vp_meta["nav_position"] = vp["position"].tolist()  # ground‑level
        vp_meta["neighbors"] = graph.get(vp["id"], [])
        all_vp_meta.append(vp_meta)

    # Save dataset metadata
    dataset_meta = {
        "scene": os.path.basename(args.usdc_path),
        "image_size": [args.image_width, args.image_height],
        "headings": args.headings,
        "elevations": args.elevations,
        "eye_height": args.eye_height,
        "num_viewpoints": len(all_vp_meta),
        "viewpoints": all_vp_meta,
        "paths": [
            {
                "from": ps["from"]["id"],
                "to": ps["to"]["id"],
                "length": float(ps["length"]) if not callable(ps["length"]) else float(ps["length"]()),
            }
            for ps in path_states
        ],
    }
    meta_path = os.path.join(output_dir, "dataset_meta.json")
    with open(meta_path, "w") as f:
        json.dump(dataset_meta, f, indent=2)
    carb.log_info(f"Dataset metadata saved to {meta_path}")
    carb.log_info(f"=== Capture complete: {len(all_vp_meta)} viewpoints ===")


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

    preprocessStage(stage)

    carb.log_info("Baking navmesh from solve_state.json...")
    states = readSolveStateJson(args)
    # i_nav = sampleWithNavMesh(states, args)
    # if i_nav is None:
    #     carb.log_error("NavMesh bake failed; cannot visualize navmesh")
    #     simulation_app.close()
    #     sys.exit(1)
    # create_navmesh_visual(stage, i_nav, area_index=0)
    path_states = sampleWithNavMesh(states, args)
    navigation_paths_visual(stage, path_states)
    navigation_points_visual(stage, states)

    # ── Image capture ──
    if args.capture:
        run_capture_pipeline(world, stage, path_states, states, args)

    if args.headless:
        for _ in range(30):
            world.step(render=True)
    else:
        setVisualThings(stage)
        carb.log_info("NavMesh visualized. Close the simulator window to exit.")
        while simulation_app.is_running():
            world.step(render=True)

    carb.log_info("=== Pipeline complete ===")


if __name__ == "__main__":
    main()
    simulation_app.close()