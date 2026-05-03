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
parser.add_argument("--headless", action="store_true", help="Run in headless mode")
parser.add_argument("--solve_state", type=str, default=None,
                    help="Path to solve_state.json (default: <usdc_dir>/../solve_state.json)")
parser.add_argument("--agent_radius", type=float, default=25.0, help="NavMesh agent radius cm")
parser.add_argument("--agent_height", type=float, default=80.0, help="NavMesh agent height cm")
parser.add_argument("--max_step_height", type=float, default=5.0, help="NavMesh max step height cm")
parser.add_argument("--max_slope", type=float, default=30.0, help="NavMesh max walkable slope")
parser.add_argument("--image_width", type=int, default=1280, help="Rendered image width")
parser.add_argument("--image_height", type=int, default=720, help="Rendered image height")

# ── Capture args ──
parser.add_argument("--rgb", action="store_true", help="Capture rgb images")
parser.add_argument("--depth", action="store_true", help="Also capture depth maps (.npy)")
parser.add_argument("--rgb_video", action="store_true", help="Capture rgb video")
parser.add_argument("--camera_height", type=float, default=1.5, help="Camera height above path point (m)")
parser.add_argument("--camera_fov", type=float, default=87.0, help="Horizontal FoV in degrees (>90 triggers panoramic cubemap capture)")
parser.add_argument("--video_fps", type=int, default=30, help="FPS for video mode")
parser.add_argument("--video_step", type=float, default=0.05, help="Interpolation step size for video (m)")
parser.add_argument("--max_capture_paths", type=int, default=3, help="Max paths to capture (0 = all)")


parser.add_argument("--visualize", action="store_true", help="Visualize points and paths")
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
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdLux
from isaacsim.core.api.objects import VisualCone, GroundPlane

from syn_utils.json import readSolveStateJson
from syn_utils.models import States
from syn_utils.sample.nav_mesh import sampleWithNavMesh
from syn_utils.capture import capture_paths


# ── 2b. preprocess ──

def preprocessStage_0(stage: Usd.Stage):
    doors = []
    for prim in stage.Traverse():
        if "door" in prim.GetName().lower():
            path_str = str(prim.GetPath())
            # 如果已有某个 door 路径是当前路径的前缀，跳过
            if not any(path_str.startswith(p + "/") for p in doors):
                doors.append(path_str)

    for door in doors:
        prims_utils.delete_prim(door)

    # from pxr import UsdLux

def preprocessStage_1(stage: Usd.Stage):
    prim = GroundPlane(
        prim_path=f"/World/Plane",
        scale=np.array([100.0, 100.0, 1.0]),
    )

    env_light_prim = prims_utils.get_prim_at_path('/World/env_light')
    env_light_prim.GetAttribute("inputs:intensity").Set(100000.0)

    for prim in stage.Traverse():
        if prim.IsA(UsdLux.SphereLight):
            prim.GetAttribute("inputs:intensity").Set(50000.0)

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

    indices = np.random.permutation(len(path_states))[:10]
    for path_i, idx in enumerate(indices):
        path_state = path_states[idx]
        point = 0
        points = path_state["navmesh_path"].get_points()
        for path_point in points:
            # create a blue visual line at the given path
            prim = VisualCone(
                prim_path=f"/World/Xform/Path{path_i}/Point{point}",
                position=np.array(path_point) + np.array([0.0, 0.0, 0.05]),  # lift it up a bit for better visibility
                radius=0.005,
                height=0.1,
                color=np.array([0.9 - path_i / len(path_states) * 0.8, 0.5, path_i / len(path_states) * 0.8 + 0.1])
            )
            point += 1

def setVisualThings(stage: Usd.Stage):
    for prim in stage.Traverse():
        if {"ceiling", "exterior"} & set(str(prim.GetPath()).split("_")):
            prim.GetAttribute("visibility").Set("invisible")
    

    env_light_prim = prims_utils.get_prim_at_path('/World/env_light')
    env_light_prim.GetAttribute("inputs:intensity").Set(5000.0)


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

    preprocessStage_0(stage)

    carb.log_info("Baking navmesh from solve_state.json...")
    states = readSolveStateJson(args)
    path_states = sampleWithNavMesh(states, args)

    preprocessStage_1(stage)

    if args.rgb_video or args.rgb or args.depth:
        # carb.settings.get_settings().set("/rtx/rendermode", "PathTracing")
        capture_paths(
            path_states=path_states, 
            world=world, 
            stage=stage, 
            args=args
            )
        carb.settings.get_settings().set("/rtx/rendermode", "RaytracedLighting")

    if args.headless:
        for _ in range(30):
            world.step(render=True)
    else:
        if args.visualize:
            navigation_points_visual(stage, states)
            navigation_paths_visual(stage, path_states)
            setVisualThings(stage)
        while simulation_app.is_running():
            world.step(render=True)

    carb.log_info("=== Pipeline complete ===")


if __name__ == "__main__":
    main()
    simulation_app.close()
