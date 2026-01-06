"""Standalone worker for VHACD to URDF generation (no bpy imports)."""
import os
import time
import numpy as np
import trimesh
import pyVHACD
from urdf_utils import create_aabb_urdf_snippet

def compute(task):
    """Load mesh data from npz, run VHACD, and write URDF."""
    name, npz_path, visual_filename, output_dir, asset_type = task

    data = np.load(npz_path)
    verts = data["verts"]
    faces = data["faces"]

    t_mesh = trimesh.Trimesh(vertices=verts, faces=faces)
    if not t_mesh.is_watertight:
        trimesh.repair.fill_holes(t_mesh)
    volume = t_mesh.volume if t_mesh.is_watertight else 0.0
    clean_verts = t_mesh.vertices.astype(np.float64)
    clean_faces = t_mesh.faces.astype(np.uint32)
    
    hulls = pyVHACD.compute_vhacd(clean_verts, clean_faces)
    
    urdf_xml = create_aabb_urdf_snippet(
        name,
        hulls,
        visual_filename,
        asset_type=asset_type,
        volume=volume,
    )
    urdf_path = os.path.join(output_dir, f"{name}.urdf")
    with open(urdf_path, "w") as f:
        f.write(urdf_xml)
    return (
        "SUCCESS: "
        f"{name} hulls={len(hulls)} "
    )
