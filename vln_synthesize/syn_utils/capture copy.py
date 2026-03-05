"""
Image / Video capture along navigation paths using Isaac Sim Replicator.

Supports two modes:
  - "image": Capture discrete frames at each path waypoint.
  - "video": Interpolate between waypoints and write an mp4.

When camera_fov > 90°, automatically switches to **cubemap** rendering
(6 × 90° faces) and stitches into equirectangular panorama.

Output structure (per path):
  <output_dir>/
    path_<idx>/
      rgb/        0000.png, 0001.png, ...
      depth/      0000.npy, ...          (optional)
      metadata.json                      (path info)
      video.mp4                          (video mode only)
"""

import json
import math
import os
from dataclasses import dataclass
from typing import Literal, Sequence

import carb
import numpy as np
import omni.replicator.core as rep
import omni.usd
from pxr import Gf, Sdf, UsdGeom


# ────────────────────────── helpers ──────────────────────────


def _look_at_matrix(eye: np.ndarray, target: np.ndarray, up: np.ndarray = np.array([0, 0, 1])):
    """Return a Gf.Matrix4d placing a camera at *eye* looking toward *target*.
    
    USD/Pixar convention: row vectors, camera looks along local -Z.
    Gf.Matrix4d rows: [0]=localX, [1]=localY, [2]=localZ, [3]=translation.
    """
    forward = target - eye
    norm = np.linalg.norm(forward)
    if norm < 1e-6:
        forward = np.array([1.0, 0.0, 0.0])
    else:
        forward = forward / norm

    right = np.cross(forward, up)
    r_norm = np.linalg.norm(right)
    if r_norm < 1e-6:
        up_alt = np.array([0, 1, 0])
        right = np.cross(forward, up_alt)
        r_norm = np.linalg.norm(right)
    right /= r_norm
    true_up = np.cross(right, forward)

    # Gf.Matrix4d is row-major; camera's local -Z = forward direction
    mat = Gf.Matrix4d(
        float(right[0]),    float(right[1]),    float(right[2]),    0,
        float(true_up[0]),  float(true_up[1]),  float(true_up[2]),  0,
        float(-forward[0]), float(-forward[1]), float(-forward[2]), 0,
        float(eye[0]),      float(eye[1]),      float(eye[2]),      1,
    )
    return mat


def _interpolate_path(points: list[np.ndarray], step_size: float = 0.05) -> list[np.ndarray]:
    """Linearly interpolate path so consecutive samples are ≤ *step_size* apart."""
    if len(points) < 2:
        return list(points)
    result: list[np.ndarray] = [points[0]]
    for i in range(1, len(points)):
        seg = points[i] - points[i - 1]
        length = float(np.linalg.norm(seg))
        if length < 1e-6:
            continue
        n_steps = max(1, int(math.ceil(length / step_size)))
        for j in range(1, n_steps + 1):
            result.append(points[i - 1] + seg * (j / n_steps))
    return result


# ────────────────────── cubemap → equirectangular ──────────────────────

# 6 cube faces: (name, forward, up) in world frame
CUBE_FACES = [
    ("front",  np.array([ 1,  0,  0]), np.array([0, 0, 1])),   # +X
    ("back",   np.array([-1,  0,  0]), np.array([0, 0, 1])),   # -X
    ("left",   np.array([ 0,  1,  0]), np.array([0, 0, 1])),   # +Y
    ("right",  np.array([ 0, -1,  0]), np.array([0, 0, 1])),   # -Y
    ("top",    np.array([ 0,  0,  1]), np.array([-1, 0, 0])),  # +Z
    ("bottom", np.array([ 0,  0, -1]), np.array([ 1, 0, 0])),  # -Z
]

PANORAMA_FOV_THRESHOLD = 90.0  # above this, auto-switch to cubemap mode


def _build_equirect_lut(pano_w: int, pano_h: int, face_size: int):
    """Pre-compute lookup table: equirectangular pixel → cube face + (u,v).

    Returns
    -------
    face_idx : (pano_h, pano_w) int   – which cube face [0..5]
    u_px, v_px : (pano_h, pano_w) float – pixel coords inside that face
    """
    theta = np.linspace(-np.pi, np.pi, pano_w, endpoint=False)
    phi   = np.linspace(np.pi / 2, -np.pi / 2, pano_h, endpoint=False)
    theta, phi = np.meshgrid(theta, phi)

    # World-frame direction
    x = np.cos(phi) * np.cos(theta)
    y = np.cos(phi) * np.sin(theta)
    z = np.sin(phi)

    abs_x, abs_y, abs_z = np.abs(x), np.abs(y), np.abs(z)

    face_idx = np.zeros_like(x, dtype=np.int32)
    uc = np.zeros_like(x)
    vc = np.zeros_like(x)

    # +X (front)
    mask = (x >= abs_y) & (x >= abs_z)
    face_idx[mask] = 0; uc[mask] = y[mask] / x[mask]; vc[mask] = z[mask] / x[mask]

    # -X (back)
    mask = (abs_x >= abs_y) & (abs_x >= abs_z) & (x < 0)
    face_idx[mask] = 1; uc[mask] = y[mask] / x[mask]; vc[mask] = -z[mask] / x[mask]

    # +Y (left)
    mask = (abs_y > abs_x) & (abs_y >= abs_z) & (y >= 0)
    face_idx[mask] = 2; uc[mask] = -x[mask] / y[mask]; vc[mask] = z[mask] / y[mask]

    # -Y (right)
    mask = (abs_y > abs_x) & (abs_y >= abs_z) & (y < 0)
    face_idx[mask] = 3; uc[mask] = -x[mask] / y[mask]; vc[mask] = -z[mask] / y[mask]

    # +Z (top)
    mask = (abs_z > abs_x) & (abs_z > abs_y) & (z >= 0)
    face_idx[mask] = 4; uc[mask] = y[mask] / z[mask]; vc[mask] = -x[mask] / z[mask]

    # -Z (bottom)
    mask = (abs_z > abs_x) & (abs_z > abs_y) & (z < 0)
    face_idx[mask] = 5; uc[mask] = -y[mask] / z[mask]; vc[mask] = -x[mask] / z[mask]

    # [-1,1] → pixel coords
    # NOTE: uc is negated (1 - uc) because the camera right-axis in
    # _look_at_matrix mirrors the standard cubemap uc direction for every face.
    u_px = ((1 - uc) * 0.5 * (face_size - 1)).astype(np.float32)
    v_px = ((1 - vc) * 0.5 * (face_size - 1)).astype(np.float32)

    return face_idx, u_px, v_px


def _stitch_cubemap_to_equirect(
    faces: list[np.ndarray],
    pano_w: int,
    pano_h: int,
    lut: tuple | None = None,
) -> np.ndarray:
    """Stitch 6 cube face images into one equirectangular panorama."""
    from scipy.ndimage import map_coordinates  # type: ignore

    face_size = faces[0].shape[0]
    if lut is None:
        face_idx, u_px, v_px = _build_equirect_lut(pano_w, pano_h, face_size)
    else:
        face_idx, u_px, v_px = lut

    n_ch = faces[0].shape[2] if faces[0].ndim == 3 else 1
    pano = np.zeros((pano_h, pano_w, n_ch), dtype=faces[0].dtype)

    for fi in range(6):
        mask = face_idx == fi
        if not np.any(mask):
            continue
        fimg = faces[fi] if faces[fi].ndim == 3 else faces[fi][:, :, np.newaxis]
        for c in range(n_ch):
            vals = map_coordinates(
                fimg[:, :, c].astype(np.float32),
                [v_px[mask], u_px[mask]],
                order=1, mode='nearest',
            )
            pano[mask, c] = vals.astype(pano.dtype)

    return pano


# ────────────────────────── config ──────────────────────────


@dataclass
class CaptureConfig:
    """Aggregated capture settings (populated from argparse)."""
    output_dir: str
    mode: Literal["image", "video"] = "image"
    width: int = 640
    height: int = 480
    camera_height: float = 1.5          # metres above path point
    camera_fov: float = 180.0           # horizontal FoV degrees
    capture_depth: bool = False
    video_step_size: float = 0.05       # interpolation step for video (m)
    video_fps: int = 30
    max_paths: int = 0                  # 0 = capture all


# ────────────────────────── core ──────────────────────────


class PathCapture:
    """Manages a USD camera and frame-by-frame capture for one path.

    When camera_fov > 90°, renders 6 cubemap faces (each 90° rectilinear)
    and stitches them into equirectangular panorama.

    Uses Replicator annotators attached once to a persistent render product.
    Frames are triggered by world.step(render=True).
    """

    CAMERA_PATH = "/World/CaptureCamera"

    def __init__(self, cfg: CaptureConfig):
        self.cfg = cfg
        self._stage = omni.usd.get_context().get_stage()
        self._is_pano = cfg.camera_fov > PANORAMA_FOV_THRESHOLD

        if self._is_pano:
            # Cubemap: render square faces; face_size = max(w,h)//2 is a good balance
            self._face_size = max(cfg.width, cfg.height) // 2
            self._cam_prim = self._ensure_camera(fov=90.0, face_size=self._face_size)
            rp_size = (self._face_size, self._face_size)
            # Pre-compute equirect LUT
            self._equirect_lut = _build_equirect_lut(cfg.width, cfg.height, self._face_size)
            carb.log_info(
                f"Panoramic mode: face_size={self._face_size}, "
                f"output={cfg.width}×{cfg.height} equirectangular"
            )
        else:
            self._face_size = 0
            self._cam_prim = self._ensure_camera(fov=cfg.camera_fov)
            rp_size = (cfg.width, cfg.height)
            self._equirect_lut = None

        # Create render product & annotators ONCE
        self._rp = rep.create.render_product(self.CAMERA_PATH, rp_size)

        self._rgb_annot = rep.AnnotatorRegistry.get_annotator("rgb")
        self._rgb_annot.attach([self._rp])

        self._depth_annot = None
        if cfg.capture_depth:
            self._depth_annot = rep.AnnotatorRegistry.get_annotator("distance_to_camera")
            self._depth_annot.attach([self._rp])

        carb.log_info("PathCapture initialised (render product + annotators attached)")

    def warm_up(self, world, n_steps: int = 20):
        """Render several frames so the annotator pipeline is fully primed."""
        if world is None:
            return
        for _ in range(n_steps):
            world.step(render=True)
        test = self._rgb_annot.get_data()
        if test is None or not hasattr(test, 'size') or test.size == 0:
            carb.log_warn("RGB annotator still empty after warm-up – trying more steps")
            for _ in range(20):
                world.step(render=True)
        else:
            pxmax = int(test.max()) if test.size > 0 else 0
            carb.log_info(f"RGB annotator primed, shape={test.shape}, max_pixel={pxmax}")
        carb.log_info("PathCapture warm-up done")

    # ── camera ──

    def _ensure_camera(self, fov: float = 90.0, face_size: int | None = None):
        prim = self._stage.GetPrimAtPath(self.CAMERA_PATH)
        if prim.IsValid():
            return prim

        h_aperture = 20.955  # mm
        focal_length = h_aperture / (2 * math.tan(math.radians(fov / 2)))
        # Square sensor for cubemap, or aspect-ratio matched for rectilinear
        if face_size is not None:
            v_aperture = h_aperture  # square
        else:
            aspect = self.cfg.width / self.cfg.height
            v_aperture = h_aperture / aspect

        cam = UsdGeom.Camera.Define(self._stage, self.CAMERA_PATH)
        cam.GetHorizontalApertureAttr().Set(h_aperture)
        cam.GetVerticalApertureAttr().Set(v_aperture)
        cam.GetFocalLengthAttr().Set(focal_length)
        cam.GetClippingRangeAttr().Set(Gf.Vec2f(0.01, 1000.0))
        carb.log_info(
            f"Created capture camera at {self.CAMERA_PATH}  "
            f"fov={fov}° focal={focal_length:.3f}mm "
            f"aperture=({h_aperture}, {v_aperture:.3f})mm"
        )
        return cam.GetPrim()

    def _set_camera_pose_from_matrix(self, mat: Gf.Matrix4d):
        """Set camera transform directly from a Gf.Matrix4d."""
        xformable = UsdGeom.Xformable(self._cam_prim)
        xformable.ClearXformOpOrder()
        op = xformable.AddTransformOp()
        op.Set(mat)

    def _set_camera_pose(self, eye: np.ndarray, target: np.ndarray):
        mat = _look_at_matrix(eye, target)
        self._set_camera_pose_from_matrix(mat)

    # ── cubemap helpers ──

    def _render_cubemap_faces(self, eye: np.ndarray, world) -> tuple[list[np.ndarray], list[np.ndarray | None]]:
        """Render 6 cube faces and return (rgb_faces, depth_faces)."""
        rgb_faces: list[np.ndarray] = []
        depth_faces: list[np.ndarray | None] = []

        for _name, forward, up in CUBE_FACES:
            target = eye + forward.astype(np.float64)
            mat = _look_at_matrix(eye, target, up=up)
            self._set_camera_pose_from_matrix(mat)

            # Render
            if world is not None:
                for _ in range(3):
                    world.step(render=True)

            # Read RGB
            rgb_data = self._rgb_annot.get_data()
            if rgb_data is not None and hasattr(rgb_data, "shape") and rgb_data.size > 0:
                if rgb_data.ndim == 3 and rgb_data.shape[2] == 4:
                    rgb_data = rgb_data[:, :, :3]
            else:
                rgb_data = np.zeros((self._face_size, self._face_size, 3), dtype=np.uint8)
            rgb_faces.append(rgb_data)

            # Read Depth
            if self._depth_annot is not None:
                depth_data = self._depth_annot.get_data()
                if depth_data is not None and hasattr(depth_data, "shape") and depth_data.size > 0:
                    depth_faces.append(depth_data.astype(np.float32))
                else:
                    depth_faces.append(None)
            else:
                depth_faces.append(None)

        return rgb_faces, depth_faces

    # ── public API ──

    def capture_path(
        self,
        path_index: int,
        path_points: Sequence[np.ndarray],
        meta: dict | None = None,
        world=None,
    ) -> str:
        if len(path_points) < 2:
            carb.log_warn(f"Path {path_index} has fewer than 2 points – skipped.")
            return ""

        points = [np.asarray(p, dtype=np.float64) for p in path_points]

        if self.cfg.mode == "video":
            points = _interpolate_path(points, self.cfg.video_step_size)

        path_dir = os.path.join(self.cfg.output_dir, f"path_{path_index:04d}")
        rgb_dir = os.path.join(path_dir, "rgb")
        os.makedirs(rgb_dir, exist_ok=True)
        if self.cfg.capture_depth:
            os.makedirs(os.path.join(path_dir, "depth"), exist_ok=True)

        frame_paths: list[str] = []

        for fi, pt in enumerate(points):
            eye = pt.copy()
            eye[2] += self.cfg.camera_height

            rgb_path = os.path.join(rgb_dir, f"{fi:04d}.png")

            if self._is_pano:
                # ── Panoramic: cubemap → equirect ──
                rgb_faces, depth_faces = self._render_cubemap_faces(eye, world)
                pano_rgb = _stitch_cubemap_to_equirect(
                    rgb_faces, self.cfg.width, self.cfg.height, lut=self._equirect_lut
                )
                from PIL import Image  # type: ignore
                Image.fromarray(pano_rgb.squeeze() if pano_rgb.shape[2] == 1 else pano_rgb).save(rgb_path)

                if self.cfg.capture_depth:
                    valid_depths = [d for d in depth_faces if d is not None]
                    if len(valid_depths) == 6:
                        depth_faces_3d = [d[:, :, np.newaxis] if d.ndim == 2 else d for d in valid_depths]
                        pano_depth = _stitch_cubemap_to_equirect(
                            depth_faces_3d, self.cfg.width, self.cfg.height, lut=self._equirect_lut
                        )
                        np.save(os.path.join(path_dir, "depth", f"{fi:04d}.npy"), pano_depth.squeeze())
            else:
                # ── Rectilinear: single shot ──
                if fi + 1 < len(points):
                    target = points[fi + 1].copy()
                else:
                    target = points[fi - 1].copy()
                    target = 2 * eye - target
                target[2] += self.cfg.camera_height

                self._set_camera_pose(eye, target)
                if world is not None:
                    for _ in range(3):
                        world.step(render=True)
                self._save_frame(rgb_path, fi, path_dir)

            frame_paths.append(rgb_path)

        # Save metadata
        metadata = {
            "path_index": path_index,
            "num_frames": len(frame_paths),
            "mode": self.cfg.mode,
            "camera_height": self.cfg.camera_height,
            "camera_fov": self.cfg.camera_fov,
            "panoramic": self._is_pano,
            "resolution": [self.cfg.width, self.cfg.height],
        }
        if meta:
            metadata.update(meta)
        with open(os.path.join(path_dir, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2, default=str)

        if self.cfg.mode == "video":
            self._assemble_video(rgb_dir, path_dir)

        carb.log_info(f"Path {path_index}: captured {len(frame_paths)} frames → {path_dir}")
        return path_dir

    # ── internal capture ──

    def _save_frame(self, rgb_path: str, frame_idx: int, path_dir: str):
        """Read annotator buffers (already updated by world.step) and save to disk."""
        from PIL import Image  # type: ignore

        rgb_data = self._rgb_annot.get_data()
        if rgb_data is not None and hasattr(rgb_data, "shape") and rgb_data.size > 0:
            if rgb_data.ndim == 3 and rgb_data.shape[2] == 4:
                img = Image.fromarray(rgb_data[:, :, :3])
            else:
                img = Image.fromarray(rgb_data)
            img.save(rgb_path)
        else:
            carb.log_warn(f"Frame {frame_idx}: empty RGB data")

        if self._depth_annot is not None:
            depth_data = self._depth_annot.get_data()
            if depth_data is not None and hasattr(depth_data, "shape") and depth_data.size > 0:
                depth_path = os.path.join(path_dir, "depth", f"{frame_idx:04d}.npy")
                np.save(depth_path, depth_data.astype(np.float32))

    def _assemble_video(self, rgb_dir: str, path_dir: str):
        """Stitch captured PNGs into an mp4 using ffmpeg (if available) or imageio."""
        video_path = os.path.join(path_dir, "video.mp4")
        try:
            import imageio.v3 as iio  # type: ignore

            frames = sorted(f for f in os.listdir(rgb_dir) if f.endswith(".png"))
            if not frames:
                return
            writer = iio.imopen(video_path, "w", plugin="pyav")
            writer.init_video_stream("libx264", fps=self.cfg.video_fps)
            for fname in frames:
                frame = iio.imread(os.path.join(rgb_dir, fname))
                writer.write_frame(frame)
            writer.close()
            carb.log_info(f"Video saved: {video_path}")
        except ImportError:
            # Fallback: try ffmpeg CLI
            import subprocess
            pattern = os.path.join(rgb_dir, "%04d.png")
            cmd = [
                "ffmpeg", "-y",
                "-framerate", str(self.cfg.video_fps),
                "-i", pattern,
                "-c:v", "libx264", "-pix_fmt", "yuv420p",
                video_path,
            ]
            try:
                subprocess.run(cmd, check=True, capture_output=True)
                carb.log_info(f"Video saved (ffmpeg): {video_path}")
            except (subprocess.CalledProcessError, FileNotFoundError) as e:
                carb.log_warn(f"Video assembly failed: {e}. Raw frames kept in {rgb_dir}")

    def destroy(self):
        """Detach annotators and clean up render product."""
        try:
            self._rgb_annot.detach([self._rp])
            if self._depth_annot is not None:
                self._depth_annot.detach([self._rp])
            self._rp = None
        except Exception:
            pass


# ────────────────────────── convenience ──────────────────────────


def capture_all_paths(
    path_states: list[dict],
    cfg: CaptureConfig,
    world=None,
) -> list[str]:
    """Iterate over computed path_states and capture images/video for each.

    Returns list of output directories.
    """
    capturer = PathCapture(cfg)
    capturer.warm_up(world, n_steps=10)
    results: list[str] = []
    total = len(path_states)
    if cfg.max_paths > 0:
        indices = np.random.permutation(total)[: cfg.max_paths]
    else:
        indices = np.arange(total)

    for seq, idx in enumerate(indices):
        ps = path_states[int(idx)]
        nav_path = ps.get("navmesh_path")
        if nav_path is None:
            continue
        raw_points = list(nav_path.get_points())  # list of Vec3-like
        points = [np.array([p.x, p.y, p.z] if hasattr(p, "x") else p) for p in raw_points]

        meta = {
            "from_id": ps.get("from", {}).get("id", ""),
            "from_type": ps.get("from", {}).get("type", ""),
            "to_id": ps.get("to", {}).get("id", ""),
            "to_type": ps.get("to", {}).get("type", ""),
            "path_length": float(nav_path.length()) if callable(nav_path.length) else float(nav_path.length),
        }

        out = capturer.capture_path(seq, points, meta=meta, world=world)
        if out:
            results.append(out)
        carb.log_info(f"[Capture] {seq + 1}/{len(indices)} done")

    capturer.destroy()
    carb.log_info(f"Capture complete: {len(results)} paths → {cfg.output_dir}")
    return results
