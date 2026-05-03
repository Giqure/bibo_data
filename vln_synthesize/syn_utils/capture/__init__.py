import os

import numpy as np

import carb

from .base import Sensor
from .camera import Camera
from .config import CameraCaptureConfig


def smooth_points_with_b_spline(points: np.ndarray) -> np.ndarray:
    from scipy.interpolate import splprep, splev

    points = np.asarray(points, dtype=float)
    if len(points) < 4:
        return points

    if not np.isfinite(points).all():
        return points

    segment_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    keep_mask = np.concatenate(([True], segment_lengths > 1e-6))
    control_points = points[keep_mask]
    if len(control_points) < 4:
        return points

    chord_lengths = np.linalg.norm(np.diff(control_points, axis=0), axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(chord_lengths)))
    total_length = cumulative[-1]
    if total_length <= 1e-6:
        return points

    u = cumulative / total_length
    sample_u = np.linspace(0.0, 1.0, len(points))
    k = min(3, len(control_points) - 1)

    try:
        tck, _ = splprep(control_points.T, u=u, k=k, s=0.2)
        smoothed = np.stack(splev(sample_u, tck), axis=1)
        smoothed[0] = points[0]
        smoothed[-1] = points[-1]
        return smoothed
    except ValueError:
        return points
        
def capture_paths(stage, world, path_states, args):
    # carb.settings.get_settings().set("/omni/replicator/debug", True)

    Sensor.world = world
    Sensor.stage = stage
    Sensor.path_states = path_states
    Sensor.max_capture_paths = args.max_capture_paths
    # indices
    total = len(Sensor.path_states)
    if Sensor.max_capture_paths > 0:
        Sensor.indices = np.random.permutation(total)[:Sensor.max_capture_paths]
    else:
        Sensor.indices = np.arange(total)

    sensors = [
        Camera(CameraCaptureConfig(
            image_width=args.image_width,
            image_height=args.image_height,
            camera_fov=args.camera_fov,
            camera_height=args.camera_height,
            video_mode=args.rgb_video,
            rgb_mode=args.rgb,
            depth_mode=args.depth,
            video_fps=args.video_fps,
            video_step=args.video_step
        ))
    ]

    os.makedirs(args.output_dir, exist_ok=True)

    for seq, idx in enumerate(Sensor.indices):
        path_state = Sensor.path_states[idx]
        Sensor.points = np.array([np.array(p) for p in path_state["navmesh_path"].get_points()])
        if len(Sensor.points) <= 2:
            carb.log_info(f"Skipping path {idx} with only {len(Sensor.points)} points.")
            continue
        # 去除极近点
        if len(Sensor.points) > 1:
            deltas = np.linalg.norm(Sensor.points[1:] - Sensor.points[:-1], axis=1)
            keep = np.concatenate([[True], deltas > 1e-6])
            Sensor.points = Sensor.points[keep]
        # 平滑
        Sensor.points = smooth_points_with_b_spline(Sensor.points)

        Sensor.path_dir = os.path.join(args.output_dir, f"path_{seq:04d}")
        os.makedirs(Sensor.path_dir, exist_ok=True)
        Sensor.meta = {
            "from_id": path_state.get("from", {}).get("id", ""),
            "from_type": path_state.get("from", {}).get("type", ""),
            "to_id": path_state.get("to", {}).get("id", ""),
            "to_type": path_state.get("to", {}).get("type", ""),
            "path_length": path_state.get("length", 0),
            "points": Sensor.points.tolist(),
        }
        # write meta
        with open(os.path.join(Sensor.path_dir, "meta.txt"), "w") as f:
            for k, v in Sensor.meta.items():
                f.write(f"{k}: {v}\n")

        # def slide_window_3(iterable):
        #     it = iter(iterable)
        #     a, b = next(it), next(it)
        #     for c in it:
        #         yield a, b, c
        #         a, b = b, c
        #
        # for prev_point, point, next_point in slide_window_3(Sensor.points):
        #     for s in sensors:
        #         s.collect(prev_point, point, next_point, meta, path_dir)

        for s in sensors:
            s.collect()
