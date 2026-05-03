from dataclasses import dataclass, field
from typing import Any
import numpy as np



@dataclass
class CameraCaptureConfig:
    # mode settings
    video_mode: bool = False
    rgb_mode: bool = False
    depth_mode: bool = False

    camera_height: float = 1.5
    image_width: int = 1280
    image_height: int = 720
    camera_fov: float = 87.0
    
    video_fps: int = 30
    video_step: float = 0.05
