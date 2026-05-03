import math
import os
import numpy as np

from .config import CameraCaptureConfig
from .base import Sensor
import omni.replicator.core as rep


class Camera(Sensor):
    def __init__(self, config: CameraCaptureConfig, camera_prim_path="/World/SdgCamera"):
        # super().__init__(**config.__dict__)
        self.config = config
        self.camera, self.render_product, self.basic_writer = self._setup_camera_pipe()

    def _setup_camera_pipe(self):
        horizontal_aperture = 20.955
        focal_length = horizontal_aperture / (
            2 * math.tan(math.radians(self.config.camera_fov / 2))
        )
        
        # Create camera
        camera = rep.create.camera(
            position = (0, 0, 0),
            rotation = (0, 0, 0),
            look_at = (0, 0, 1),
            look_at_up_axis = None,
            focal_length = focal_length,
            focus_distance = 400.0,
            f_stop = 0.0,
            horizontal_aperture = horizontal_aperture,
            horizontal_aperture_offset = 0.0,
            vertical_aperture_offset = 0.0,
            clipping_range = (1.0, 1000000.0),
            projection_type = "pinhole",
            fisheye_nominal_width = 1936.0,
            fisheye_nominal_height = 1216.0,
            fisheye_optical_centre_x = 970.94244,
            fisheye_optical_centre_y = 600.37482,
            openCV_focal_x = 731.78788,
            openCV_focal_y = 731.78789,
            fisheye_max_fov = 200.0,
            fisheye_polynomial_a = 0.0,
            fisheye_polynomial_b = 0.00245,
            fisheye_polynomial_c = 0.0,
            fisheye_polynomial_d = 0.0,
            fisheye_polynomial_e = 0.0,
            fisheye_polynomial_f = 0.0,
            fisheye_p0 = -0.00037,
            fisheye_p1 = -0.00074,
            fisheye_s0 = -0.00058,
            fisheye_s1 = -0.00022,
            fisheye_s2 = 0.00019,
            fisheye_s3 = -0.0002,
            cross_camera_reference_name = None,
            count = 1,
            parent = None,
            name = 'SdgCamera'
        )

        render_product = rep.create.render_product(
            camera,
            (self.config.image_width, self.config.image_height),
        )
        basic_writer = rep.WriterRegistry.get("BasicWriter")

        return camera, render_product, basic_writer
    
    def collect(self):
        if self.config.video_mode:
            self.video_path = os.path.join(Sensor.path_dir, "video")
            os.makedirs(self.video_path, exist_ok=True)
        if self.config.rgb_mode or self.config.depth_mode:
            self.image_path = os.path.join(Sensor.path_dir, "image")
            os.makedirs(self.image_path, exist_ok=True)

        self.basic_writer.initialize(
            output_dir = self.image_path,  # TODO: fix AttributeError when rgb_mode/depth_mode is False
            s3_bucket = None,
            s3_region = None,
            s3_endpoint = None,
            semantic_types = None,
            rgb = self.config.rgb_mode,
            bounding_box_2d_tight = False,
            bounding_box_2d_loose = False,
            semantic_segmentation = False,
            instance_id_segmentation = False,
            instance_segmentation = False,
            distance_to_camera = self.config.depth_mode,
            distance_to_image_plane = False,
            bounding_box_3d = False,
            occlusion = False,
            normals = False,
            motion_vectors = False,
            camera_params = False,
            pointcloud = False,
            pointcloud_include_unlabelled = False,
            image_output_format = "png",
            colorize_semantic_segmentation = True,
            colorize_instance_id_segmentation = True,
            colorize_instance_segmentation = True,
            colorize_depth = self.config.depth_mode,
            skeleton_data = False,
            frame_padding = 4,
            semantic_filter_predicate = None,
            use_common_output_dir = False,
            backend = None,
        )
        self.basic_writer.attach(self.render_product)

        if self.world is not None:
            for _ in range(5):
                self.world.step(render=False)

        height_offset = np.array([0, 0, self.config.camera_height])
        eyes = (Sensor.points[:-1] + height_offset).tolist()
        next_eyes = (Sensor.points[1:] + height_offset).tolist()

        with rep.trigger.on_frame(max_execs=len(eyes)):
            with self.camera:
                rep.modify.pose(
                    position=rep.distribution.sequence(eyes),
                    look_at=rep.distribution.sequence(next_eyes)
                )
                
        rep.orchestrator.run()
        rep.orchestrator.wait_until_complete()
