import bpy
import os
import json
import numpy as np
import mathutils
import time
from shapely.geometry import Polygon, Point
from shapely.ops import unary_union

from asset_processor import AssetProcessor
from blender_util import (
    ensure_object_mode,
    deselect_all,
    select_only,
)

class SceneProcessor:
    def __init__(self, output_dir="dataset"):
        self.output_dir = output_dir
        self.assets_dir = os.path.join(output_dir, "assets")
        # 初始化 AssetProcessor
        self.processor = AssetProcessor(output_dir=self.assets_dir)
        
    def identify_static_dynamic(self, obj):
        """Spec 逻辑：区分 Static/Dynamic"""
        name_lower = obj.name.lower()
        if any(k in name_lower for k in ['wall', 'floor', 'ceiling', 'ground']):
            return 'static'
        
        # 计算体积
        bbox = [obj.matrix_world @ mathutils.Vector(corner) for corner in obj.bound_box]
        min_c = np.min(bbox, axis=0)
        max_c = np.max(bbox, axis=0)
        volume = np.prod(max_c - min_c)

        if volume > 0.5:
            return 'static'
        return 'dynamic'

    def get_object_polygon(self, obj):
        """Layout计算辅助"""
        bbox = [obj.matrix_world @ mathutils.Vector(corner) for corner in obj.bound_box]
        points_2d = [(p.x, p.y) for p in bbox]
        return Point(points_2d[0]).convex_hull if len(points_2d) == 1 else Polygon(points_2d).convex_hull

    def sample_agent_position(self, floor_objs, obstacle_objs):
        """Spec: Shapely 布局采样"""
        print("Computing 2D Layout Map...")
        try:
            floors_poly = unary_union([self.get_object_polygon(o) for o in floor_objs])
            if not obstacle_objs:
                walkable = floors_poly
            else:
                obs_poly = unary_union([self.get_object_polygon(o) for o in obstacle_objs])
                walkable = floors_poly.difference(obs_poly)
            
            # Spec: Padding to prevent boundary collisions
            safe_zone = walkable.buffer(-0.25) # 25cm padding
            
            if safe_zone.is_empty:
                return [0, 0, 0.5]

            min_x, min_y, max_x, max_y = safe_zone.bounds
            for _ in range(200): # 尝试200次
                p = Point(np.random.uniform(min_x, max_x), np.random.uniform(min_y, max_y))
                if safe_zone.contains(p):
                    return [p.x, p.y, 0.5]
        except Exception as e:
            print(f"Layout Error: {e}")
        
        return [0, 0, 0]

    def process(self, scene_id="scene_001"):
        print(f"=== Processing Scene: {scene_id} ===")
        t_start = time.time()

        ensure_object_mode()
        deselect_all()

        # 1) 删除 LIGHT/CAMERA
        for o in list(bpy.data.objects):
            if o.type in {'LIGHT', 'CAMERA'}:
                bpy.data.objects.remove(o, do_unlink=True)

        # 3) 准备阶段
        tasks = []
        scene_assets = []
        floor_objs = []
        obstacle_objs = []

        convertible_types = {'MESH', 'CURVE', 'SURFACE', 'META', 'FONT', 'GPENCIL'}
        all_root_objects = [
            o for o in bpy.context.scene.objects
            if o.parent is None and o.type in convertible_types
        ]
        total_objs = len(all_root_objects)
        print(f"Phase 1: Preparing {total_objs} assets in Blender...")
        
        for idx, obj in enumerate(all_root_objects):
            # 记录用于 Layout 的原始信息
            obj_type = self.identify_static_dynamic(obj)
            if 'floor' in obj.name.lower():
                floor_objs.append(obj)
            else:
                obstacle_objs.append(obj)

            pos = obj.matrix_world.translation.copy()
            rot = obj.matrix_world.to_euler().copy()
            scale = obj.scale.copy()

            # 复制：显式控制 selected/active，避免后续 ops 报 context 错
            subtree = [obj] + list(obj.children_recursive)
            select_only(subtree, active=obj)

            if not bpy.ops.object.duplicate.poll():
                print(f"  [WARN] duplicate.poll() failed for {obj.name}, skipping.")
                deselect_all()
                continue

            bpy.ops.object.duplicate()
            dup_root = bpy.context.view_layer.objects.active
            if not dup_root:
                deselect_all()
                continue

            dup_root.matrix_world = mathutils.Matrix.Identity(4)

            task_data = self.processor.preprocess(dup_root, asset_type=obj_type)

            if task_data:
                tasks.append(task_data)
                safe_name = task_data[0]
                scene_assets.append({
                    "name": obj.name,
                    "urdf_path": f"assets/{safe_name}.urdf",
                    "type": obj_type,
                    "position": [pos.x, pos.y, pos.z],
                    "rotation": [rot.x, rot.y, rot.z],
                    "scale": [scale.x, scale.y, scale.z],
                })

            # 删除复制体：选中当前重复出来的所有对象再删，避免删错
            try:
                dup_selection = list(bpy.context.selected_objects)
                select_only(dup_selection, active=dup_root)
                if bpy.ops.object.delete.poll():
                    bpy.ops.object.delete()
                else:
                    # 兜底：直接 remove 所有选中对象
                    for o in dup_selection:
                        if o and o.name in bpy.data.objects:
                            bpy.data.objects.remove(o, do_unlink=True)
            finally:
                deselect_all()

            if idx % 20 == 0:
                print(f"  Prepared {idx}/{total_objs}...")

        # 3. 计算阶段 (Parallel Pool)
        # ---------------------------
        print(f"Phase 2: Executing physics generation for {len(tasks)} items...")
        self.processor.process_tasks(tasks)
        
        # 4. 布局与元数据保存
        # ---------------------------
        agent_pos = self.sample_agent_position(floor_objs, obstacle_objs)
        
        metadata = {
            "scene_id": scene_id,
            "assets": scene_assets,
            "agent_start_pos": agent_pos
        }
        
        json_path = os.path.join(self.output_dir, f"{scene_id}.json")
        with open(json_path, 'w') as f:
            json.dump(metadata, f, indent=2)
            
        print(f"Scene Complete. Time: {time.time()-t_start:.2f}s. Saved to {json_path}")

if __name__ == "__main__":
    sp = SceneProcessor(output_dir="dataset_run")
    sp.process(scene_id="scene_demo_01")