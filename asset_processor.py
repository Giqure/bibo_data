"""asset_processor.py"""
import bpy
import numpy as np
import time
import os
import gc
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from physics_worker import compute

from blender_util import (
    deselect_all,
    select_only,
    delete_objects,
    join_into_active,
    apply_modifier,
    convert_object_to_mesh_object,
)

# --- 主处理器类 ---
class AssetProcessor:
    def __init__(self, output_dir="output", max_workers=None):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        cpu_count = os.cpu_count() or 1
        self.max_workers = max_workers if max_workers else max(1, cpu_count - 2)
        print(f"[Init] FastAssetProcessor initialized with {self.max_workers} worker threads.")

    @staticmethod
    def get_safe_name(obj):
        if obj is None:
            return "unknown"
        return obj.name.replace("/", "_").replace("\\", "_")

    def preprocess(self, obj, asset_type='dynamic'):
        """
        [主线程任务]
        仅执行必须在 Blender 内部完成的操作：清洗、合并、简化、OBJ导出、数据提取。
        返回：用于后台计算的 task_data 元组，或者 None (如果失败)。
        """
        deselect_all()

        depsgraph = bpy.context.evaluated_depsgraph_get()
        obj = convert_object_to_mesh_object(obj, depsgraph)

        if obj is None or obj.type != 'MESH' or obj.data is None:
            return None

        # 1) 清洗：移除不需要的子物体
        objects_to_remove = []
        for child in list(obj.children_recursive):
            nm = child.name.lower()
            if ("footrest" in nm) or ("blanket" in nm):
                objects_to_remove.append(child)

        if objects_to_remove:
            delete_objects(objects_to_remove)

        # 2) 合并：把剩余 children join 到 obj
        children = list(obj.children_recursive)
        if children:
            join_into_active(obj, children)

        # 3) 确保导出/后续操作上下文稳定：Object 模式 + 唯一选中 + active=obj
        select_only([obj], active=obj)

        # 4) 几何修改 (Solidify 加厚 / Decimate 简化)
        bpy.context.view_layer.update()
        dims = obj.dimensions
        is_flat = min(dims) < 0.001
        is_floor = any(x in obj.name.lower() for x in ['floor', 'ground', 'terrain'])

        if is_flat or is_floor:
            mod = obj.modifiers.new("Solidify", 'SOLIDIFY')
            mod.thickness = 0.02
            mod.offset = 0
            mod.use_even_offset = True
            if not apply_modifier(obj, mod.name):
                print(f"[WARN] modifier_apply failed (Solidify) for {obj.name}")
                # 不直接失败，继续尝试后续步骤

        n_faces = len(obj.data.polygons) if obj.data else 0
        target_faces = 10000
        if n_faces > target_faces:
            ratio = target_faces / max(1, n_faces)
            mod = obj.modifiers.new("Decimate", 'DECIMATE')
            mod.ratio = ratio
            if not apply_modifier(obj, mod.name):
                print(f"[WARN] modifier_apply failed (Decimate) for {obj.name}")

        safe_name = self.get_safe_name(obj)

        # 5) 保存处理后的资产 到 OBJ（强制 active+selected）
        visual_filename = f"{safe_name}.obj"
        visual_path = os.path.join(self.output_dir, visual_filename)
        try:
            select_only([obj], active=obj)
            bpy.ops.wm.obj_export(filepath=visual_path, export_selected_objects=True)
        except Exception as e:
            print(f"Blender Export Error {safe_name}: {e}")
            return None

        # 6) 提取数据到 Numpy（不依赖 selection/active）
        depsgraph = bpy.context.evaluated_depsgraph_get()
        obj_eval = obj.evaluated_get(depsgraph)
        mesh = obj_eval.to_mesh()

        n_verts = len(mesh.vertices)
        verts = np.zeros(n_verts * 3, dtype=np.float32)
        mesh.vertices.foreach_get("co", verts)
        verts = verts.reshape((-1, 3)).astype(np.float64)

        mesh.calc_loop_triangles()
        n_tris = len(mesh.loop_triangles)
        tri_indices = np.zeros(n_tris * 3, dtype=np.int32)
        mesh.loop_triangles.foreach_get("vertices", tri_indices)
        faces = tri_indices.reshape((-1, 3)).astype(np.uint32)

        obj_eval.to_mesh_clear()

        npz_path = os.path.join(self.output_dir, f"{safe_name}.npz")
        np.savez_compressed(npz_path, verts=verts, faces=faces)
        return (safe_name, npz_path, visual_filename, self.output_dir, asset_type)

    def process(self, objects_list):
        """
        批量处理入口，只在本文调用，新函数process_tasks现用于多进程池调用。
        """
        total = len(objects_list)
        print(f"--- Starting Batch Processing for {total} assets ---")
        t0 = time.time()
        
        compute_tasks = []
        
        # 阶段 1  Blender 主线程准备: preprocess
        # 这一步必须串行，因为 bpy 不支持并行修改场景
        print("Phase 1: Blender Geometry Preparation ...")
        for idx, obj in enumerate(objects_list):
            if idx % 50 == 0: print(f"  Prepared {idx}/{total}...")
            
            # 隔离对象处理
            # 在实际 SceneProcessor 中，这里应该在 duplicate 之后调用
            task_data = self.preprocess(obj)
            
            if task_data:
                compute_tasks.append(task_data)
        
        t1 = time.time()
        print(f"Phase 1 Complete. Time: {t1-t0:.2f}s. Submitting {len(compute_tasks)} tasks to {self.max_workers} process pool.")

        # 阶段 2  凸包计算并生成 URDF (进程并行，绕过 GIL)
        results = []
        ctx = get_context("fork") # Linux 推荐 "fork"，Windows 必须使用 "spawn"
        with ProcessPoolExecutor(max_workers=self.max_workers, mp_context=ctx) as executor:
            futures = [executor.submit(compute, data) for data in compute_tasks]
            
            for i, future in enumerate(as_completed(futures)):
                res = future.result()
                results.append(res)
                if i % 10 == 0:
                    print(f"  [Worker] Progress: {i+1}/{len(compute_tasks)} - {res}")

        t2 = time.time()
        print(f"--- Batch Complete. Total Time: {t2-t0:.2f}s (Speedup Phase: {t2-t1:.2f}s) ---")
        
        # 阶段 3: 内存治理
        # 处理完一批后强制垃圾回收，防止成千上万个 numpy 数组滞留内存
        gc.collect()

    def process_tasks(self, task_list):
        """
        [多进程入口]
        接收 SceneProcessor 收集好的 task_list，统一并行计算。
        """
        if not task_list:
            return
        
        print(f"--- Launching VHACD Pool for {len(task_list)} tasks ---")
        t0 = time.time()
        
        # Linux 下 fork 启动极快，且能利用 copy-on-write
        ctx = get_context("fork") 
        with ProcessPoolExecutor(max_workers=self.max_workers, mp_context=ctx) as executor:
            # 提交任务
            futures = [executor.submit(compute, task) for task in task_list]
            
            for i, future in enumerate(as_completed(futures)):
                try:
                    res = future.result()
                    if i % 10 == 0: # 减少日志刷屏
                        print(f"  [Worker] {i+1}/{len(task_list)} - {res}")
                except Exception as e:
                    print(f"  [Worker Error] {e}")

        print(f"--- Pool Complete. Time: {time.time()-t0:.2f}s ---")
        gc.collect()

# --- 压力测试脚本 ---
def stress_test():
    # 创建模拟场景：生成 50 个物体进行测试
    bpy.ops.wm.read_factory_settings(use_empty=True)
    
    test_objects = []
    print("Generating 50 random test objects...")
    for i in range(50):
        # 随机生成一些圆环或猴头
        if i % 2 == 0:
            bpy.ops.mesh.primitive_torus_add(location=(i*2, 0, 0))
        else:
            bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=3, location=(i*2, 2, 0))
        
        obj = bpy.context.active_object
        obj.name = f"Asset_{i:04d}"
        test_objects.append(obj)
    
    processor = AssetProcessor(output_dir="fast_output")
    processor.process(test_objects)

if __name__ == "__main__":
    stress_test()