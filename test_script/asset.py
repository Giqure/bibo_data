import bpy
import pyVHACD
import numpy as np
import time

def verify_blender_vhacd_bridge():
    print("-" * 30)
    print("Starting Blender-VHACD Bridge Test")
    
    # 1. 准备场景：清空默认对象，创建一个复杂的几何体（如环面 Torus）
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.mesh.primitive_torus_add(
        major_radius=1.5, minor_radius=0.25, 
        major_segments=48, minor_segments=12, 
        location=(0, 0, 0)
    )
    obj = bpy.context.active_object
    print(f"[Step 1] Created Object: {obj.name}")
    
    # 2. 数据提取：从 Blender Mesh 提取顶点和面数据到 Numpy
    # 必须确保应用了变换（虽然新建物体通常是归一化的，但生产环境需鲁棒）
    depsgraph = bpy.context.evaluated_depsgraph_get()
    obj_eval = obj.evaluated_get(depsgraph)
    mesh = obj_eval.to_mesh()
    
    n_verts = len(mesh.vertices)    
    verts = np.zeros(n_verts * 3, dtype=np.float32)
    mesh.vertices.foreach_get("co", verts)
    verts = verts.reshape((-1, 3)).astype(np.float64) # pyVHACD 需要 double 精度
    
    # 提取面索引（VHACD 需要三角面，Torus 默认生成四边形，需三角化或处理）
    mesh.calc_loop_triangles()
    n_tris = len(mesh.loop_triangles)
    tri_indices = np.zeros(n_tris * 3, dtype=np.int32)
    mesh.loop_triangles.foreach_get("vertices", tri_indices)
    faces = tri_indices.reshape((-1, 3)).astype(np.uint32)

    print(f"[Step 2] Extracted Data: {len(verts)} verts, {len(faces)} faces")
    
    # 3. 核心计算：调用 VHACD
    t0 = time.time()
    # resolution=100000 maxConvexHulls=32 原文
    res = pyVHACD.compute_vhacd(verts, faces)
    t1 = time.time()
    
    print(f"[Step 3] VHACD Complete in {t1-t0:.4f}s")
    print(f"         Generated {len(res)} convex hulls")
    
    # 4. 结果验证：简单的体积/数量校验
    if len(res) > 1:
        print("SUCCESS: Complex geometry successfully decomposed.")
    else:
        print("WARNING: Geometry might be too simple or decomposition failed.")

    # 清理
    obj_eval.to_mesh_clear()

if __name__ == "__main__":
    try:
        verify_blender_vhacd_bridge()
    except Exception as e:
        print(f"CRITICAL ERROR: {e}")
        import traceback
        traceback.print_exc()