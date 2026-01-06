import bpy
import numpy as np
import os
import trimesh
import pyVHACD 
from urdf_utils import create_aabb_urdf_snippet

class AssetProcessor:
    def __init__(self, output_dir="output"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def clean_scene(self):
        bpy.ops.wm.read_factory_settings(use_empty=True)


    @staticmethod
    def get_safe_name(obj):
        """
        [核心属性] 统一的文件名清洗逻辑
        将 Blender 内部合法的路径符号 (/) 转换为文件系统合法的符号 (_)
        """
        if obj is None: return "unknown_asset"
        # 替换路径分隔符
        safe_name = obj.name.replace("/", "_").replace("\\", "_")
        return safe_name

    def preprocess_asset(self, obj):
        """
        预处理：清洗 -> 简化 -> 导出 OBJ
        1. Remove footrests/blankets (基于命名或层级).
        2. Simplify mesh (< 10,000 faces).
        """
        print(f"Processing asset: {obj.name}")
        
        # --- 1. 删除特定子部件 ---
        # 遍历子对象，查找名称中包含特定关键词的物体
        objects_to_remove = []
        for child in obj.children_recursive:
            if "footrest" in child.name.lower() or "blanket" in child.name.lower():
                objects_to_remove.append(child)
        
        if objects_to_remove:
            print(f"  - Filtering: Removing {len(objects_to_remove)} unwanted parts...")
            bpy.ops.object.select_all(action='DESELECT')
            for rm_obj in objects_to_remove:
                rm_obj.select_set(True)
            bpy.ops.object.delete() 

        # --- 2. 合并网格 ---
        bpy.ops.object.select_all(action='DESELECT')
        bpy.context.view_layer.objects.active = obj
        obj.select_set(True)
        for child in obj.children_recursive:
            child.select_set(True)
            
        # 只有当有子物体时才执行 join，否则 join 会报错或无效
        if len(obj.children_recursive) > 0:
            print(f"  - Joining {len(obj.children_recursive)} parts into main mesh...")
            bpy.ops.object.join()
        
        # --- 3. 添加厚度 ---
        bpy.context.view_layer.update()
        dims = obj.dimensions
        is_flat = min(dims) < 0.001
        is_floor = any(x in obj.name.lower() for x in ['floor', 'ground', 'terrain'])
        
        if is_flat or is_floor:
            print(f"  - Solidifying: Object '{obj.name}' is too thin or floor. Adding thickness.")
            mod = obj.modifiers.new("Solidify", 'SOLIDIFY')
            mod.thickness = 0.01  # 1cm 厚度，足以让 VHACD 识别
            mod.offset = 0        # 向两侧扩展
            mod.use_even_offset = True
            bpy.ops.object.modifier_apply(modifier="Solidify")
            
        # --- 4. 简化网格 ---
        n_faces = len(obj.data.polygons)
        target_faces = 10000
        if n_faces > target_faces:
            ratio = target_faces / n_faces
            print(f"  - Decimating: {n_faces} faces -> target {target_faces} (ratio {ratio:.2f})")
            mod = obj.modifiers.new("Decimate", 'DECIMATE')
            mod.ratio = ratio
            bpy.ops.object.modifier_apply(modifier="Decimate")
        
        # --- 5. 导出 Visual Mesh (.obj) ---    
        safe_name = self.get_safe_name(obj)
        visual_path = os.path.join(self.output_dir, f"{safe_name}.obj")

        # 仅导出选中的物体（即合并后的 obj）
        try:
            bpy.ops.wm.obj_export(filepath=visual_path, export_selected_objects=True)
        except Exception as e:
            print(f"Error exporting {safe_name}: {e}")
            return None # 返回空以示失败
        return visual_path

    def process_collision(self, obj, visual_path):
        """
        执行 PyVHACD 分解并生成 URDF
        """
        if not visual_path: return
        # --- 提取网格数据 ---
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

        print(f"  - Extracted Mesh: {len(verts)} verts, {len(faces)} faces")
        obj_eval.to_mesh_clear()
        
        # --- 填补网格 ---
        t_mesh = trimesh.Trimesh(vertices=verts, faces=faces)
        if not t_mesh.is_watertight:
            trimesh.repair.fill_holes(t_mesh)
        
        clean_verts = t_mesh.vertices.astype(np.float64)
        clean_faces = t_mesh.faces.astype(np.uint32)

        print("  - Running VHACD...")
        hulls = pyVHACD.compute_vhacd(clean_verts, clean_faces)
        
        # 生成 URDF
        safe_name = self.get_safe_name(obj)
        urdf_xml = create_aabb_urdf_snippet(safe_name, hulls, os.path.basename(visual_path))
        urdf_path = os.path.join(self.output_dir, f"{safe_name}.urdf")
        with open(urdf_path, "w") as f:
            f.write(urdf_xml)
            
        print(f"  - Completed: {urdf_path}")

def run_test_scenario():
    processor = AssetProcessor(output_dir="test_output")
    processor.clean_scene()
    
    # --- 构造模拟数据 ---
    # 1. 创建方块模拟沙发
    bpy.ops.mesh.primitive_cube_add(size=2, location=(0,0,1))
    sofa = bpy.context.active_object
    sofa.name = "Sofa_Gen_001/1"
    
    # 2. 创建“脚踏” (应被移除)
    bpy.ops.mesh.primitive_cube_add(size=0.5, location=(2,0,0.25))
    footrest = bpy.context.active_object
    footrest.name = "Sofa_Footrest_Part"
    footrest.parent = sofa
    
    # 3. 增加面数以测试简化逻辑
    bpy.context.view_layer.objects.active = sofa
    bpy.ops.object.modifier_add(type='SUBSURF')
    sofa.modifiers["Subdivision"].levels = 4
    bpy.ops.object.modifier_apply(modifier="Subdivision")
    
    print(f"Test Scene Created: Sofa with {len(sofa.data.polygons)} faces and child '{footrest.name}'")
    
    # --- 执行管线 ---
    # 选中沙发
    bpy.ops.object.select_all(action='DESELECT')
    sofa.select_set(True)
    bpy.context.view_layer.objects.active = sofa
    
    # 运行
    v_path = processor.preprocess_asset(sofa)
    processor.process_collision(sofa, v_path)

if __name__ == "__main__":
    run_test_scenario()