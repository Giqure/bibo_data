"""urdf_utils.py"""
"""将 VHACD 凸包转为 AABB 并存入 URDF"""
import numpy as np
import xml.etree.ElementTree as ET
from xml.dom import minidom

def create_aabb_urdf_snippet(object_name, convex_hulls, visual_mesh_path, asset_type='dynamic', volume=0.0):
    """
    根据说明书要求：将凸包列表转换为一组 AABB 碰撞体盒子。
    """
    robot = ET.Element('robot', name=object_name)
    link = ET.SubElement(robot, 'link', name=f"{object_name}_link")

    # 1. 视觉部分 (Visual)：引用原始或简化后的 OBJ
    visual = ET.SubElement(link, 'visual')
    v_origin = ET.SubElement(visual, 'origin', xyz="0 0 0", rpy="0 0 0")
    v_geometry = ET.SubElement(visual, 'geometry')
    v_mesh = ET.SubElement(v_geometry, 'mesh', filename=visual_mesh_path, scale="1 1 1")

    # 2. 碰撞部分 (Collision)：VHACD Hulls -> AABB Boxes
    # 说明书要求：Each convex hull is further abstracted by its AABB.
    # print(f"  [urdf] Processing {len(convex_hulls)} hulls for collision generation...")
    
    for idx, (hull_verts, hull_faces) in enumerate(convex_hulls):
        # 计算该凸包的 AABB
        min_coords = np.min(hull_verts, axis=0)
        max_coords = np.max(hull_verts, axis=0)
        
        # AABB 中心点 (Box 的原点)
        center = (min_coords + max_coords) / 2.0
        # AABB 尺寸 (长宽高)
        extent = max_coords - min_coords
        
        # 过滤掉极小的退化包围盒 (可选，防止物理引擎震荡)
        if np.any(extent < 0.001):
            continue

        collision = ET.SubElement(link, 'collision', name=f"col_{idx}")
        # 设置 Box 的中心位置
        c_origin = ET.SubElement(collision, 'origin', 
                                 xyz=f"{center[0]:.6f} {center[1]:.6f} {center[2]:.6f}", 
                                 rpy="0 0 0")
        c_geometry = ET.SubElement(collision, 'geometry')
        # 设置 Box 的尺寸
        ET.SubElement(c_geometry, 'box', size=f"{extent[0]:.6f} {extent[1]:.6f} {extent[2]:.6f}")

    # 3. 惯性矩阵 (Inertial)：简化处理，假设均匀密度
    # 说明书提及 dynamic assets density ~ 50kg/m^3
    # 此处仅生成占位符，实际需根据体积计算
    inertial = ET.SubElement(link, 'inertial')
    ET.SubElement(inertial, 'origin', xyz="0 0 0", rpy="0 0 0")
    
    if asset_type == 'static':
        # 静态物体在 IsaacGym 中通常通过 fix_base_link=True 处理
        # 但 URDF 中仍需给一个虚拟质量防止加载报错，或者干脆设为极大
        mass_val = 1000.0
    else:
        # 动态物体：Mass = Density * Volume
        density = 50.0 # kg/m^3
        # 如果体积计算失败（如非闭合），给一个默认最小体积 0.001
        safe_volume = max(volume, 0.001) 
        mass_val = density * safe_volume
    
    ET.SubElement(inertial, 'mass', value=f"{mass_val:.4f}")
    # 简化惯性张量 (假设为球体或立方体近似)
    inertia_scale = 0.1 * mass_val
    ET.SubElement(inertial, 'inertia', 
                  ixx=f"{inertia_scale:.4f}", ixy="0", ixz="0", 
                  iyy=f"{inertia_scale:.4f}", iyz="0", 
                  izz=f"{inertia_scale:.4f}")

    # 格式化输出
    raw_str = ET.tostring(robot, 'utf-8')
    parsed = minidom.parseString(raw_str)
    return parsed.toprettyxml(indent="  ")

# 简单测试桩
if __name__ == "__main__":
    # 模拟一个简单的凸包数据 (中心在 1,1,1，大小 2x2x2)
    dummy_hull_verts = np.array([[0,0,0], [2,2,2]], dtype=np.float64)
    dummy_hulls = [(dummy_hull_verts, None)] # faces 不影响 AABB 计算
    
    xml_output = create_aabb_urdf_snippet("test_obj", dummy_hulls, "test_obj.obj")
    print(xml_output)