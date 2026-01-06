"""
verify_scene.py
功能：加载生成的 Scene JSON 和 URDF 到 IsaacGym，验证物理稳定性。
前置条件：已安装 isaacgym python 包。
"""

import json
import os
import numpy as np
from isaacgym import gymapi, gymutil
from scipy.spatial.transform import Rotation as R

# 初始化参数
args = gymutil.parse_arguments(description="Verify Generated Scene")
# 强制使用图形界面
args.graphics_device_id = 0 
args.compute_device_id = 0

def euler_to_quat(euler_xyz):
    """将 Blender 的 Euler (XYZ) 转换为 IsaacGym 的 Quaternion (x, y, z, w)"""
    # Blender 导出的是弧度还是角度需确认，通常 Blender Python API 给出的是弧度
    # 假设 JSON 中存储的是弧度 (根据 scene_processor.py 逻辑)
    r = R.from_euler('xyz', euler_xyz, degrees=False)
    q = r.as_quat()
    return gymapi.Quat(q[0], q[1], q[2], q[3])

def verify(scene_path):
    # 1. 初始化 IsaacGym
    gym = gymapi.acquire_gym()
    
    # 配置仿真参数
    sim_params = gymapi.SimParams()
    sim_params.dt = 1.0 / 60.0
    sim_params.substeps = 2
    sim_params.up_axis = gymapi.UP_AXIS_Z
    sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.8)
    
    # 物理引擎选择 (PhysX)
    sim_params.physx.use_gpu = True
    sim_params.physx.solver_type = 1
    sim_params.physx.num_position_iterations = 6
    sim_params.physx.num_velocity_iterations = 1
    sim_params.physx.contact_offset = 0.01
    sim_params.physx.rest_offset = 0.0
    
    sim = gym.create_sim(args.compute_device_id, args.graphics_device_id, gymapi.SIM_PHYSX, sim_params)
    if sim is None:
        print("Failed to create sim")
        return

    # 创建地面
    plane_params = gymapi.PlaneParams()
    plane_params.normal = gymapi.Vec3(0, 0, 1)
    gym.add_ground(sim, plane_params)

    # 创建 Viewer
    viewer = gym.create_viewer(sim, gymapi.CameraProperties())
    if viewer is None:
        print("Failed to create viewer")
        return

    # 2. 加载场景元数据
    if not os.path.exists(scene_path):
        print(f"Scene JSON not found: {scene_path}")
        return
        
    base_dir = os.path.dirname(scene_path)
    with open(scene_path, 'r') as f:
        metadata = json.load(f)
        
    print(f"Loading Scene: {metadata.get('scene_id', 'Unknown')}")
    
    # 创建环境 (Env)
    # IsaacGym 需要先创建 Asset，再在 Env 中创建 Actor
    # 这里简化为一个 Env
    spacing = 20.0
    lower = gymapi.Vec3(-spacing, -spacing, 0.0)
    upper = gymapi.Vec3(spacing, spacing, spacing)
    env = gym.create_env(sim, lower, upper, 1)

    # 3. 加载资产
    asset_cache = {} # 避免重复加载相同 URDF
    
    for item in metadata['assets']:
        name = item['name']
        urdf_rel_path = item['urdf_path']
        full_urdf_path = os.path.join(base_dir, urdf_rel_path)
        
        # 资产配置
        asset_options = gymapi.AssetOptions()
        if item['type'] == 'static':
            asset_options.fix_base_link = True
            asset_options.disable_gravity = True
        else:
            asset_options.fix_base_link = False
            asset_options.density = 50.0 # 对应生成脚本中的设定
            
        # 加载 URDF (缓存机制)
        if full_urdf_path not in asset_cache:
            if not os.path.exists(full_urdf_path):
                print(f"[WARN] URDF missing: {full_urdf_path}")
                continue
            root = os.path.dirname(full_urdf_path)
            file = os.path.basename(full_urdf_path)
            asset = gym.load_asset(sim, root, file, asset_options)
            asset_cache[full_urdf_path] = asset
        
        asset = asset_cache[full_urdf_path]
        
        # 设置位姿
        pose = gymapi.Transform()
        p = item['position']
        pose.p = gymapi.Vec3(p[0], p[1], p[2])
        
        # 旋转转换
        r_euler = item['rotation']
        pose.r = euler_to_quat(r_euler)
        
        # 创建 Actor
        actor_handle = gym.create_actor(env, asset, pose, name, 0, 1)
        
        # 设置缩放 (如果 URDF 不支持动态缩放，这里可能无效，需依赖 Blender 里的 Apply Scale)
        # IsaacGym 的 scale 通常在 load_asset 时不支持动态调整，除非使用 soft body 或特定 API
        # 这里的生成脚本已经在 Blender 里 Apply Scale 了，所以 URDF 内部已经是正确尺寸
        pass

    # 4. 标记 Agent 起始位置
    agent_pos = metadata.get('agent_start_pos', [0,0,0])
    print(f"Agent Start Position: {agent_pos}")
    
    sphere_asset = gym.create_sphere(sim, 0.2, gymapi.AssetOptions())
    agent_pose = gymapi.Transform()
    agent_pose.p = gymapi.Vec3(agent_pos[0], agent_pos[1], agent_pos[2])
    gym.create_actor(env, sphere_asset, agent_pose, "agent_marker", 0, 0)
    # 将 Agent 标记为红色
    gym.set_rigid_body_color(env, 0, 0, gymapi.MESH_VISUAL_AND_COLLISION, gymapi.Vec3(1, 0, 0))

    # 5. 运行仿真循环
    # 开启碰撞体显示，检查 AABB 是否过于粗糙
    gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_C, "toggle_collision")
    show_collisions = True
    
    print("Simulation Started. Press 'C' to toggle collision visualization.")
    
    while not gym.query_viewer_has_closed(viewer):
        # 步进物理
        gym.simulate(sim)
        gym.fetch_results(sim, True)
        
        # 渲染
        gym.step_graphics(sim)
        
        if show_collisions:
            gym.clear_lines(viewer)
            # 这是一个 Debug 渲染，实际使用中可能需要更复杂的 draw_lines 逻辑
            # 但 IsaacGym Viewer 自带 "Render -> Collision Shapes" 选项
            pass 

        gym.draw_viewer(viewer, sim, True)
        gym.sync_frame_time(sim)

    gym.destroy_viewer(viewer)
    gym.destroy_sim(sim)

if __name__ == "__main__":
    # 假设测试第一个生成的场景
    target_scene = "Bibo_Dataset/processed/scene_0000/scene_0000.json"
    verify(target_scene)