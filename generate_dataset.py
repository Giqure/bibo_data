import bpy
import os
import time
import sys
import subprocess
import gc
from scene_processor import SceneProcessor

# 确保能导入同级目录的模块
sys.path.append(os.getcwd())

# ================= 配置区域 =================
INFINIGEN_REPO_PATH = "/home/jwm/infinigen"

PYTHON_EXE = sys.executable # 使用当前 Conda 环境的 Python
# ===========================================

def find_scene(root_dir):
    """
    鲁棒的文件查找逻辑：
    InfiniGen 有时会输出到 output/scene.blend，有时是 output/seed/scene.blend
    """
    # 1. 直接检查根目录
    direct = os.path.join(root_dir, "scene.blend")
    if os.path.exists(direct):
        return direct
    
    # 2. 递归查找 (深度限制为 3 以防性能损耗)
    for root, dirs, files in os.walk(root_dir):
        if "scene.blend" in files:
            return os.path.join(root, "scene.blend")
        # 防止遍历太深
        if root.count(os.sep) - root_dir.count(os.sep) > 2:
            del dirs[:] 
            
    return None

def generate(num_scenes=1, output_dir="Bibo_Dataset"):
    start_time = time.time()
    
    # 1. 准备目录
    abs_output_dir = os.path.abspath(output_dir)
    abs_tmp_dir = os.path.join(abs_output_dir, "tmp")
    abs_output_dir = os.path.join(abs_output_dir, output_dir) # 存放最终 Dataset
    
    os.makedirs(abs_tmp_dir, exist_ok=True)
    os.makedirs(abs_output_dir, exist_ok=True)
    
    # 2. 主循环
    print(f"=== Starting Generation ===")
    print(f"Output Root: {abs_output_dir}")
    
    for i in range(num_scenes):
        scene_id = f"scene_{i:04d}"
        seed = i # 固定种子范围便于复现
        scene_id_seed = f"scene_{seed}"
        
        print(f"\n[{i+1}/{num_scenes}] Processing {scene_id} (Seed: {seed}) ...")
        
        # 构造当前seed的场景目录
        abs_scene_dir = os.path.join(abs_tmp_dir, scene_id_seed)
        os.makedirs(abs_scene_dir, exist_ok=True)

        # ========================================
        # 阶段 A: 生成场景
        # ========================================
        
        # 检查是否已经生成
        blend_path = find_scene(abs_scene_dir)
        
        if not blend_path:
            print(f"  > Launching InfiniGen subprocess...")
            
            # 构造生成命令
            # --task coarse: 仅生成粗糙几何，速度快，适合物理碰撞体生成
            cmd = [
                PYTHON_EXE, "-m", "infinigen_examples.generate_indoors",
                "--seed", str(seed),
                "--task", "coarse", 
                "--output_folder", abs_scene_dir,
                "-g", "fast_solve.gin", "singleroom.gin", "no_assets.gin", 
                "-p", "compose_indoors.terrain_enabled=False"
            ]
            
            # 环境变量注入：确保 PYTHONPATH 包含 infinigen 仓库
            env = os.environ.copy()
            env["PYTHONPATH"] = f"{INFINIGEN_REPO_PATH}:{env.get('PYTHONPATH', '')}"
            
            try:
                # 关键：cwd 必须设为仓库根目录，否则找不到 gin 文件
                result = subprocess.run(
                    cmd, 
                    cwd=INFINIGEN_REPO_PATH, 
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )
                
                if result.returncode != 0:
                    print(f"  [ERROR] InfiniGen failed (Code {result.returncode})")
                    print("  >> STDERR Snippet:\n" + "\n".join(result.stderr.splitlines()[-20:]))
                    continue # 跳过此场景
                else:
                    print("  > Generation subprocess finished.")
                    
            except Exception as e:
                print(f"  [ERROR] Subprocess execution error: {e}")
                continue
                
            # 再次查找
            blend_path = find_scene(abs_scene_dir)
        
        if not blend_path:
            print(f"  [ERROR] .blend file missing after generation in {abs_scene_dir}")
            continue

        print(f"  > Found scene file: {blend_path}")

        # ========================================
        # 阶段 B: 场景处理
        # ========================================
        print(f"  > Switching Context & Processing...")
        
        try:
            # 1. 重置当前 Blender 状态、Python垃圾回收
            bpy.ops.wm.read_factory_settings(use_empty=True)
            gc.collect()
            # 2. 加载新场景 (这会替换 bpy.data)
            bpy.ops.wm.open_mainfile(filepath=blend_path)
            
            # 3. 实例化处理器 (注意：输出目录设为 processed/scene_XXXX)
            # 这里的 output_dir 是单个场景的根目录
            scene_output_dir = os.path.join(abs_output_dir, scene_id)
            processor = SceneProcessor(output_dir=scene_output_dir)
            
            # 4. 执行全流程
            processor.process(scene_id=scene_id)
            
            print(f"  > [SUCCESS] {scene_id} Completed.")
            gc.collect()
            
        except Exception as e:
            print(f"  [Processing Error] {e}")
            import traceback
            traceback.print_exc()
            # 不中断循环，继续下一个
            continue

    total_time = time.time() - start_time
    print(f"\n=== Pipeline Finished ===")
    print(f"Total Time: {total_time:.2f}s")
    print(f"Dataset: {abs_output_dir}")

if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    generate(num_scenes=n)