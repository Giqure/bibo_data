"""该脚本不使用"""

import bpy
import os
import time
import sys
import subprocess
import shutil
from scene_processor import SceneProcessor

# ================= 配置区域 =================
# 指向您 clone 的 infinigen 仓库根目录
INFINIGEN_REPO_PATH = "/home/jwm/infinigen" 
# ===========================================

def run_dataset_generation(num_scenes=1, output_dir="Bibo_Dataset_v1"):
    start_time = time.time()
    
    # 1. 准备目录
    abs_output_dir = os.path.abspath(output_dir)
    temp_gen_dir = os.path.join(output_dir, "temp_gen")
    assets_dir = os.path.join(output_dir, "assets")
    os.makedirs(temp_gen_dir, exist_ok=True)
    os.makedirs(assets_dir, exist_ok=True)
    
    for i in range(num_scenes):
        scene_id = f"scene_{i:04d}"
        seed = i
        
        print(f"\n[{i+1}/{num_scenes}] ==================================================")
        print(f"Generating Geometry for {scene_id} (Seed: {seed})...")
        
        # 为当前场景创建独立的临时输出目录
        current_job_dir = os.path.join(temp_gen_dir, scene_id)
        # 清理旧数据防止混淆
        if os.path.exists(current_job_dir) and not os.path.exists(os.path.join(current_job_dir, "scene.blend")):
            shutil.rmtree(current_job_dir)
        os.makedirs(current_job_dir, exist_ok=True)


        # ========================================
        # 定位生成的 .blend 文件
        # generate_indoors --task coarse 通常直接生成在 output_folder/scene.blend
        blend_path = os.path.join(current_job_dir, "scene.blend")

        misplaced_path = os.path.join(INFINIGEN_REPO_PATH, output_dir, "temp_gen", scene_id, "scene.blend")
        if os.path.exists(misplaced_path) and not os.path.exists(blend_path):
            print(f"  > [RECOVERY] Found misplaced file in repo. Moving it...")
            print(f"    From: {misplaced_path}")
            print(f"    To:   {blend_path}")
            shutil.move(misplaced_path, blend_path)
            # 尝试清理空的遗留目录
            try:
                os.rmdir(os.path.dirname(misplaced_path))
            except: pass

        if not os.path.exists(blend_path):
            # 2. 构造 Infinigen 原子命令
            # 参考 HelloRoom.md: python -m infinigen_examples.generate_indoors --task coarse ...
            cmd = [
                sys.executable, "-m", "infinigen_examples.generate_indoors",
                "--seed", str(seed),
                "--task", "coarse",  # 关键：仅生成几何，跳过渲染
                "--output_folder", current_job_dir,
                "-g", "fast_solve.gin", "singleroom.gin", "no_assets.gin", # 基础配置
                "-p", "compose_indoors.terrain_enabled=False" # 禁用地形(室内场景通常不需要外部地形)
            ]
            
            # 注意：如果您的 Infinigen 安装没有 'no_assets.gin'，请移除它。
            # 'fast_solve.gin' 会牺牲质量换取速度，适合测试。生产时可移除。

            print(f"  > Executing Infinigen Core...")
            
            # 3. 设置环境变量并执行
            env = os.environ.copy()
            # 确保能找到 infinigen_examples
            env["PYTHONPATH"] = f"{INFINIGEN_REPO_PATH}:{env.get('PYTHONPATH', '')}"
            
            try:
                # 必须在 INFINIGEN_REPO_PATH 下运行以解析 gin configs
                result = subprocess.run(
                    cmd, 
                    cwd=INFINIGEN_REPO_PATH, 
                    env=env,
                    capture_output=True, 
                    text=True
                )
                
                if result.returncode != 0:
                    print(f"  [ERROR] Generation failed with code {result.returncode}")
                    # 打印关键错误信息
                    print("  STDERR Tail:", result.stderr[-1000:])
                    continue
                else:
                    print("  > Core generation successful.")
                    
            except Exception as e:
                print(f"  [CRITICAL] Subprocess failed: {e}")
                continue
        else:
            print("  > Using existing scene.blend (Skipped generation)")
        
        if not os.path.exists(blend_path):
            print(f"  [ERROR] Expected .blend not found at {blend_path}")
            print("  Check STDOUT for details if needed.")
            continue
            
        print(f"  > Loading scene into Blender: {blend_path}")
        
        # 5. 重置 Blender 并加载文件
        try:
            bpy.ops.wm.read_factory_settings(use_empty=True)
            bpy.ops.wm.open_mainfile(filepath=blend_path)
        except Exception as e:
            print(f"  [ERROR] Blender load failed: {e}")
            continue

        # 6. 运行后处理管线 (SceneProcessor)
        # 此时 bpy.context 已经是新加载的场景
        # 指定最终输出目录为 Bibo_Dataset_v1/scene_XXXX
        final_scene_dir = os.path.join(output_dir, scene_id)
        processor = SceneProcessor(output_dir=final_scene_dir)
        
        try:
            print(f"  > Starting Physics Processing...")
            processor.process(scene_id=scene_id)
            print(f"  > [SUCCESS] Scene {scene_id} ready.")
        except Exception as e:
            print(f"  [ERROR] Processing failed: {e}")
            import traceback
            traceback.print_exc()

    total_time = time.time() - start_time
    print(f"\n=== Batch Complete ===")
    print(f"Total time: {total_time:.2f}s")
    print(f"Dataset location: {os.path.abspath(output_dir)}")

if __name__ == "__main__":
    # 用法: python generate_dataset.py [num_scenes]
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    run_dataset_generation(num_scenes=n)