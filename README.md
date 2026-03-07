# Bibo_data
Bibo是一个VLN工作，原文
[arxiv: 2511.00041](https://arxiv.org/pdf/2511.00041)


# 本项目
本项目是一个VLN 数据合成的工作，由infinigen生成室内场景，并在isaacsim中实现路径的采样和路径的信息采集

TODO: 
当前项目正在开发中...

[x] 导入infinigen场景
[x] 解读infinigen物品和房间信息
[x] 生成导航目标点
[x] 生成导航路径
[-] 在导航路径上做信息采集
[ ] 为路径添加prompt
[ ] 大场景测试/批量生成/并行优化
[ ] 使用infinigen_utils实现replicate
[ ] 使用生成数据训练VLN

## 架构
当infinigen生成的室内场景导出为usdc后，vln_synthesize负责读取场景并生成数据
syner是主pipeline
syn_utils是syner用到的模块，包括
- json 读取infinigen生成的solve_state.json
- models 定义场房间、物品等基本格式
- simulation 封装isaacsim运行仿真时的函数
- capture 路径上图像信息的采集
- points/nav_mesh 使用navmesh导航算法，实现路径采样
- points/possion 使用柏松盘算法，在2维平面内采样

本项目由AI辅助完成，AI原稿收归于archive文件夹

## 运行前
1. 按照 [infinigen](https://github.com/princeton-vl/infinigen/blob/main/docs/Installation.md) 手册[生成场景](https://github.com/princeton-vl/infinigen/blob/main/docs/HelloRoom.md)，[导出为USDC](https://github.com/princeton-vl/infinigen/blob/main/docs/ExportingToSimulators.md)
1. 按照 [Isaac Sim](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/pip_installation.html#installing-isaac-sim) 手册配置python环境，本项目isaac-sim版本 5.1.0. 根据 [Python API](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/python_scripting/core_api_overview.html)手册
    > Important
    >
    > Isaac Sim 5.0.0 has introduced the Core Experimental API: a rewritten implementation of the current Core API designed to be more robust, flexible, and powerful, yet still maintain the core utilities and wrapper concepts.
    >
    > Going forward, it will become the base API used in all Isaac Sim source code. The current Core API will be deprecated and removed in future releases.
    >
    > Therefore, we strongly encourage early adoption and use of the Core Experimental API.

    本项目使用的API符合 [Isaac Sim Python API手册](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/py/index.html)
1. 图像采集中需要ffmepg进行视频采集
1. 本项目在Ubuntu-2204上测试并开发中。infinigen生成的室内场景较大时，需要较大的显存；场景中物品较多时，需要较大的内存。
1. 

## 运行
简易选项
```bash
# Image mode
python vln_synthesize/syner.py --usdc_path <scene.usdc> --output_dir ./output --capture_mode image --headless

# Video mode
python vln_synthesize/syner.py --usdc_path <scene.usdc> --output_dir ./output --capture_mode video --headless
```
完整选项
```bash
python vln_synthesize/syner.py \
    --usdc_path <scene.usdc> \
    --output_dir ./output \
    --image_width 640 \
    --image_height 480 \
    --headless \
    --solve_state <solve_state.json> \
    --agent_radius 25.0 \
    --agent_height 80.0 \
    --max_step_height 5.0 \
    --max_slope 30.0 \
    --capture_mode image \
    --capture_depth \
    --camera_height 1.5 \
    --camera_fov 180.0 \
    --video_fps 30 \
    --video_step 0.05 \
    --max_capture_paths 3
```

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--usdc_path` | USDC场景文件路径 | 必需 |
| `--output_dir` | 输出目录 | 必需 |
| `--image_width` | 渲染图像宽度 | 640 |
| `--image_height` | 渲染图像高度 | 480 |
| `--headless` | 无头模式运行 | - |
| `--solve_state` | solve_state.json路径 | `<usdc_dir>/../solve_state.json` |
| `--agent_radius` | NavMesh代理半径(cm) | 25.0 |
| `--agent_height` | NavMesh代理高度(cm) | 80.0 |
| `--max_step_height` | NavMesh最大步高(cm) | 5.0 |
| `--max_slope` | NavMesh最大可行走斜坡 | 30.0 |
| `--capture_mode` | 采集模式: image/video/无，”无“时生成可视化标记 | None |
| `--capture_depth` | 同步采集深度图(.npy) | False |
| `--camera_height` | 相机高度(m) | 1.5 |
| `--camera_fov` | 水平视场角(°)，fov大时切换为全景图像拼接 | 90.0 |
| `--video_fps` | 视频帧率 | 30 |
| `--video_step` | 视频插值步长(m) | 0.05 |
| `--max_capture_paths` | 采集路径数(0=全部) | 3 |

### 运行时
除了生成场景外，本项目中路径采样的时间最长，测试时40个物品需要大于7秒，4000个物品需要大于7小时，这一部分取决于CPU

### 运行后
注意保存文件，本项目生成时默认覆盖原文件
生成结果位于 <output_dir>/capture ，按路径建立文件夹

每个路径文件夹中包含

| | |
|-|-|
| rgb   | 图像，png格式 | 
| depth | 深度信息，numpy格式 |
| video | 视频 |
| metadata.json |  路径信息 |
| | |
