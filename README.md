# Bibo_data
Bibo_data项目是对Bibo项目中数据集构建的复现，原文
[arxiv: 2511.00041](https://arxiv.org/pdf/2511.00041)

# Easy use
python3 ./generate_dataset.py

# 架构简述
原文使用开源项目infinigen构建室内场景(scene)，infinigen会生成一个blend文件。
然后交给scene_processor处理，
 - 删除灯光和摄像机
