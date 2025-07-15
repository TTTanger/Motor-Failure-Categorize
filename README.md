# 基于TKEO与CNN的电机转子断条故障图像识别方法

## 项目简介
本项目旨在通过TKEO（Teager-Kaiser能量算子）与卷积神经网络（CNN）相结合，实现电机转子断条故障的自动化图像识别。项目包含数据预处理、特征提取、图像生成、模型训练与测试等完整流程。

## 目录结构
```
├── cnn.py                  # CNN模型定义与训练
├── dataset_setting.py      # 数据集标签生成与设置
├── images_generating.py    # 信号转图像的生成脚本
├── pre_process.py          # 原始数据预处理
├── test.py                 # 模型测试脚本
├── requirements.txt        # 依赖包列表
├── model/                  # 训练得到的模型文件
├── dataset/                # 数据集
│   ├── fine_tuning_dataset/
│   ├── pre_training_dataset/
│   └── test/
├── workplace/
│   ├── origin_data/        # 原始数据及说明
│   ├── result/             # 结果图像与标签
│   └── sorted_data/        # 中间处理数据
```

## 环境依赖
请先安装Python 3.8+，并通过如下命令安装依赖：

```bash
pip install -r requirements.txt
```

## 主要功能说明
- **数据预处理**：`pre_process.py` 对原始采集的电流信号进行清洗、分段与采样。
- **特征提取与图像生成**：`images_generating.py` 利用TKEO算子将时序信号转为特征矩阵，并保存为图像。
- **数据集标签生成**：`dataset_setting.py` 根据文件名自动生成normal/abnormal标签。
- **模型训练与测试**：`cnn.py` 定义CNN结构并训练，`test.py` 用于模型评估。

## 数据集说明
- `dataset/fine_tuning_dataset/`、`pre_training_dataset/`、`test/` 下分别存放微调、预训练和测试用的图像及标签。
- 标签文件内容为`normal`或`abnormal`，与图像一一对应。

## 快速开始
1. **数据预处理**
   - 修改`pre_process.py`中的路径，运行以生成标准化CSV。
2. **特征提取与图像生成**
   - 运行`images_generating.py`，将CSV转为图像。
3. **生成标签**
   - 运行`dataset_setting.py`，自动生成标签文件。
4. **模型训练与测试**
   - 运行`cnn.py`进行训练，`test.py`进行评估。

## 参考
- TKEO能量算子原理
- 卷积神经网络（CNN）基础

---
如有问题请联系项目维护者。
