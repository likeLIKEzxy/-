# Transformer Small — 从零实现的 Transformer 模型

本项目基于 PyTorch 从零实现了一个简化版 **Transformer (Encoder–Decoder)**，  
并在 **IWSLT2017 (德语→英语)** 数据集上进行了训练与消融实验。

---

## 📦 环境要求

- Python >= 3.9  
- PyTorch >= 2.0  
- Transformers  
- Datasets  
- Matplotlib  
- PyYAML  
- tqdm  

安装依赖：
```bash
pip install -r requirements.txt


transformer-small/
├── src/
│   ├── model.py              # Transformer 主体结构（Encoder + Decoder）
│   ├── decoder.py            # Decoder 模块
│   ├── data_iwslt.py         # IWSLT2017 数据加载与处理
│   ├── train.py              # 训练脚本（可从 YAML 读取配置）
│   ├── run_experiments.py    # 批量运行多组实验
│   ├── utils.py              # 工具函数
│   └── configs/              # 各实验配置文件
│       ├── base.yaml
│       ├── no_posenc.yaml
│       ├── no_residual.yaml
│       ├── single_head.yaml
│       ├── small_ffn.yaml
│       ├── lr_1e-4.yaml
│       └── lr_1e-3.yaml
├── data/                     # IWSLT2017 数据文件（本地）
├── results/                  # 实验结果（loss 曲线、模型权重等）
├── requirements.txt          # 依赖文件
├── report.tex                # LaTeX 实验报告
└── README.md                 # 项目说明
单次训练
python -m src.train --config src/configs/base.yaml
批量运行
python run_experiments.py
复现步骤如下
# 1. 克隆仓库
git clone https://github.com/likeLIKEzxy/transformer-small.git
cd transformer-small

# 2. 安装依赖
pip install -r requirements.txt

# 3. 下载数据集到 data/ 目录

# 4. 运行训练
python -m src.train --config src/configs/base.yaml

# 5. 或运行全部实验
python run_experiments.py
