# Transformer & Mamba 架构可视化项目

基于数学原理的深度学习架构分析工具集。通过动画和交互实验，帮助你理解 Transformer 和 Mamba (SSM) 架构的核心机制。

配套学习笔记：[beginML - Self-Attention](https://github.com/just-for-dream-0x10/beginML/tree/master/other/Self-Attention)

## 功能特性

### 1. 动画模块
使用 Manim 生成的数学原理动画，涵盖以下主题：

**Transformer 核心组件**
- 编码器流程：多头拆分、自注意力计算、残差连接
- 解码器：因果掩码机制、自回归生成
- 交叉注意力：编码器-解码器交互机制
- 多头注意力：多头注意力权重分布与计算
- 位置编码：正弦位置编码原理
- 残差与归一化：残差连接与层归一化
- FFN 与 SwiGLU：前馈网络与门控激活函数

**Mamba 状态空间模型**
- Mamba 选择机制：动态参数生成与状态更新
- 离散化：连续系统到离散递推的转换
- Transformer vs Mamba：架构对比与复杂度分析

**训练与优化技术**
- AdamW 优化器：解耦权重衰减机制
- BPE 分词：字节对编码算法原理
- 混合精度训练：FP16/BF16 训练策略
- RoPE 旋转位置编码：相对位置编码机制
- 训练损失函数：交叉熵与梯度优化

### 2. 交互实验平台
基于 Streamlit 的 Web 应用，提供以下交互功能：

**参数调优实验室**
- 文本输入与分词
- Transformer 参数配置：嵌入维度、注意力头数
- Mamba 参数设置：状态维度、选择性参数
- 训练超参数：学习率、温度采样

**数学计算可视化**
- 完整的 Attention 计算过程
- Softmax 温度调节演示
- 位置编码热力图可视化
- 多头注意力权重分布
- FFN 维度变换过程

**模型行为分析**
- 注意力模式分析（局部/全局/因果）
- 逐层特征演化追踪
- 参数量对比
- 计算复杂度对比（Transformer vs Mamba）

### 3. 训练优化实验
- 学习率调度对比（Warmup + Cosine Decay、Step Decay、Exponential Decay）
- 优化器性能对比（SGD、Adam、AdamW、Lion）
- 混合精度训练效果分析
- 损失函数行为分析与过拟合检测

### 4. 高级分析页面

**Transformer 架构分析** (`pages/transformer_analysis.py`)
- 模型参数分布分析
- 计算复杂度（FLOPs）分解
- GPU 显存占用分析（训练与推理）
- 性能热点识别
- 实时梯度流分析与残差连接对比

**Mamba 模型分析** (`pages/mamba_analysis.py`)
- Mamba/SSM 架构深度分析
- Mamba vs Transformer 性能对比
- 序列长度扩展性分析
- 选择性扫描机制可视化

**Token 旅程追踪** (`pages/token_journey.py`)
- 追踪 Token 在 Transformer 各层的变换过程
- 残差连接分析：修正 vs 替换
- 逐层注意力权重分析
- Logits 到概率的转换可视化

**Kimi 注意力残差 (AttnRes)** (`pages/kimi_attnres.py`)
- PreNorm 稀释问题分析
- Full AttnRes 核心机制与深度方向注意力
- Block AttnRes：工程友好的分块变体
- 三方对比：标准残差 vs Full AttnRes vs Block AttnRes

**训练监控** (`pages/training_monitor.py`)
- 实时训练曲线监控
- 逐层健康状态追踪
- 训练异常检测
- 训练报告生成

**权重分析** (`pages/weight_analysis.py`)
- 权重分布分析
- 权重异常检测（离群值、死权重）
- 权重演化模拟
- 层间权重相关性分析

**初始化对比** (`pages/init_comparison.py`)
- 初始化方法对比（Xavier、Kaiming、Orthogonal 等）
- 逐层激活值演化
- 梯度流对比
- 梯度健康度评估

**梯度流分析** (`pages/gradient_flow.py`)
- 深度网络中的梯度传播可视化
- 激活函数对比（ReLU、Tanh、Sigmoid）
- 残差连接效果分析
- 梯度流报告生成

**注意力模式** (`pages/attention_patterns.py`)
- 注意力热力图可视化
- 多头注意力模式分析
- 头多样性分析
- 头间相似度矩阵

**架构演进** (`pages/architecture_evolution.py`)
- 从 RNN 到 Transformer 到 Mamba 的时间线
- 计算复杂度对比
- 关键特性演进追踪
- 架构能力雷达图

## 安装与使用

### 环境要求
- Python >= 3.10
- FFmpeg（用于 Manim 视频渲染）

### 安装步骤

1. **安装依赖**
```bash
pip install -r requirement.txt
```

2. **生成动画视频**（可选）
```bash
chmod +x generate_all_videos.sh

./generate_all_videos.sh
```

3. **启动交互应用**
```bash
streamlit run app.py
```

### 项目结构
```
Transformer_Explorer/
├── app.py                          # Streamlit 主应用
├── pages/                          # Streamlit 多页面应用
│   ├── transformer_analysis.py      # Transformer 架构分析
│   ├── mamba_analysis.py            # Mamba/SSM 模型分析
│   ├── token_journey.py            # Token 旅程追踪
│   ├── kimi_attnres.py             # Kimi 注意力残差分析
│   ├── training_monitor.py         # 训练动态监控
│   ├── weight_analysis.py          # 模型权重分析
│   ├── init_comparison.py          # 初始化方法对比
│   ├── gradient_flow.py            # 梯度流可视化
│   ├── attention_patterns.py       # 注意力模式可视化
│   └── architecture_evolution.py   # 架构演进时间线
├── utils/                          # 工具模块
│   ├── model_profiler.py           # Transformer 模型分析器
│   ├── mamba_profiler.py           # Mamba 模型分析器
│   ├── token_tracker.py            # Token 旅程追踪器
│   ├── attnres_tracker.py          # 注意力残差追踪器
│   ├── training_monitor.py         # 训练监控器
│   ├── weight_analyzer.py          # 权重分析器
│   ├── initialization_comparator.py # 初始化方法对比器
│   ├── gradient_flow_visualizer.py # 梯度流可视化器
│   ├── gradient_tracker.py         # 梯度追踪器
│   ├── attention_visualizer.py     # 注意力可视化器
│   ├── architecture_evolution.py   # 架构演进时间线
│   ├── base_models.py              # 基础模型定义
│   └── interactive_tuner.py        # 交互式参数调优器
├── tests/                          # 单元测试
│   ├── test_model_profiler.py
│   ├── test_mamba_profiler.py
│   ├── test_token_tracker.py
│   ├── test_attnres_tracker.py
│   ├── test_training_monitor.py
│   ├── test_weight_analyzer.py
│   ├── test_init_comparator.py
│   ├── test_gradient_flow_visualizer.py
│   ├── test_gradient_tracker.py
│   └── test_attention_visualizer.py
├── scene/                          # Manim 动画脚本
│   ├── scene_struct.py             # Transformer 架构
│   ├── scene_mamba_core.py         # Mamba 机制
│   └── scene_*.py                  # 其他主题动画
├── assets/                         # 视频资源文件
├── media/                          # Manim 输出目录
├── requirement.txt                 # 项目依赖
├── Readme.md                       # 英文 README
└── readme_zh.md                    # 中文 README（本文件）
```

## 技术栈

- **动画引擎**：Manim - 数学动画生成工具
- **交互框架**：Streamlit - Web 应用框架
- **数值计算**：NumPy、PyTorch
- **数据可视化**：Plotly、Pandas

## 测试

使用 pytest 运行测试套件：

```bash
python -m pytest tests/
```

## 参考文献

- "Attention Is All You Need" (Vaswani et al., 2017)
- "Mamba: Linear-Time Sequence Modeling with Selective State Spaces" (Gu & Dao, 2023)
- "Rotary Position Embedding" (Su et al., 2021)
- "AdamW: Decoupled Weight Decay Regularization" (Loshchilov & Hutter, 2019)
- "Attention Residuals" (arXiv:2603.15031) - Moonshot AI (Kimi)
