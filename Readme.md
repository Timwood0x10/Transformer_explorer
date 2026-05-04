# Transformer & Mamba Architecture Visualization Project

A math-principled deep learning architecture analysis toolkit. Uses animations and interactive experiments to help you understand the core mechanisms of Transformer and Mamba (SSM) architectures.

Companion study notes: [beginML - Self-Attention](https://github.com/just-for-dream-0x10/beginML/tree/master/other/Self-Attention)

## Features

### 1. Animation Module
Math-principle animations generated with Manim, covering the following topics:

**Transformer Core Components**
- Encoder flow: multi-head splitting, self-attention computation, residual connections
- Decoder: causal masking mechanism, autoregressive generation
- Cross Attention: encoder-decoder interaction mechanism
- Multi-Head Attention: multi-head attention weight distribution and computation
- Positional Encoding: sinusoidal positional encoding principles
- Residual & Norm: residual connections and layer normalization
- FFN & SwiGLU: feed-forward networks and gated activation functions

**Mamba State Space Model**
- Mamba selective mechanism: dynamic parameter generation and state updates
- Discretization: continuous system to discrete recursion conversion
- Transformer vs Mamba: architecture comparison and complexity analysis

**Training & Optimization Techniques**
- AdamW optimizer: decoupled weight decay mechanism
- BPE tokenization: byte pair encoding algorithm principles
- Mixed precision training: FP16/BF16 training strategies
- RoPE rotary positional encoding: relative positional encoding mechanism
- Training loss functions: cross-entropy and gradient optimization

### 2. Interactive Experiment Platform
A Streamlit-based web application providing the following interactive features:

**Parameter Tuning Lab**
- Text input and tokenization
- Transformer parameter configuration: embedding dimension, number of heads
- Mamba parameter settings: state dimension, selective parameters
- Training hyperparameters: learning rate, temperature sampling

**Mathematical Computation Visualization**
- Complete Attention computation process
- Softmax temperature adjustment demonstration
- Positional encoding heatmap visualization
- Multi-head attention weight distribution
- FFN dimension transformation process

**Model Behavior Analysis**
- Attention pattern analysis (local/global/causal)
- Layer-wise feature evolution tracking
- Parameter count comparison
- Computational complexity comparison (Transformer vs Mamba)

### 3. Training Optimization Experiments
- Learning rate schedule comparison (Warmup + Cosine Decay, Step Decay, Exponential Decay)
- Optimizer performance comparison (SGD, Adam, AdamW, Lion)
- Mixed precision training effect analysis
- Loss function behavior analysis and overfitting detection

### 4. Advanced Analysis Pages

**Transformer Architecture Analysis** (`pages/transformer_analysis.py`)
- Model parameter distribution analysis
- Computational complexity (FLOPs) breakdown
- GPU memory usage analysis (training & inference)
- Performance hotspot identification
- Real-time gradient flow analysis with residual connection comparison

**Mamba Model Analysis** (`pages/mamba_analysis.py`)
- Mamba/SSM architecture deep analysis
- Mamba vs Transformer performance comparison
- Scalability analysis across sequence lengths
- Selective scan mechanism visualization

**Token Journey Tracker** (`pages/token_journey.py`)
- Track a token's transformation through each Transformer layer
- Residual connection analysis: correction vs replacement
- Attention weight analysis across layers
- Logits to probability conversion visualization

**Kimi Attention Residuals (AttnRes)** (`pages/kimi_attnres.py`)
- PreNorm dilution problem analysis
- Full AttnRes core mechanism with depth-direction attention
- Block AttnRes: engineering-friendly chunked variant
- Three-way comparison: Standard Residual vs Full AttnRes vs Block AttnRes

**Training Monitor** (`pages/training_monitor.py`)
- Real-time training curve monitoring
- Layer-wise health status tracking
- Training anomaly detection
- Training report generation

**Weight Analysis** (`pages/weight_analysis.py`)
- Weight distribution analysis
- Weight anomaly detection (outliers, dead weights)
- Weight evolution simulation
- Inter-layer weight correlation analysis

**Initialization Comparison** (`pages/init_comparison.py`)
- Comparison of initialization methods (Xavier, Kaiming, Orthogonal, etc.)
- Activation value evolution across layers
- Gradient flow comparison
- Gradient health assessment

**Gradient Flow Analysis** (`pages/gradient_flow.py`)
- Gradient propagation visualization in deep networks
- Activation function comparison (ReLU, Tanh, Sigmoid)
- Residual connection effect analysis
- Gradient flow report generation

**Attention Patterns** (`pages/attention_patterns.py`)
- Attention heatmap visualization
- Multi-head attention pattern analysis
- Head diversity analysis
- Inter-head similarity matrix

**Architecture Evolution** (`pages/architecture_evolution.py`)
- Timeline from RNN to Transformer to Mamba
- Computational complexity comparison
- Key feature evolution tracking
- Architecture capability radar chart

## Installation & Usage

### Requirements
- Python >= 3.10
- FFmpeg (for Manim video rendering)

### Setup

1. **Install dependencies**
```bash
pip install -r requirement.txt
```

2. **Generate animation videos** (optional)
```bash
chmod +x generate_all_videos.sh

./generate_all_videos.sh
```

3. **Launch the interactive app**
```bash
streamlit run app.py
```

### Project Structure
```
Transformer_Explorer/
├── app.py                          # Streamlit main application
├── pages/                          # Streamlit multi-page app
│   ├── transformer_analysis.py      # Transformer architecture analysis
│   ├── mamba_analysis.py            # Mamba/SSM model analysis
│   ├── token_journey.py            # Token journey tracker
│   ├── kimi_attnres.py             # Kimi Attention Residuals analysis
│   ├── training_monitor.py         # Training dynamics monitor
│   ├── weight_analysis.py          # Model weight analysis
│   ├── init_comparison.py          # Initialization method comparison
│   ├── gradient_flow.py            # Gradient flow visualization
│   ├── attention_patterns.py       # Attention pattern visualization
│   └── architecture_evolution.py   # Architecture evolution timeline
├── utils/                          # Utility modules
│   ├── i18n.py                     # Internationalization (i18n) module
│   ├── language_switcher.py        # Language switcher component
│   ├── model_profiler.py           # Transformer model profiler
│   ├── mamba_profiler.py           # Mamba model profiler
│   ├── token_tracker.py            # Token journey tracker
│   ├── attnres_tracker.py          # Attention Residuals tracker
│   ├── training_monitor.py         # Training monitor
│   ├── weight_analyzer.py          # Weight analyzer
│   ├── initialization_comparator.py # Initialization comparator
│   ├── gradient_flow_visualizer.py # Gradient flow visualizer
│   ├── gradient_tracker.py         # Gradient tracker
│   ├── attention_visualizer.py     # Attention visualizer
│   ├── architecture_evolution.py   # Architecture evolution timeline
│   ├── base_models.py              # Base model definitions
│   └── interactive_tuner.py        # Interactive parameter tuner
├── tests/                          # Unit tests
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
├── scene/                          # Manim animation scripts
│   ├── scene_struct.py             # Transformer architecture
│   ├── scene_mamba_core.py         # Mamba mechanism
│   └── scene_*.py                  # Other topic animations
├── assets/                         # Video resource files
├── media/                          # Manim output directory
├── requirement.txt                 # Project dependencies
├── Readme.md                       # English README (this file)
└── readme_zh.md                    # Chinese README
```

## Tech Stack

- **Animation Engine**: Manim - math animation generation tool
- **Interactive Framework**: Streamlit - web application framework
- **Numerical Computing**: NumPy, PyTorch
- **Data Visualization**: Plotly, Pandas

## Testing

Run the test suite with pytest:

```bash
python -m pytest tests/
```

## References

- "Attention Is All You Need" (Vaswani et al., 2017)
- "Mamba: Linear-Time Sequence Modeling with Selective State Spaces" (Gu & Dao, 2023)
- "Rotary Position Embedding" (Su et al., 2021)
- "AdamW: Decoupled Weight Decay Regularization" (Loshchilov & Hutter, 2019)
- "Attention Residuals" (arXiv:2603.15031) - Moonshot AI (Kimi)
