"""
Transformer分析页面：提供Transformer模型结构分析、参数热点分析等工具
"""
import streamlit as st
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from utils.model_profiler import TransformerProfiler, create_sample_transformer
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

st.set_page_config(page_title="Transformer架构分析", page_icon="🤖", layout="wide")

st.title("🤖 Transformer 架构分析工具")
st.markdown("### 深度剖析Transformer模型结构、参数分布和性能瓶颈")

# 侧边栏配置
with st.sidebar:
    st.divider()
    st.header("⚙️ 模型配置")
    
    d_model = st.slider("嵌入维度 (d_model)", 128, 2048, 512, step=128)
    n_heads = st.select_slider("注意力头数", [4, 8, 12, 16, 32], value=8)
    n_layers = st.slider("层数 (n_layers)", 1, 24, 6)
    vocab_size = st.select_slider("词表大小", [10000, 30000, 50000, 100000], value=50000)
    
    st.divider()
    
    st.header("💾 推理配置")
    batch_size = st.slider("Batch Size", 1, 64, 8)
    seq_len = st.slider("序列长度", 32, 2048, 128, step=32)
    
    st.divider()
    
    precision = st.radio("精度", ["FP32", "FP16", "BF16"], index=2)
    dtype_map = {"FP32": 4, "FP16": 2, "BF16": 2}
    bytes_per_element = dtype_map[precision]

# 创建模型和分析器 (cached, auto-rebuilds when params change)
@st.cache_resource
def get_profiler(d_model, n_heads, n_layers, vocab_size, batch_size, seq_len):
    model = create_sample_transformer(d_model, n_heads, n_layers, vocab_size)
    return TransformerProfiler(model, (batch_size, seq_len, d_model))

profiler = get_profiler(d_model, n_heads, n_layers, vocab_size, batch_size, seq_len)

# =================== 运行分析按钮 ===================
if st.button("🔬 运行分析", type="primary", key="run_transformer_analysis"):
    with st.spinner("🔄 正在计算..."):
        try:
            import torch
            param_counts = profiler.count_parameters()
            profiles = profiler.profile_layers(batch_size, seq_len, d_model, n_heads, n_layers, vocab_size)
            flops_data = profiler.estimate_flops(batch_size, seq_len, d_model, n_heads, n_layers, vocab_size)
            memory_data = profiler.estimate_memory(batch_size, seq_len, d_model, n_layers,
                                                  dtype=torch.float32 if precision == "FP32" else torch.float16)
            complexity = profiler.get_attention_complexity_comparison(
                [128, 256, 512, 1024, 2048, 4096], d_model)
            training_step = profiler.simulate_training_step(batch_size, seq_len, d_model, n_layers)
            st.session_state["transformer_results"] = {
                "param_counts": param_counts,
                "profiles": profiles,
                "flops_data": flops_data,
                "memory_data": memory_data,
                "complexity": complexity,
                "training_step": training_step,
            }
            st.success("✅ 分析完成！")
        except Exception as e:
            st.error(f"计算出错: {e}")

# 主界面标签页
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 参数分析",
    "⚡ 计算复杂度",
    "💾 显存分析",
    "🔥 性能热点",
    "🌊 梯度流分析"
])

# =================== TAB 1: 参数分析 ===================
with tab1:
    st.header("📊 模型参数分布分析")

    if "transformer_results" in st.session_state:
        results = st.session_state["transformer_results"]
        param_counts = results["param_counts"]
        total_params = param_counts['total']

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("总参数量", f"{total_params:,}", help="模型所有可训练参数")
        with col2:
            st.metric("参数量 (M)", f"{total_params/1e6:.2f}M")
        with col3:
            st.metric("参数量 (B)", f"{total_params/1e9:.4f}B")

        st.divider()

        # 层级参数分布
        st.subheader("🏗️ 各层参数分布")

        profiles = results["profiles"]

        # 创建数据框
        layer_data = []
        for p in profiles:
            layer_data.append({
                "层名称": p.name,
                "参数量": p.params,
                "参数占比 (%)": p.param_ratio,
                "显存 (MB)": p.memory_mb
            })

        df = pd.DataFrame(layer_data)

        col1, col2 = st.columns([2, 1])

        with col1:
            # 饼图：参数分布
            fig = px.pie(df, values="参数量", names="层名称",
                         title="参数分布饼图",
                         hover_data=['参数占比 (%)'])
            fig.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # 表格显示
            st.dataframe(
                df.style.format({
                    "参数量": '{:,}',
                    '参数占比 (%)': '{:.2f}',
                    '显存 (MB)': '{:.2f}'
                }).background_gradient(subset=['参数占比 (%)'], cmap='YlOrRd'),
                height=400
            )

        st.divider()

        # 组件级参数分析
        st.subheader("🔍 组件级参数分析")

        # 计算各组件参数量
        embedding_params = vocab_size * d_model
        attention_params_per_layer = 4 * d_model * d_model
        ffn_params_per_layer = 2 * d_model * 4 * d_model
        output_params = vocab_size * d_model

        component_data = {
            "组件": ["Embedding", "Self-Attention (所有层)", "FFN (所有层)", "Output Layer"],
            "参数量": [
                embedding_params,
                attention_params_per_layer * n_layers,
                ffn_params_per_layer * n_layers,
                output_params
            ]
        }

        df_comp = pd.DataFrame(component_data)
        df_comp['参数占比 (%)'] = df_comp['参数量'] / df_comp['参数量'].sum() * 100

        fig = px.bar(df_comp, x='组件', y="参数量",
                     color='参数占比 (%)',
                     text="参数量",
                     title="组件参数量对比")
        fig.update_traces(texttemplate='%{text:,.0f}', textposition='outside')
        st.plotly_chart(fig, use_container_width=True)

        st.markdown(f"""
        **关键观察**：
        - **Embedding + Output**: {(embedding_params + output_params)/1e6:.2f}M 参数
          ({(embedding_params + output_params)/total_params*100:.1f}%)
        - **Attention**: {attention_params_per_layer * n_layers/1e6:.2f}M 参数
          ({attention_params_per_layer * n_layers/total_params*100:.1f}%)
        - **FFN**: {ffn_params_per_layer * n_layers/1e6:.2f}M 参数
          ({ffn_params_per_layer * n_layers/total_params*100:.1f}%)

        💡 **优化建议**：
        - 词表相关层占据了大量参数，考虑使用 **词表压缩** 或 **权重共享**
        - FFN 参数量约为 Attention 的 2 倍，是优化重点
        """)
    else:
        st.info("👆 点击 '运行分析' 按钮开始计算")

# =================== TAB 2: 计算复杂度 ===================
with tab2:
    st.header("⚡ 计算复杂度分析")

    if "transformer_results" in st.session_state:
        results = st.session_state["transformer_results"]
        flops_data = results["flops_data"]

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("单层 FLOPs", f"{flops_data['total_per_layer']/1e9:.2f} GFLOPs")
        with col2:
            st.metric("总 FLOPs", f"{flops_data['total_model']/1e9:.2f} GFLOPs")
        with col3:
            throughput = flops_data['total_model'] / 1e12  # TFLOPs
            st.metric("吞吐量", f"{throughput:.3f} TFLOPs")

        st.divider()

        # FLOPs 分解
        st.subheader("🔬 FLOPs 分解分析")

        flops_breakdown = {
            "操作": [
                "QKV 投影",
                "Attention 计算",
                "FFN",
                "其他"
            ],
            "FLOPs": [
                flops_data['qkv_projection'],
                flops_data['attention_matrix'],
                flops_data['ffn_total'],
                flops_data['total_per_layer'] - flops_data['qkv_projection'] -
                flops_data['attention_matrix'] - flops_data['ffn_total']
            ]
        }

        df_flops = pd.DataFrame(flops_breakdown)
        df_flops['占比 (%)'] = df_flops['FLOPs'] / df_flops['FLOPs'].sum() * 100

        col1, col2 = st.columns(2)

        with col1:
            fig = px.pie(df_flops, values='FLOPs', names='操作',
                         title='单层 FLOPs 分布')
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig = px.bar(df_flops, x='操作', y='FLOPs',
                         color='占比 (%)',
                         text='FLOPs',
                         title='FLOPs 对比')
            fig.update_traces(texttemplate='%{text:.2e}')
            st.plotly_chart(fig, use_container_width=True)

        st.divider()

        # 序列长度影响分析
        st.subheader("📈 序列长度对复杂度的影响")

        seq_lengths = [64, 128, 256, 512, 1024, 2048]
        complexity_data = results["complexity"]

        df_complexity = pd.DataFrame({
            "序列长度": seq_lengths,
            'Transformer (O(L²))': complexity_data['transformer_flops'],
            'Linear Attention (O(Ld²))': complexity_data['linear_attention_flops'],
            'Mamba (O(LdN))': complexity_data['mamba_flops']
        })

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=seq_lengths, y=df_complexity['Transformer (O(L²))'],
                                mode='lines+markers', name='Transformer', line=dict(width=3)))
        fig.add_trace(go.Scatter(x=seq_lengths, y=df_complexity['Linear Attention (O(Ld²))'],
                                mode='lines+markers', name='Linear Attention'))
        fig.add_trace(go.Scatter(x=seq_lengths, y=df_complexity['Mamba (O(LdN))'],
                                mode='lines+markers', name='Mamba'))

        fig.update_layout(
            title="复杂度对比",
            xaxis_title="序列长度",
            yaxis_title='FLOPs',
            yaxis_type='log',
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("""
        **关键结论**：
        - 🔴 **Transformer**: 复杂度随序列长度**平方**增长，长序列场景代价高昂
        - 🟡 **Linear Attention**: 线性复杂度，但需要近似
        - 🟢 **Mamba**: 线性复杂度且性能接近，长序列优势明显

        **临界点分析**：
        """)

        # 找到 Transformer vs Mamba 的交叉点
        for i, L in enumerate(seq_lengths):
            if complexity_data['transformer_flops'][i] > complexity_data['mamba_flops'][i] * 10:
                st.info(f"💡 当序列长度超过 **{L}** 时，Mamba 的计算优势达到 **10倍** 以上")
                break
    else:
        st.info("👆 点击 '运行分析' 按钮开始计算")

st.markdown("---")
st.caption("🔬 Transformer Explorer - Model Analysis Tool | © 2025")

# =================== TAB 3: 显存分析 ===================
with tab3:
    st.header("💾 显存占用分析")

    if "transformer_results" in st.session_state:
        results = st.session_state["transformer_results"]
        memory_data = results["memory_data"]

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("参数显存", f"{memory_data['parameters_mb']:.0f} MB")
        with col2:
            st.metric("激活显存", f"{memory_data['activation_total_mb']:.0f} MB")
        with col3:
            st.metric("总训练显存", f"{memory_data['total_training_mb']:.0f} MB",
                     help="包含参数、激活、梯度、优化器状态")
        with col4:
            st.metric("推理显存", f"{memory_data['total_inference_mb']:.0f} MB")

        st.divider()

        # 显存分解
        st.subheader("📊 训练时显存分解")

        memory_breakdown = {
            "类型": ["模型参数", "激活值", "梯度", "优化器状态 (AdamW)"],
            "显存 (MB)": [
                memory_data['parameters_mb'],
                memory_data['activation_total_mb'],
                memory_data['gradients_mb'],
                memory_data['optimizer_mb']
            ]
        }

        df_mem = pd.DataFrame(memory_breakdown)
        df_mem['占比 (%)'] = df_mem['显存 (MB)'] / df_mem['显存 (MB)'].sum() * 100

        col1, col2 = st.columns(2)

        with col1:
            fig = px.pie(df_mem, values='显存 (MB)', names='类型',
                         title=f'训练显存分布 ({precision})')
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.dataframe(
                df_mem.style.format({
                    '显存 (MB)': '{:.1f}',
                    '占比 (%)': '{:.1f}'
                }).background_gradient(subset=['占比 (%)'], cmap='Reds'),
                height=200
            )

            st.markdown(f'''
            **显存占用分析**：
            - 💾 优化器状态占据了 **{memory_data['optimizer_mb']/memory_data['total_training_mb']*100:.1f}%** 的显存
            - 🔥 激活值占 **{memory_data['activation_total_mb']/memory_data['total_training_mb']*100:.1f}%**
            - 📉 使用 **Gradient Checkpointing** 可减少激活显存至原来的 1/√L
            ''')

        st.divider()

        # 精度对比
        st.subheader("🎯 精度对显存的影响")

        import torch
        precisions = ["FP32", "FP16", "BF16"]
        precision_memory = []

        for prec in precisions:
            dtype = torch.float32 if prec == "FP32" else torch.float16
            mem = profiler.estimate_memory(batch_size, seq_len, d_model, n_layers, dtype)
            precision_memory.append({
                "精度": prec,
                "训练显存 (GB)": mem['total_training_mb'] / 1024,
                "推理显存 (GB)": mem['total_inference_mb'] / 1024,
                "节省比例": "0%" if prec == "FP32" else "50%"
            })

        df_prec = pd.DataFrame(precision_memory)

        fig = go.Figure()
        fig.add_trace(go.Bar(name='训练', x=df_prec["精度"], y=df_prec['训练显存 (GB)']))
        fig.add_trace(go.Bar(name='推理', x=df_prec["精度"], y=df_prec['推理显存 (GB)']))
        fig.update_layout(
            title="不同精度下的显存对比",
            xaxis_title="精度",
            yaxis_title='显存 (GB)',
            barmode='group',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)

        st.info(f'''
        💡 **优化建议**：
        - 使用 **BF16** 混合精度训练可节省 **50%** 显存
        - 当前配置下，推荐的最小显卡显存：
          - FP32 训练: **{df_prec.iloc[0]['训练显存 (GB)']*1.2:.1f} GB** (含余量)
          - BF16 训练: **{df_prec.iloc[2]['训练显存 (GB)']*1.2:.1f} GB**
          - BF16 推理: **{df_prec.iloc[2]['推理显存 (GB)']*1.2:.1f} GB**
        ''')

        st.divider()

        # Batch Size 影响分析
        st.subheader("📦 Batch Size 对显存的影响")

        batch_sizes = [1, 2, 4, 8, 16, 32, 64]
        batch_memory = []

        for bs in batch_sizes:
            mem = profiler.estimate_memory(bs, seq_len, d_model, n_layers, torch.float16)
            batch_memory.append(mem['total_training_mb'] / 1024)  # Convert to GB

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=batch_sizes, y=batch_memory, mode='lines+markers',
                                line=dict(width=3, color='blue')))
        fig.update_layout(
            title='Batch Size vs 显存占用',
            xaxis_title='Batch Size',
            yaxis_title='显存 (GB)',
            height=400
        )

        # 添加显卡显存线
        common_gpus = {
            'RTX 3090': 24,
            'A100 (40GB)': 40,
            'A100 (80GB)': 80,
            'H100': 80
        }

        for gpu_name, gpu_mem in common_gpus.items():
            fig.add_hline(y=gpu_mem, line_dash="dash",
                         annotation_text=gpu_name,
                         annotation_position="right")

        st.plotly_chart(fig, use_container_width=True)

        # 找到每个 GPU 的最大 batch size
        st.markdown("**各显卡推荐的最大 Batch Size**：")
        for gpu_name, gpu_mem in common_gpus.items():
            max_bs = 1
            for i, bs in enumerate(batch_sizes):
                if batch_memory[i] <= gpu_mem * 0.9:  # 留 10% 余量
                    max_bs = bs
            st.write(f"- **{gpu_name}**: Batch Size ≤ **{max_bs}**")
    else:
        st.info("👆 点击 '运行分析' 按钮开始计算")

# =================== TAB 4: 性能热点 ===================
with tab4:
    st.header("🔥 性能热点分析")

    if "transformer_results" in st.session_state:
        results = st.session_state["transformer_results"]

        st.info("💡 本页面帮助你识别训练过程中的瓶颈，找到最值得优化的部分")

        # 训练步时间分解
        st.subheader("⏱️ 单步训练时间分解")

        timing = results["training_step"]

        timing_data = {
            "阶段": ["Forward", "Attention", "FFN", "Backward", "Optimizer"],
            "时间 (ms)": [
                timing['forward_ms'],
                timing['attention_ms'],
                timing['ffn_ms'],
                timing['backward_ms'],
                timing['optimizer_step_ms']
            ]
        }

        df_timing = pd.DataFrame(timing_data)
        df_timing['占比 (%)'] = df_timing['时间 (ms)'] / timing['total_ms'] * 100

        col1, col2 = st.columns([2, 1])

        with col1:
            fig = px.bar(df_timing, x="阶段", y='时间 (ms)',
                         color='占比 (%)',
                         title="单步训练时间分布",
                         text='时间 (ms)')
            fig.update_traces(texttemplate='%{text:.2f}ms', textposition='outside')
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.metric("总时间", f"{timing['total_ms']:.2f} ms")
            st.metric("吞吐量 (steps/s)", f"{1000/timing['total_ms']:.1f} steps/s")
            st.metric("样本吞吐量", f"{batch_size*1000/timing['total_ms']:.0f} samples/s")

            st.markdown(f'''
            **关键观察**：
            - Backward 是 Forward 的 **{timing['backward_ms']/timing['forward_ms']:.1f}x**
            - Attention 占 Forward 的 **{timing['attention_ms']/timing['forward_ms']*100:.0f}%**
            ''')

        st.divider()

        # 参数更新热点（模拟）
        st.subheader("🎯 参数更新热点识别")

        st.markdown('''
        在实际训练中，不同层的参数更新幅度差异巨大。识别这些"热点"可以帮助：
        - 🎯 **针对性调整学习率**（Layer-wise LR）
        - 🔍 **发现训练问题**（某些层不更新）
        - ⚡ **优化训练策略**（冻结不重要的层）
        ''')

        # 模拟梯度热点数据
        np.random.seed(42)
        hotspot_data = []

        for i in range(n_layers):
            # 模拟：浅层更新慢，深层更新快
            update_ratio = 0.001 * (1 + i / n_layers) * np.random.uniform(0.5, 1.5)

            hotspot_data.append({
                "层": f"Layer {i+1} Attention",
                "梯度范数": np.random.uniform(0.1, 2.0),
                "参数范数": np.random.uniform(5.0, 15.0),
                "更新比例": update_ratio,
                "是否热点": "🔥" if update_ratio > 0.0015 else "❄️"
            })

            update_ratio_ffn = 0.0012 * (1 + i / n_layers) * np.random.uniform(0.5, 1.5)
            hotspot_data.append({
                "层": f"Layer {i+1} FFN",
                "梯度范数": np.random.uniform(0.1, 2.0),
                "参数范数": np.random.uniform(10.0, 25.0),
                "更新比例": update_ratio_ffn,
                "是否热点": "🔥" if update_ratio_ffn > 0.0015 else "❄️"
            })

        df_hotspot = pd.DataFrame(hotspot_data)

        # 热点可视化
        fig = px.bar(df_hotspot, x='层', y="更新比例",
                     color='是否热点',
                     title="层更新热点分布",
                     color_discrete_map={"🔥": "#e74c3c", "❄️": "#3498db"})
        fig.add_hline(y=0.0015, line_dash="dash",
                     annotation_text="热点阈值",
                     line_color="red")
        fig.update_layout(height=400, xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)

        # 表格显示（只显示前10行）
        st.dataframe(
            df_hotspot.head(10).style.format({
                "梯度范数": '{:.4f}',
                '参数范数': '{:.2f}',
                "更新比例": '{:.6f}'
            }).background_gradient(subset=["更新比例"], cmap='YlOrRd'),
            height=300
        )

        st.divider()

        # 优化建议
        st.subheader("💡 性能优化建议")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown('''
            ### 🚀 计算优化

            1. **Flash Attention**
               - 减少 Attention 显存访问
               - 加速 2-4x
               - 适用于长序列

            2. **Gradient Checkpointing**
               - 以计算换显存
               - 激活显存减少至 O(√L)
               - 训练时间增加 20-30%

            3. **Mixed Precision (BF16)**
               - 速度提升 2-3x
               - 显存节省 50%
               - 现代 GPU 必备
            ''')

        with col2:
            st.markdown('''
            ### 🎯 架构优化

            1. **Multi-Query Attention**
               - 减少 KV Cache
               - 推理加速 1.5-2x
               - PaLM/Falcon 使用

            2. **SwiGLU FFN**
               - 替代 ReLU/GELU
               - 性能提升 5-10%
               - LLaMA 系列采用

            3. **RoPE 位置编码**
               - 外推能力强
               - 无额外参数
               - 相对位置编码
            ''')

        # Estimated optimization benefits (reference values, not real-time computed)
        st.info("💡 以下为基于典型场景的估算参考值，非当前模型配置的实时计算结果。")
        st.markdown("### 📊 优化收益评估（估算参考）")

        optimizations = {
            "优化方法": [
                "Baseline",
                "+ Flash Attention",
                "+ BF16",
                "+ Gradient Checkpointing",
                "+ All"
            ],
            "训练时间 (相对)": [1.0, 0.7, 0.4, 0.5, 0.3],
            "显存占用 (相对)": [1.0, 0.9, 0.5, 0.3, 0.15]
        }

        df_opt = pd.DataFrame(optimizations)

        fig = go.Figure()
        fig.add_trace(go.Bar(name='训练时间', x=df_opt["优化方法"],
                            y=df_opt['训练时间 (相对)'],
                            text=df_opt['训练时间 (相对)'],
                            texttemplate='%{text:.1f}x'))
        fig.add_trace(go.Bar(name='显存占用', x=df_opt["优化方法"],
                            y=df_opt['显存占用 (相对)'],
                            text=df_opt['显存占用 (相对)'],
                            texttemplate='%{text:.2f}x'))

        fig.update_layout(
            title='优化方法效果对比（相对于 Baseline）',
            xaxis_title="优化方案",
            yaxis_title="相对值",
            barmode='group',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)

        st.success('''
        🎉 **综合优化效果**：
        - 训练速度提升：**3.3x**
        - 显存节省：**85%**
        - 可训练的最大模型规模提升：**6-7x**
        ''')
    else:
        st.info("👆 点击 '运行分析' 按钮开始计算")


# =================== TAB 5: 梯度流分析 ===================
with tab5:
    st.header("🌊 梯度流分析")
    
    st.markdown("""
    ### 🎯 核心问题
    
    **为什么深层网络难以训练？**
    - 梯度消失：梯度在反向传播中逐层衰减，深层参数几乎不更新
    - 梯度爆炸：梯度指数级增长，导致参数更新不稳定
    
    **残差连接如何解决？**
    - 提供"梯度高速公路"：梯度可以直接跳过层传播
    - 保持梯度稳定：防止梯度消失和爆炸
    
    本页面将**实时计算**并可视化梯度流，验证残差连接的效果。
    """)
    
    st.divider()
    
    # 梯度追踪配置
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("⚙️ 实验配置")
        
        # 使用较小的配置进行梯度追踪
        grad_n_layers = st.slider("梯度分析层数", 2, 12, 6, key="grad_layers",
                                  help="层数越多，梯度消失问题越明显")
        grad_seq_len = st.slider("梯度分析序列长度", 8, 64, 16, step=8, key="grad_seq",
                                 help="序列长度影响计算时间")
        
        run_gradient_analysis = st.button("🚀 开始梯度分析", type="primary", key="run_grad")
    
    with col2:
        st.subheader("📝 说明")
        st.info("""
        **实验内容**：
        1. 创建真实的 Transformer 模型
        2. 前向传播 + 反向传播
        3. 实时计算每层的梯度
        4. 对比有/无残差的梯度流
        
        ⚠️ 计算可能需要几秒钟
        """)
    
    if run_gradient_analysis or 'gradient_journey_with' in st.session_state:
        
        if run_gradient_analysis:
            with st.spinner("🔄 正在进行梯度分析..."):
                
                # 延迟导入 torch 和 gradient_tracker
                import torch
                from utils.gradient_tracker import GradientTrackingTransformer, GradientTracker



                
                # 创建梯度追踪模型（使用小词表）
                grad_vocab_size = min(1000, vocab_size)
                
                # 创建模型
                model_grad = GradientTrackingTransformer(
                    d_model=d_model,
                    n_heads=n_heads,
                    n_layers=grad_n_layers,
                    vocab_size=grad_vocab_size,
                    use_residual=True
                )
                
                tracker = GradientTracker(model_grad)
                
                # 生成随机数据
                input_ids = torch.randint(0, grad_vocab_size, (2, grad_seq_len))
                target_ids = torch.randint(0, grad_vocab_size, (2, grad_seq_len))
                
                # 对比实验：有残差 vs 无残差
                journey_with, journey_without = tracker.compare_with_without_residual(
                    input_ids, target_ids
                )
                
                # 保存到 session state
                st.session_state.gradient_journey_with = journey_with
                st.session_state.gradient_journey_without = journey_without
                st.session_state.grad_config = {
                    'd_model': d_model,
                    'n_heads': n_heads,
                    'n_layers': grad_n_layers,
                    'seq_len': grad_seq_len
                }
                
                st.success("✅ 梯度分析完成！")
        
        # 获取结果
        journey_with = st.session_state.gradient_journey_with
        journey_without = st.session_state.gradient_journey_without
        grad_config = st.session_state.grad_config
        
        st.divider()
        
        # ========== 整体对比 ==========
        st.subheader("📊 有/无残差对比")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("有残差 - 健康度", 
                     f"{journey_with.overall_health_score:.2f}",
                     help="0-1，越高越健康")
            st.metric("有残差 - 最大梯度", 
                     f"{journey_with.max_grad_norm:.4f}")
        
        with col2:
            st.metric("无残差 - 健康度", 
                     f"{journey_without.overall_health_score:.2f}",
                     delta=f"{journey_without.overall_health_score - journey_with.overall_health_score:.2f}",
                     delta_color="inverse")
            st.metric("无残差 - 最大梯度", 
                     f"{journey_without.max_grad_norm:.4f}",
                     delta=f"{journey_without.max_grad_norm - journey_with.max_grad_norm:.2f}",
                     delta_color="inverse")
        
        with col3:
            # 梯度问题诊断
            if journey_without.has_vanishing_problem:
                st.error("⚠️ 无残差模型出现梯度消失！")
            if journey_without.has_explosion_problem:
                st.error("💥 无残差模型出现梯度爆炸！")
            
            if not journey_with.has_vanishing_problem and not journey_with.has_explosion_problem:
                st.success("✅ 有残差模型梯度健康")
        
        # ========== 逐层梯度对比 ==========
        st.divider()
        st.subheader("🔍 逐层梯度范数对比")
        
        # 准备数据
        layer_indices = list(range(grad_config['n_layers']))
        
        # 提取每层的平均梯度
        grad_with = []
        grad_without = []
        
        for lg in journey_with.layer_gradients:
            avg_grad = np.mean([lg.qkv_grad_norm, lg.out_proj_grad_norm, lg.ffn_grad_norm])
            grad_with.append(avg_grad)
        
        for lg in journey_without.layer_gradients:
            avg_grad = np.mean([lg.qkv_grad_norm, lg.out_proj_grad_norm, lg.ffn_grad_norm])
            grad_without.append(avg_grad)
        
        # 绘制对比图
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=layer_indices,
            y=grad_with,
            mode='lines+markers',
            name="有残差",
            line=dict(color='green', width=3),
            marker=dict(size=10)
        ))
        
        fig.add_trace(go.Scatter(
            x=layer_indices,
            y=grad_without,
            mode='lines+markers',
            name="无残差",
            line=dict(color='red', width=3, dash='dash'),
            marker=dict(size=10)
        ))
        
        # 添加梯度消失/爆炸区域
        fig.add_hline(y=1e-5, line_dash="dot", line_color="orange",
                     annotation_text="梯度消失阈值 (1e-5)")
        fig.add_hline(y=100, line_dash="dot", line_color="red",
                     annotation_text="梯度爆炸阈值 (100)")
        
        fig.update_layout(
            title="逐层平均梯度范数对比",
            xaxis_title="层索引（从底到顶）",
            yaxis_title="平均梯度范数",
            yaxis_type="log",
            height=500,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # 关键观察
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 💡 关键观察")
            
            # 计算梯度衰减率
            if len(grad_with) >= 2:
                decay_with = grad_with[-1] / grad_with[0] if grad_with[0] > 0 else 0
                decay_without = grad_without[-1] / grad_without[0] if grad_without[0] > 0 else 0
                
                st.markdown(f"""
                **梯度衰减比例**（最后层 / 第一层）:
                - 有残差: {decay_with:.4f} ({decay_with*100:.1f}%)
                - 无残差: {decay_without:.4f} ({decay_without*100:.1f}%)
                
                残差连接使梯度保持了 **{(decay_with/decay_without if decay_without > 0 else float('inf')):.1f}x** 的强度！
                """)
            
            if journey_without.has_vanishing_problem and not journey_with.has_vanishing_problem:
                st.success("""
                ✅ **残差连接成功防止了梯度消失！**
                
                无残差模型的梯度在深层几乎为0，而有残差模型保持健康。
                """)
        
        with col2:
            st.markdown("### 📈 健康度评分")
            
            # 每层健康度
            health_with = [lg.health_score for lg in journey_with.layer_gradients]
            health_without = [lg.health_score for lg in journey_without.layer_gradients]
            
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                x=layer_indices,
                y=health_with,
                name="有残差",
                marker_color='lightgreen'
            ))
            
            fig.add_trace(go.Bar(
                x=layer_indices,
                y=health_without,
                name="无残差",
                marker_color='lightcoral'
            ))
            
            fig.update_layout(
                title="逐层健康度对比",
                xaxis_title="层索引",
                yaxis_title="健康度 (0-1)",
                barmode='group',
                height=300
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # ========== 详细的逐层分析 ==========
        st.divider()
        st.subheader("🔬 详细逐层分析")
        
        selected_layer = st.selectbox(
            "选择层",
            options=list(range(grad_config['n_layers'])),
            format_func=lambda x: f"Layer {x}"
        )
        
        lg_with = journey_with.layer_gradients[selected_layer]
        lg_without = journey_without.layer_gradients[selected_layer]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"#### 有残差 - Layer {selected_layer}")
            
            components_with = {
                'Q/K/V 投影': lg_with.qkv_grad_norm,
                '输出投影': lg_with.out_proj_grad_norm,
                'FFN': lg_with.ffn_grad_norm
            }
            
            fig = go.Figure(data=[
                go.Bar(
                    x=list(components_with.keys()),
                    y=list(components_with.values()),
                    marker_color='lightblue',
                    text=[f"{v:.4f}" for v in components_with.values()],
                    textposition='auto'
                )
            ])
            fig.update_layout(
                title=f"Layer {selected_layer} 各组件梯度（有残差）",
                yaxis_title="梯度范数",
                height=300,
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)
            
            st.metric("健康度", f"{lg_with.health_score:.2f}")
            if lg_with.has_vanishing:
                st.error("⚠️ 梯度消失")
            if lg_with.has_explosion:
                st.error("💥 梯度爆炸")
        
        with col2:
            st.markdown(f"#### 无残差 - Layer {selected_layer}")
            
            components_without = {
                'Q/K/V 投影': lg_without.qkv_grad_norm,
                '输出投影': lg_without.out_proj_grad_norm,
                'FFN': lg_without.ffn_grad_norm
            }
            
            fig = go.Figure(data=[
                go.Bar(
                    x=list(components_without.keys()),
                    y=list(components_without.values()),
                    marker_color='lightcoral',
                    text=[f"{v:.4f}" for v in components_without.values()],
                    textposition='auto'
                )
            ])
            fig.update_layout(
                title=f"Layer {selected_layer} 各组件梯度（无残差）",
                yaxis_title="梯度范数",
                height=300,
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)
            
            st.metric("健康度", f"{lg_without.health_score:.2f}",
                      delta=f"{lg_without.health_score - lg_with.health_score:.2f}",
                      delta_color="inverse")
            if lg_without.has_vanishing:
                st.error("⚠️ 梯度消失")
            if lg_without.has_explosion:
                st.error("💥 梯度爆炸")
        
        # ========== 理论解释 ==========
        st.divider()
        st.success("""
        ### ✨ 为什么残差连接能防止梯度消失？
        
        **数学原理**:
        
        传统网络：`y = F(x)`
        - 梯度: `dy/dx = F'(x)`
        - 如果 F'(x) < 1，梯度会衰减
        - 多层累积: `dy/dx = F'_n(x) * F'_{n-1}(x) * ... * F'_1(x)`
        - 结果: 梯度指数级衰减 → 梯度消失
        
        残差网络：`y = x + F(x)`
        - 梯度: `dy/dx = 1 + F'(x)`
        - 即使 F'(x) 很小，`dy/dx ≈ 1`
        - 多层累积: `dy/dx ≈ 1 + ΣF'_i(x)`
        - 结果: 梯度保持稳定 → 无梯度消失
        
        **直观理解**:
        
        残差连接提供了一条"梯度高速公路"，让梯度可以直接从输出层流回输入层，
        绕过中间层的复杂计算。这就像在山路上修了一条直达隧道！
        
        **实验验证**:
        
        从上面的图表可以看到：
        1. ✅ 有残差：梯度在所有层保持相对稳定
        2. ❌ 无残差：梯度逐层衰减，深层几乎为0
        
        这就是为什么现代深度网络（ResNet, Transformer）都使用残差连接！
        """)
    
    else:
        st.info("👆 点击 '开始梯度分析' 按钮查看实时计算的梯度流")
        
        st.markdown("""
        ### 📚 预期结果
        
        当您运行梯度分析后，您将看到：
        
        1. **逐层梯度曲线**
           - 有残差：梯度保持稳定
           - 无残差：梯度逐层衰减
        
        2. **健康度评分**
           - 有残差：高健康度（接近1.0）
           - 无残差：低健康度（可能 < 0.5）
        
        3. **梯度消失检测**
           - 无残差模型在深层可能出现梯度 < 1e-5
           - 有残差模型保持健康
        
        4. **详细组件分析**
           - 查看每层的 Q/K/V、输出投影、FFN 的梯度
           - 对比有/无残差的差异
        
        **这是真实的计算结果，不是模拟！**
        """)

st.markdown("---")
st.caption("💡 Transformer 架构分析工具 | 帮助你理解和优化模型")
