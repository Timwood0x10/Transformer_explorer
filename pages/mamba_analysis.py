"""
Mamba 模型分析页面：专门分析 Mamba/SSM 架构
"""
import streamlit as st
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import torch

from utils.mamba_profiler import MambaProfiler, create_sample_mamba
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np




st.set_page_config(page_title="Mamba 模型分析", page_icon="🐍", layout="wide")

st.title("🐍 Mamba 模型分析工具")
st.markdown("### 深度剖析 Mamba/SSM 架构的性能优势")

# 侧边栏配置
with st.sidebar:
    st.divider()
    st.header("⚙️ Mamba 配置")
    
    d_model = st.slider("模型维度 (d_model)", 128, 2048, 512, step=128)
    d_state = st.slider("状态维度 (d_state/N)", 4, 64, 16, step=4)
    d_conv = st.slider("卷积核大小 (d_conv)", 2, 8, 4)
    expand = st.slider("扩展因子 (expand)", 1, 4, 2)
    
    st.divider()
    
    st.header("💾 推理配置")
    batch_size = st.slider("Batch Size", 1, 64, 8)
    seq_len = st.slider("序列长度", 32, 8192, 1024, step=32)
    
    st.divider()
    
    st.header("📊 对比设置")
    compare_transformer = st.checkbox("对比 Transformer", value=True)
    n_heads = st.select_slider("Transformer 头数", [4, 8, 12, 16], value=8)

# 创建 Mamba 分析器 (cached, auto-rebuilds when params change)
@st.cache_resource
def get_profiler(d_model, d_state, d_conv, expand):
    return MambaProfiler(d_model, d_state, d_conv, expand)

profiler = get_profiler(d_model, d_state, d_conv, expand)

# =================== 运行分析按钮 ===================
if st.button("🔬 运行分析", type="primary", key="run_mamba_analysis"):
    with st.spinner("🔄 正在计算..."):
        try:
            param_counts = profiler.count_parameters()
            flops_data = profiler.estimate_flops(batch_size, seq_len)
            memory_data = profiler.estimate_memory(batch_size, seq_len, dtype=torch.float16)
            scaling_data = profiler.analyze_scaling(
                [128, 256, 512, 1024, 2048, 4096, 8192], batch_size)
            scan_data = profiler.get_selective_scan_analysis()
            if compare_transformer:
                comparison = profiler.compare_with_transformer(batch_size, seq_len, n_heads)
            else:
                comparison = None
            st.session_state["mamba_results"] = {
                "param_counts": param_counts,
                "flops_data": flops_data,
                "memory_data": memory_data,
                "scaling_data": scaling_data,
                "scan_data": scan_data,
                "comparison": comparison,
            }
            st.success("✅ 分析完成！")
        except Exception as e:
            st.error(f"计算出错: {e}")

# 主界面标签页
tab1, tab2, tab3, tab4 = st.tabs([
    "🧬 架构分析",
    "⚡ 性能对比",
    "📈 扩展性分析",
    "🎯 选择性扫描机制"
])

# =================== TAB 1: 架构分析 ===================
with tab1:
    st.header("🧬 Mamba 架构深度分析")

    if "mamba_results" in st.session_state:
        results = st.session_state["mamba_results"]

        # 配置展示
        st.subheader("📋 当前配置")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("模型维度", f"{d_model}")
        with col2:
            st.metric("状态维度", f"{d_state}")
        with col3:
            st.metric("内部维度", f"{d_model * expand}")
        with col4:
            st.metric("卷积核大小", f"{d_conv}")

        st.divider()

        # 参数分析
        st.subheader("📊 参数分布")

        params = results["param_counts"]

        # 创建参数分布数据
        param_data = {
            "组件": [
                "Input Projection",
                "Conv1d",
                "SSM Parameters",
                "  ├─ A Matrix",
                "  ├─ B Projection",
                "  ├─ C Projection",
                "  ├─ Δ Projection",
                "  └─ D Skip",
                "Output Projection",
                "Norm"
            ],
            "参数量": [
                params['input_projection'],
                params['conv1d'],
                params['ssm_parameters'],
                params['  - A_matrix'],
                params['  - B_projection'],
                params['  - C_projection'],
                params['  - dt_projection'],
                params['  - D_skip'],
                params['output_projection'],
                params['norm']
            ]
        }

        df_params = pd.DataFrame(param_data)
        df_params['占比 (%)'] = df_params["参数量"] / params['total_per_block'] * 100

        col1, col2 = st.columns([2, 1])

        with col1:
            # 参数分布饼图（只显示主要组件）
            main_components = df_params[~df_params['组件'].str.startswith('  ')].copy()
            fig = px.pie(main_components, values="参数量", names='组件',
                         title='Mamba 块参数分布')
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.metric("总参数量", f"{params['total_per_block']:,}")
            st.metric("SSM 参数", f"{params['ssm_parameters']:,}")
            st.metric("SSM 占比", f"{params['ssm_parameters']/params['total_per_block']*100:.1f}%")

            st.markdown("""
            **关键特征**：
            - SSM 参数规模较小
            - 线性投影占主要部分
            - 状态维度 N 可独立调整
            """)

        st.divider()

        # 详细参数表
        st.subheader("📋 详细参数统计")
        st.dataframe(
            df_params.style.format({
                "参数量": '{:,}',
                '占比 (%)': '{:.2f}'
            }).background_gradient(subset=['占比 (%)'], cmap='Greens'),
            height=400
        )

        st.divider()

        # FLOPs 分析
        st.subheader("⚡ 计算复杂度")

        flops = results["flops_data"]

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("总 FLOPs", f"{flops['total_per_block']/1e9:.2f} GFLOPs")
        with col2:
            st.metric("计算复杂度", flops['complexity'])
        with col3:
            st.metric("SSM FLOPs", f"{flops['ssm_total']/1e9:.2f} GFLOPs")

        # FLOPs 分解
        flops_breakdown = {
            "操作": [
                "Input Projection",
                "Conv1d",
                "SSM (Total)",
                "  ├─ B, C 计算",
                "  ├─ 状态更新",
                "  └─ 输出计算",
                "Output Projection",
                "Activation"
            ],
            "FLOPs": [
                flops['input_projection'],
                flops['conv1d'],
                flops['ssm_total'],
                flops['  - bc_computation'],
                flops['  - state_update'],
                flops['  - output_computation'],
                flops['output_projection'],
                flops['activation']
            ]
        }

        df_flops = pd.DataFrame(flops_breakdown)
        df_flops['占比 (%)'] = df_flops['FLOPs'] / flops['total_per_block'] * 100

        fig = px.bar(df_flops[~df_flops['操作'].str.startswith('  ')],
                     x='操作', y='FLOPs',
                     color='占比 (%)',
                     title='FLOPs 分布',
                     text='FLOPs')
        fig.update_traces(texttemplate='%{text:.2e}')
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("👆 点击 '运行分析' 按钮开始计算")

# =================== TAB 2: 性能对比 ===================
with tab2:
    st.header("⚡ Mamba vs Transformer 性能对比")

    if "mamba_results" in st.session_state:
        results = st.session_state["mamba_results"]

        if results["comparison"] is not None:
            comparison = results["comparison"]

            # 核心指标对比
            st.subheader("🎯 核心性能指标")

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(
                    "FLOPs 加速比",
                    f"{comparison['flops_speedup']:.2f}x",
                    delta=f"{(comparison['flops_speedup']-1)*100:.0f}%",
                    delta_color="normal"
                )
            with col2:
                st.metric(
                    "显存节省",
                    comparison['memory_savings'],
                    help="相比 Transformer 的显存节省"
                )
            with col3:
                st.metric(
                    "Cache 加速比",
                    f"{comparison['cache_speedup']:.2f}x",
                    help="推理时 KV Cache 的优势"
                )

            st.divider()

            # FLOPs 对比
            st.subheader("💻 计算量对比")

            flops_comparison = pd.DataFrame({
                '模型': ['Mamba', 'Transformer'],
                'FLOPs (GFLOPs)': [
                    comparison['mamba_flops'] / 1e9,
                    comparison['transformer_flops'] / 1e9
                ],
                '计算复杂度': ['O(LdN)', 'O(L²d)']
            })

            col1, col2 = st.columns(2)

            with col1:
                fig = px.bar(flops_comparison, x='模型', y='FLOPs (GFLOPs)',
                            color='模型',
                            title=f'FLOPs 对比 (Seq={seq_len})',
                            text='FLOPs (GFLOPs)',
                            color_discrete_map={'Mamba': '#2ecc71', 'Transformer': '#3498db'})
                fig.update_traces(texttemplate='%{text:.2f}')
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                st.dataframe(
                    flops_comparison.style.format({
                        'FLOPs (GFLOPs)': '{:.2f}'
                    }).background_gradient(subset=['FLOPs (GFLOPs)'], cmap='RdYlGn_r'),
                    height=150
                )

                if comparison['flops_speedup'] > 1:
                    st.success(f"✅ Mamba 快 **{comparison['flops_speedup']:.1f}x**")
                else:
                    st.info(f"ℹ️ 在序列长度 {seq_len} 下，Transformer 可能更快")

            st.divider()

            # 显存对比
            st.subheader("💾 显存占用对比")

            memory_comparison = pd.DataFrame({
                '模型': ['Mamba', 'Transformer'],
                '训练显存 (MB)': [
                    comparison['mamba_memory_mb'],
                    comparison['transformer_memory_mb']
                ],
                'Cache 显存 (MB)': [
                    comparison['mamba_cache_mb'],
                    comparison['transformer_cache_mb']
                ]
            })

            fig = go.Figure()
            fig.add_trace(go.Bar(
                name='训练显存',
                x=memory_comparison['模型'],
                y=memory_comparison['训练显存 (MB)'],
                text=memory_comparison['训练显存 (MB)'],
                texttemplate='%{text:.0f} MB'
            ))
            fig.add_trace(go.Bar(
                name='Cache 显存',
                x=memory_comparison['模型'],
                y=memory_comparison['Cache 显存 (MB)'],
                text=memory_comparison['Cache 显存 (MB)'],
                texttemplate='%{text:.0f} MB'
            ))
            fig.update_layout(
                title='显存占用对比',
                barmode='group',
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)

            st.markdown(f"""
            **关键观察**：
            - Mamba 的 KV Cache 显存: **{comparison['mamba_cache_mb']:.1f} MB**
            - Transformer 的 KV Cache 显存: **{comparison['transformer_cache_mb']:.1f} MB**
            - 节省比例: **{comparison['cache_savings']}**

            💡 在长序列推理时，Mamba 的固定大小状态空间优势显著！
            """)

        else:
            st.info("请在侧边栏勾选 '对比 Transformer' 以查看对比分析")
    else:
        st.info("👆 点击 '运行分析' 按钮开始计算")

# =================== TAB 3: 扩展性分析 ===================
with tab3:
    st.header("📈 扩展性分析：Mamba 的真正优势")

    if "mamba_results" in st.session_state:
        results = st.session_state["mamba_results"]

        st.markdown("""
        Mamba 的核心优势在于 **线性复杂度 O(L)**，而 Transformer 是 **平方复杂度 O(L²)**。
        随着序列长度增加，这个优势会越来越明显。
        """)

        # 序列长度扫描
        st.subheader("🔍 不同序列长度下的性能")

        # Use seq_lengths from the actual computation results
        scaling_data = results["scaling_data"]
        seq_lengths = scaling_data['seq_lengths']

        df_scaling = pd.DataFrame({
            "序列长度": seq_lengths,
            'Mamba FLOPs (GFLOPs)': [f/1e9 for f in scaling_data['mamba_flops']],
            'Transformer FLOPs (GFLOPs)': [f/1e9 for f in scaling_data['transformer_flops']],
            '加速比': scaling_data['speedup_ratio'],
            'Mamba 显存 (GB)': [m/1024 for m in scaling_data['mamba_memory']],
            'Transformer 显存 (GB)': [m/1024 for m in scaling_data['transformer_memory']]
        })

        # FLOPs 曲线
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=seq_lengths,
            y=df_scaling['Transformer FLOPs (GFLOPs)'],
            mode='lines+markers',
            name='Transformer (O(L²))',
            line=dict(color='#3498db', width=3),
            marker=dict(size=8)
        ))
        fig.add_trace(go.Scatter(
            x=seq_lengths,
            y=df_scaling['Mamba FLOPs (GFLOPs)'],
            mode='lines+markers',
            name='Mamba (O(L))',
            line=dict(color='#2ecc71', width=3),
            marker=dict(size=8)
        ))
        fig.update_layout(
            title='FLOPs vs 序列长度（对数坐标）',
            xaxis_title="序列长度",
            yaxis_title='FLOPs (GFLOPs)',
            yaxis_type='log',
            height=500,
            hovermode='x unified'
        )
        st.plotly_chart(fig, use_container_width=True)

        st.divider()

        # 加速比曲线
        st.subheader("🚀 加速比增长趋势")

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=seq_lengths,
            y=df_scaling['加速比'],
            mode='lines+markers',
            line=dict(color='#e74c3c', width=3),
            marker=dict(size=10),
            fill='tozeroy',
            fillcolor='rgba(231, 76, 60, 0.2)'
        ))
        fig.add_hline(y=1, line_dash="dash", line_color="gray",
                     annotation_text="等速线")
        fig.update_layout(
            title='Mamba 相对 Transformer 的加速比',
            xaxis_title="序列长度",
            yaxis_title='加速比 (倍数)',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)

        # 临界点分析
        crossover_point = None
        for i, speedup in enumerate(scaling_data['speedup_ratio']):
            if speedup > 1:
                crossover_point = seq_lengths[i]
                break

        if crossover_point:
            st.success(f"""
            🎯 **临界点分析**：
            - 当序列长度超过 **{crossover_point}** 时，Mamba 开始超越 Transformer
            - 在 **{seq_lengths[-1]}** 长度下，Mamba 快 **{df_scaling['加速比'].iloc[-1]:.1f}x**
            """)

        st.divider()

        # 详细数据表
        st.subheader("📊 详细数据")
        st.dataframe(
            df_scaling.style.format({
                'Mamba FLOPs (GFLOPs)': '{:.2f}',
                'Transformer FLOPs (GFLOPs)': '{:.2f}',
                '加速比': '{:.2f}x',
                'Mamba 显存 (GB)': '{:.2f}',
                'Transformer 显存 (GB)': '{:.2f}'
            }).background_gradient(subset=['加速比'], cmap='RdYlGn'),
            height=350
        )
    else:
        st.info("👆 点击 '运行分析' 按钮开始计算")

# =================== TAB 4: 选择性扫描机制 ===================
with tab4:
    st.header("🎯 Mamba 核心：选择性扫描机制")

    if "mamba_results" in st.session_state:
        results = st.session_state["mamba_results"]
        ssm_analysis = results["scan_data"]

        # 机制介绍
        st.subheader("🧬 什么是选择性状态空间模型？")

        col1, col2 = st.columns([3, 2])

        with col1:
            st.markdown(f"""
            ### 核心创新：{ssm_analysis['key_innovation']}

            传统的 SSM（如 S4）使用**固定**的参数 A, B, C。
            Mamba 的突破：让这些参数**依赖于输入**，实现选择性记忆！

            #### 数学表达式：

            **状态更新**：
            ```
            hₜ = Ā(xₜ) · hₜ₋₁ + B̄(xₜ) · xₜ
            yₜ = C(xₜ) · hₜ + D · xₜ
            ```

            其中：
            - **hₜ**: 当前时刻的隐藏状态
            - **xₜ**: 当前时刻的输入
            - **Ā(xₜ)**: 输入依赖的状态转移矩阵
            - **B̄(xₜ)**: 输入依赖的输入矩阵
            - **C(xₜ)**: 输入依赖的输出矩阵
            - **D**: 直接连接矩阵（跳过连接）
            - **Δ(xₜ)**: 控制离散化步长（选择性遗忘）
            - 状态维度: **{ssm_analysis['state_dimension']}**
            - 复杂度: **{ssm_analysis['complexity']}**

            #### 关键特性：
            1. **选择性记忆**：重要信息保留，无关信息遗忘
            2. **固定状态**：不随序列长度增长
            3. **线性复杂度**：O(L) vs Transformer 的 O(L²)
            """)

        with col2:
            st.info(f"""
            **机制名称**：
            {ssm_analysis['mechanism']}

            **状态维度**：
            {ssm_analysis['state_dimension']}

            **计算复杂度**：
            {ssm_analysis['complexity']}
            """)

            # 可视化状态空间
            st.markdown("#### 状态空间示意")

            # 创建一个简单的状态演化图
            t = np.linspace(0, 10, 100)
            h = np.exp(-0.3 * t) * np.sin(2 * t)

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=t, y=h, mode='lines',
                                    name='隐状态 h(t)',
                                    line=dict(color='#9b59b6', width=2)))
            fig.update_layout(
                title='选择性状态演化示例',
                xaxis_title='时间步',
                yaxis_title='状态值',
                height=300
            )
            st.plotly_chart(fig, use_container_width=True)

        st.divider()

        # 优势分析
        st.subheader("✨ 核心优势")

        advantages = ssm_analysis['advantages']
        for i, adv in enumerate(advantages, 1):
            st.markdown(f"{i}. ✅ **{adv}**")

        st.divider()

        # 权衡分析
        st.subheader("⚖️ 技术权衡")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### 🔴 潜在限制")
            for trade_off in ssm_analysis['trade_offs']:
                st.markdown(f"- {trade_off}")

        with col2:
            st.markdown("#### 🟢 最佳场景")
            for scenario in ssm_analysis['optimal_scenarios']:
                st.markdown(f"- {scenario}")

        st.divider()

        # 实际应用建议
        st.subheader("💡 应用建议")

        st.success(f"""
        ### 何时选择 Mamba？

        **强烈推荐** 👍：
        - 序列长度 > 2048
        - 推理性能关键（实时应用）
        - 显存受限（边缘设备）
        - 长期依赖建模

        **谨慎考虑** 🤔：
        - 序列长度 < 512
        - 训练速度最优先
        - 需要完全的并行性
        - 成熟工具链支持

        **当前配置分析**：
        - 序列长度: {seq_len}
        - 建议: {"✅ 使用 Mamba" if seq_len > 1024 else "⚠️ 考虑 Transformer"}
        """)
    else:
        st.info("👆 点击 '运行分析' 按钮开始计算")

st.markdown("---")
st.caption("🐍 Mamba Model Analysis Tool | © 2025 Transformer Explorer")
