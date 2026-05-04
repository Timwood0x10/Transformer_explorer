"""
梯度流分析页面：可视化梯度在深度网络中的传播机制
"""
import streamlit as st
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

from utils.gradient_flow_visualizer import GradientFlowVisualizer




st.set_page_config(page_title="梯度流分析", page_icon="🌊", layout="wide")

st.title("🌊 梯度流分析")
st.markdown("### 可视化梯度在深度网络中的传播，对比不同激活函数和残差连接的效果")

# Sidebar configuration
with st.sidebar:
    st.divider()
    st.header("⚙️ 网络配置")

    input_size = st.slider("输入维度", 64, 1024, 512, step=64)
    num_batches = st.slider("批次数量", 1, 50, 10)

    st.divider()

    st.header("📊 分析选项")
    network_type = st.selectbox(
        "网络类型",
        options=["deep_relu", "deep_tanh", "deep_sigmoid", "residual", "lstm"],
        format_func=lambda x: {
            "deep_relu": "深度网络 (ReLU)",
            "deep_tanh": "深度网络 (Tanh)",
            "deep_sigmoid": "深度网络 (Sigmoid)",
            "residual": "残差网络",
            "lstm": "LSTM 网络"
        }[x]
    )


# Create visualizer
@st.cache_resource
def create_visualizer():
    """Create gradient flow visualizer"""
    return GradientFlowVisualizer()


visualizer = create_visualizer()

# Main tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "🌊 梯度流分析",
    "🔄 激活函数对比",
    "🔗 残差连接效果",
    "📋 分析报告"
])

# =================== TAB 1: 梯度流分析 ===================
with tab1:
    st.header("🌊 梯度流分析")

    if st.button("🚀 开始梯度流分析", type="primary", key="run_gf"):
        with st.spinner("🔄 正在分析梯度流..."):
            st.session_state.gf_network_type = network_type
            st.session_state.gf_input_size = input_size
            st.session_state.gf_num_batches = num_batches
            st.session_state.gf_done = True
            st.success("✅ 梯度流分析完成！")

    if st.session_state.get('gf_done', False):
        nt = st.session_state.gf_network_type
        inp_size = st.session_state.gf_input_size

        fig = visualizer.visualize_gradient_flow(nt)
        st.plotly_chart(fig, use_container_width=True)

        st.divider()

        # Detailed gradient stats
        networks = visualizer.create_sample_networks()
        if nt in networks:
            analysis = visualizer.analyze_gradient_flow(
                networks[nt], (32, inp_size), num_batches
            )

            st.subheader("📋 梯度详细统计")
            grad_data = []
            for name, stats in analysis['gradient_stats'].items():
                if 'weight' in name:
                    health = stats['gradient_health']
                    health_icon = "✅" if health == "健康" else "⚠️"
                    grad_data.append({
                        "层名称": name,
                        "平均梯度范数": f"{stats['avg_norm']:.6f}",
                        "梯度标准差": f"{stats['std_norm']:.6f}",
                        "最小梯度范数": f"{stats['min_norm']:.6f}",
                        "最大梯度范数": f"{stats['max_norm']:.6f}",
                        "健康状况": f"{health_icon} {health}"
                    })
            if grad_data:
                st.dataframe(pd.DataFrame(grad_data), height=300)
    else:
        st.info("👆 点击 '开始梯度流分析' 按钮查看梯度流可视化")

# =================== TAB 2: 激活函数对比 ===================
with tab2:
    st.header("🔄 激活函数对比")

    if st.button("🔄 运行激活函数对比", type="primary", key="run_act_cmp"):
        with st.spinner("🔄 正在对比不同激活函数的梯度流..."):
            st.session_state.act_cmp_done = True
            st.success("✅ 激活函数对比完成！")

    if st.session_state.get('act_cmp_done', False):
        fig = visualizer.compare_activation_functions()
        st.plotly_chart(fig, use_container_width=True)

        st.divider()

        # Health summary
        st.subheader("📋 激活函数健康度对比")
        networks = visualizer.create_sample_networks()
        health_data = []

        for act_name in ['relu', 'tanh', 'sigmoid']:
            net_key = f'deep_{act_name}'
            if net_key in networks:
                analysis = visualizer.analyze_gradient_flow(
                    networks[net_key], (32, input_size), num_batches
                )
                health_counts = {}
                for stats in analysis['gradient_stats'].values():
                    health = stats['gradient_health']
                    health_counts[health] = health_counts.get(health, 0) + 1

                health_data.append({
                    "激活函数": act_name.upper(),
                    "健康层数": health_counts.get("健康", 0),
                    "梯度消失层数": health_counts.get("梯度消失", 0),
                    "梯度爆炸层数": health_counts.get("梯度爆炸", 0),
                    "不稳定层数": health_counts.get("不稳定", 0)
                })

        if health_data:
            st.dataframe(pd.DataFrame(health_data), height=200)

        st.markdown("""
        **关键结论**：
        - **ReLU**：深层网络中梯度流最稳定，是深度学习的默认选择
        - **Tanh**：在深层网络中容易出现梯度消失，但比 Sigmoid 好
        - **Sigmoid**：深层网络中梯度消失最严重，不推荐用于隐藏层
        """)
    else:
        st.info("👆 点击 '运行激活函数对比' 按钮查看对比结果")

# =================== TAB 3: 残差连接效果 ===================
with tab3:
    st.header("🔗 残差连接效果")

    if st.button("🔄 运行残差连接对比", type="primary", key="run_res_cmp"):
        with st.spinner("🔄 正在对比有/无残差连接的梯度流..."):
            st.session_state.res_cmp_done = True
            st.success("✅ 残差连接对比完成！")

    if st.session_state.get('res_cmp_done', False):
        fig = visualizer.visualize_residual_connections()
        st.plotly_chart(fig, use_container_width=True)

        st.divider()

        # Detailed comparison
        st.subheader("📋 残差连接效果对比")
        networks = visualizer.create_sample_networks()
        comparison_data = []

        for net_type, label in [('deep_relu', '无残差连接'), ('residual', '有残差连接')]:
            if net_type in networks:
                analysis = visualizer.analyze_gradient_flow(
                    networks[net_type], (32, input_size), num_batches
                )
                grad_norms = []
                for stats in analysis['gradient_stats'].values():
                    if 'weight' in stats:
                        grad_norms.append(stats['avg_norm'])

                if grad_norms:
                    comparison_data.append({
                        "网络类型": label,
                        "平均梯度范数": f"{np.mean(grad_norms):.6f}",
                        "最小梯度范数": f"{np.min(grad_norms):.6f}",
                        "最大梯度范数": f"{np.max(grad_norms):.6f}",
                        "梯度衰减比": f"{np.min(grad_norms) / (np.max(grad_norms) + 1e-10):.6f}"
                    })

        if comparison_data:
            st.dataframe(pd.DataFrame(comparison_data), height=200)

        st.success("""
        **核心结论**：
        - 残差连接通过提供"梯度高速公路"有效防止梯度消失
        - 有残差连接的网络的梯度衰减比远高于无残差连接的网络
        - 这就是为什么 ResNet 和 Transformer 都使用残差连接
        """)
    else:
        st.info("👆 点击 '运行残差连接对比' 按钮查看对比结果")

# =================== TAB 4: 分析报告 ===================
with tab4:
    st.header("📋 梯度流分析报告")

    if st.button("📄 生成完整报告", type="primary", key="gen_gf_report"):
        with st.spinner("🔄 正在生成梯度流分析报告..."):
            report = visualizer.create_gradient_flow_report()
            st.session_state.gf_report = report
            st.success("✅ 报告生成完成！")

    if 'gf_report' in st.session_state:
        st.markdown(st.session_state.gf_report)
    else:
        st.info("👆 点击 '生成完整报告' 按钮查看详细分析报告")

st.markdown("---")
st.caption("🌊 梯度流分析工具 | 理解梯度传播机制")
