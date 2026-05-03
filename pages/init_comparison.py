"""
初始化对比页面：对比不同参数初始化方法对模型训练的影响
"""
import streamlit as st
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

from utils.initialization_comparator import InitializationComparator

st.set_page_config(page_title="初始化对比", page_icon="⚖️", layout="wide")

st.title("⚖️ 参数初始化方法对比")
st.markdown("### 对比不同初始化方法对权重分布、激活值演化和梯度流的影响")

# Sidebar configuration
with st.sidebar:
    st.header("⚙️ 网络配置")

    layer_sizes_str = st.text_input(
        "层大小（逗号分隔）",
        value="512, 256, 128, 64, 10",
        help="定义网络各层的大小"
    )

    n_samples = st.slider("采样数量", 100, 5000, 1000, step=100)

    st.divider()

    st.header("📊 可视化选项")
    show_all_methods = st.checkbox("显示所有初始化方法", value=False)


# Parse layer sizes
def parse_layer_sizes(s):
    """Parse comma-separated layer sizes"""
    try:
        return [int(x.strip()) for x in s.split(',') if x.strip()]
    except ValueError:
        return [512, 256, 128, 64, 10]


layer_sizes = parse_layer_sizes(layer_sizes_str)

# Create comparator
@st.cache_resource
def create_comparator(_layer_sizes):
    """Create initialization comparator"""
    return InitializationComparator(_layer_sizes)


comparator = create_comparator(tuple(layer_sizes))

# Main tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 权重分布",
    "📈 激活值演化",
    "🌊 梯度流",
    "📋 对比报告"
])

# =================== TAB 1: 权重分布 ===================
with tab1:
    st.header("📊 权重分布对比")

    if st.button("🔄 运行初始化对比分析", type="primary", key="run_init"):
        with st.spinner("🔄 正在分析各初始化方法..."):
            comparator.init_results.clear()
            comparator.compare_all_initializations()
            st.success(f"✅ 完成对 {len(comparator.init_results)} 种初始化方法的分析！")

    if comparator.init_results:
        # Weight distribution visualization
        fig = comparator.visualize_weight_distributions()
        st.plotly_chart(fig, use_container_width=True)

        st.divider()

        # Detailed weight stats for first layer
        st.subheader("📋 第一层权重统计对比")
        weight_data = []
        for method, result in comparator.init_results.items():
            key = 'layer_0_weight'
            if key in result['activations']:
                stats = result['activations'][key]
                weight_data.append({
                    "初始化方法": method,
                    "均值": f"{stats['mean']:.6f}",
                    "标准差": f"{stats['std']:.6f}",
                    "最小值": f"{stats['min']:.6f}",
                    "最大值": f"{stats['max']:.6f}",
                    "Frobenius范数": f"{stats.get('frobenius_norm', 0):.4f}"
                })
        if weight_data:
            st.dataframe(pd.DataFrame(weight_data), height=300)
    else:
        st.info("👆 点击 '运行初始化对比分析' 按钮查看权重分布对比")

# =================== TAB 2: 激活值演化 ===================
with tab2:
    st.header("📈 激活值演化")

    if comparator.init_results:
        fig = comparator.visualize_activation_evolution()
        st.plotly_chart(fig, use_container_width=True)

        st.divider()

        # Activation stats per layer
        st.subheader("📋 各层激活值统计")
        activation_data = []
        for method, result in comparator.init_results.items():
            if not show_all_methods and method not in ['xavier_normal', 'kaiming_normal', 'random_normal']:
                continue
            for key, stats in result['activations'].items():
                if 'output' in key:
                    layer_num = key.split('_')[1]
                    activation_data.append({
                        "初始化方法": method,
                        "层": f"Layer {layer_num}",
                        "均值": f"{stats['mean']:.4f}",
                        "标准差": f"{stats['std']:.4f}",
                        "最小值": f"{stats['min']:.4f}",
                        "最大值": f"{stats['max']:.4f}"
                    })
        if activation_data:
            df_act = pd.DataFrame(activation_data)
            st.dataframe(df_act, height=300)
    else:
        st.info("请先在 '权重分布' 标签页中运行分析")

# =================== TAB 3: 梯度流 ===================
with tab3:
    st.header("🌊 梯度流对比")

    if comparator.init_results:
        fig = comparator.visualize_gradient_flow()
        st.plotly_chart(fig, use_container_width=True)

        st.divider()

        # Gradient health assessment
        st.subheader("📋 梯度健康状况评估")
        grad_data = []
        for method, result in comparator.init_results.items():
            grad_norms = [stats['norm'] for stats in result['gradient_stats'].values()]
            min_norm = min(grad_norms) if grad_norms else 0
            max_norm = max(grad_norms) if grad_norms else 0
            avg_norm = np.mean(grad_norms) if grad_norms else 0

            if min_norm < 1e-6:
                health = "⚠️ 梯度消失"
            elif max_norm > 10:
                health = "⚠️ 梯度爆炸"
            else:
                health = "✅ 健康"

            grad_data.append({
                "初始化方法": method,
                "最小梯度范数": f"{min_norm:.6f}",
                "最大梯度范数": f"{max_norm:.4f}",
                "平均梯度范数": f"{avg_norm:.4f}",
                "健康状况": health
            })
        if grad_data:
            df_grad = pd.DataFrame(grad_data)
            st.dataframe(df_grad, height=300)

        st.markdown("""
        **关键观察**：
        - **Xavier 初始化**：适用于 tanh/sigmoid 激活，保持前向和反向传播的方差一致
        - **Kaiming 初始化**：适用于 ReLU 激活，补偿 ReLU 将一半神经元置零的影响
        - **正交初始化**：保持梯度范数稳定，适合深度网络和 RNN
        - **随机初始化**：简单基准方法，在深层网络中容易出现梯度问题
        """)
    else:
        st.info("请先在 '权重分布' 标签页中运行分析")

# =================== TAB 4: 对比报告 ===================
with tab4:
    st.header("📋 初始化对比报告")

    if comparator.init_results:
        report = comparator.create_initialization_report()
        st.markdown(report)
    else:
        st.info("请先在 '权重分布' 标签页中运行分析以生成报告")

st.markdown("---")
st.caption("⚖️ 初始化对比工具 | 帮助选择最佳初始化方法")
