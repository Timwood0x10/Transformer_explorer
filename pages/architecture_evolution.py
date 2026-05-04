"""
架构演进历史页面：展示从RNN到Transformer到Mamba的架构发展历程
"""
import streamlit as st
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

from utils.architecture_evolution import ArchitectureEvolutionTimeline




st.set_page_config(page_title="架构演进", page_icon="📅", layout="wide")

st.title("📅 序列建模架构演进")
st.markdown("### 从 RNN 到 Transformer 到 Mamba：序列建模架构的发展历程")

# Create timeline
@st.cache_resource
def create_timeline():
    """Create architecture evolution timeline"""
    return ArchitectureEvolutionTimeline()


timeline = create_timeline()

# Main tabs (no sidebar config needed - pure data visualization)
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📅 演进时间线",
    "⚡ 复杂度对比",
    "🔧 特性演化",
    "🎯 架构对比",
    "📋 演进报告"
])

# =================== TAB 1: 演进时间线 ===================
with tab1:
    st.header("📅 架构演进时间线")

    fig = timeline.create_evolution_timeline()
    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # Architecture details
    st.subheader("📋 架构详情")

    # Category filter
    categories = ["所有类别", "RNN", "Attention", "Transformer", "Efficient", "SSM"]
    selected_cat = st.selectbox("筛选类别", categories, key="timeline_cat")

    for arch in timeline.architectures:
        if selected_cat != "所有类别" and arch["category"] != selected_cat:
            continue

        with st.expander(f"**{arch['name']}** ({arch['year']}) - {arch['category']}"):
            st.markdown(f"**论文**：{arch['paper']} - *{arch['citation']}*")
            st.markdown(f"**描述**：{arch['description']}")
            st.markdown(f"**关键特性**：{', '.join(arch['key_features'])}")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("计算复杂度", arch["complexity"])
            with col2:
                st.metric("可并行化", "是" if arch["parallelizable"] else "否")
            with col3:
                st.metric("长距离依赖", arch["long_range"])

# =================== TAB 2: 复杂度对比 ===================
with tab2:
    st.header("⚡ 计算复杂度对比")

    fig = timeline.create_complexity_comparison()
    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # Complexity table
    st.subheader("📋 复杂度详细对比")
    complexity_data = []
    for arch in timeline.architectures:
        complexity_data.append({
            "架构": arch["name"],
            "年份": arch["year"],
            "类别": arch["category"],
            "计算复杂度": arch["complexity"],
            "可并行化": "是" if arch["parallelizable"] else "否",
            "长距离依赖": arch["long_range"]
        })

    df = pd.DataFrame(complexity_data)
    st.dataframe(df, height=500)

    st.markdown("""
    **复杂度等级说明**：
    - **O(T)**：线性复杂度，序列长度增长时计算量线性增长
    - **O(T log T)**：对数线性复杂度，略高于线性
    - **O(T√T)**：亚二次复杂度，介于线性和二次之间
    - **O(T²)**：二次复杂度，长序列时计算代价高昂
    """)

# =================== TAB 3: 特性演化 ===================
with tab3:
    st.header("🔧 关键特性演化")

    fig = timeline.create_feature_evolution_chart()
    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # Feature milestones
    st.subheader("📋 特性里程碑")

    features_timeline = [
        ("2014", "注意力机制", "Bahdanau 引入注意力权重，实现软对齐"),
        ("2016", "残差连接", "ResNet 的残差连接被引入序列建模"),
        ("2017", "多头注意力 + 位置编码", "Transformer 的核心创新"),
        ("2019", "稀疏注意力", "Longformer/Sparse Transformer 降低复杂度"),
        ("2020", "线性注意力", "Linformer/Performer 实现线性复杂度"),
        ("2021", "FlashAttention", "IO 感知的高效注意力计算"),
        ("2023", "状态空间模型 + 选择性机制", "Mamba 的核心创新"),
    ]

    for year, feature, desc in features_timeline:
        col1, col2 = st.columns([1, 4])
        with col1:
            st.markdown(f"**{year}**")
        with col2:
            st.markdown(f"**{feature}**：{desc}")

    st.divider()

    # Feature presence matrix
    st.subheader("📊 特性矩阵")
    feature_matrix_data = []
    key_features = ["自注意力", "残差连接", "位置编码", "稀疏注意力", "线性复杂度", "状态空间模型"]
    selected_archs = ["Seq2Seq + RNN", "Transformer", "Longformer", "FlashAttention", "Mamba"]

    feature_presence = {
        "Seq2Seq + RNN": [0, 0, 0, 0, 1, 0],
        "Transformer": [1, 1, 1, 0, 0, 0],
        "Longformer": [1, 1, 1, 1, 1, 0],
        "FlashAttention": [1, 1, 1, 0, 0, 0],
        "Mamba": [0, 0, 0, 0, 1, 1],
    }

    for arch in selected_archs:
        row = {"架构": arch}
        for i, feat in enumerate(key_features):
            row[feat] = "✅" if feature_presence[arch][i] else "❌"
        feature_matrix_data.append(row)

    st.dataframe(pd.DataFrame(feature_matrix_data), height=250)

# =================== TAB 4: 架构对比 ===================
with tab4:
    st.header("🎯 架构特性对比")

    fig = timeline.create_architecture_comparison_matrix()
    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # Detailed comparison table
    st.subheader("📋 详细对比")

    selected_archs = ["Seq2Seq + RNN", "Transformer", "Longformer", "Mamba"]
    arch_data = {arch["name"]: arch for arch in timeline.architectures if arch["name"] in selected_archs}

    comparison_data = []
    for name in selected_archs:
        if name in arch_data:
            arch = arch_data[name]
            comparison_data.append({
                "架构": name,
                "年份": arch["year"],
                "类别": arch["category"],
                "计算复杂度": arch["complexity"],
                "可并行化": "是" if arch["parallelizable"] else "否",
                "长距离依赖": arch["long_range"],
                "关键特性": ", ".join(arch["key_features"][:3])
            })

    st.dataframe(pd.DataFrame(comparison_data), height=250)

    st.markdown("""
    **雷达图解读**：
    - **并行性**：是否支持完全并行计算
    - **长距离依赖**：建模长距离依赖的能力
    - **计算复杂度**：数值越高表示复杂度越低（越高效）
    - **内存效率**：内存使用效率
    - **可解释性**：模型决策的可解释程度

    **关键观察**：
    - **Transformer** 在并行性和长距离依赖方面表现优异，但计算复杂度较高
    - **Longformer** 在保持 Transformer 优势的同时降低了计算复杂度
    - **Mamba** 在所有维度上表现均衡，特别是计算效率和内存效率
    - **RNN** 在并行性方面受限，但内存效率较高
    """)

# =================== TAB 5: 演进报告 ===================
with tab5:
    st.header("📋 架构演进报告")

    report = timeline.create_evolution_report()
    st.markdown(report)

st.markdown("---")
st.caption("📅 架构演进工具 | 序列建模架构发展历程")
