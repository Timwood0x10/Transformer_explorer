"""
注意力模式可视化页面：展示Transformer注意力机制的工作原理和模式
"""
import streamlit as st
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

from utils.attention_visualizer import AttentionVisualizer




st.set_page_config(page_title="注意力模式", page_icon="🔥", layout="wide")

st.title("🔥 注意力模式可视化")
st.markdown("### 展示Transformer注意力机制的工作原理、多头注意力和头多样性分析")

# Sidebar configuration
with st.sidebar:
    st.divider()
    st.header("⚙️ 模型配置")

    d_model = st.slider("模型维度 (d_model)", 128, 1024, 512, step=128)
    n_heads = st.select_slider("注意力头数", [1, 2, 4, 8, 12, 16], value=8)

    st.divider()

    st.header("📝 文本类型")
    text_type = st.radio(
        "选择任务类型",
        options=["machine_translation", "text_summarization", "question_answering"],
        format_func=lambda x: {
            "machine_translation": "机器翻译",
            "text_summarization": "文本摘要",
            "question_answering": "问答系统"
        }[x]
    )


# Create visualizer
@st.cache_resource
def create_visualizer(_d_model, _n_heads):
    """Create attention visualizer"""
    return AttentionVisualizer(d_model=_d_model, n_heads=_n_heads)


visualizer = create_visualizer(d_model, n_heads)

# Main tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "🔥 注意力热力图",
    "🧠 多头注意力",
    "📊 头多样性分析",
    "📋 分析报告"
])

# =================== TAB 1: 注意力热力图 ===================
with tab1:
    st.header("🔥 注意力热力图")

    # Select head to visualize
    head_idx = st.selectbox(
        "选择注意力头",
        options=list(range(n_heads)),
        format_func=lambda x: f"头 {x}",
        key="heatmap_head"
    )

    fig = visualizer.visualize_attention_heatmap(head_idx, text_type)
    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # Show pattern info
    data = visualizer.generate_attention_patterns(text_type)
    patterns = data["patterns"]
    tokens = data["tokens"]

    if f"head_{head_idx}" in patterns:
        pattern_info = patterns[f"head_{head_idx}"]
        col1, col2 = st.columns(2)
        with col1:
            st.metric("模式类型", pattern_info["pattern_type"])
        with col2:
            st.metric("Token 数量", len(tokens))

        st.markdown(f"**模式描述**：{pattern_info['description']}")

    st.divider()

    # Show all tokens
    st.subheader("📝 输入文本")
    st.write(f"**原文**：`{' '.join(tokens)}`")

# =================== TAB 2: 多头注意力 ===================
with tab2:
    st.header("🧠 多头注意力模式")

    st.markdown(f"当前配置：**{n_heads}** 个注意力头，模型维度 **{d_model}**，头维度 **{d_model // n_heads}**")

    # Generate patterns for all heads
    data = visualizer.generate_attention_patterns(text_type)
    tokens = data["tokens"]
    patterns = data["patterns"]

    # Display each head's heatmap
    cols = st.columns(4)
    for head in range(n_heads):
        if f"head_{head}" in patterns:
            attn_weights = patterns[f"head_{head}"]["weights"]
            pattern_type = patterns[f"head_{head}"]["pattern_type"]

            with cols[head % 4]:
                fig = go.Figure(data=go.Heatmap(
                    z=attn_weights,
                    x=tokens,
                    y=tokens,
                    colorscale='Blues',
                    showscale=False,
                    hovertemplate='%{z:.3f}<extra></extra>'
                ))
                fig.update_layout(
                    title=f'头 {head} ({pattern_type})',
                    height=250,
                    margin=dict(l=50, r=20, t=40, b=50),
                    xaxis_tickangle=-45
                )
                st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # Pattern distribution summary
    st.subheader("📋 模式分布统计")
    pattern_counts = {}
    for head_data in patterns.values():
        pt = head_data["pattern_type"]
        pattern_counts[pt] = pattern_counts.get(pt, 0) + 1

    fig = px.pie(
        values=list(pattern_counts.values()),
        names=list(pattern_counts.keys()),
        title="模式类型分布"
    )
    st.plotly_chart(fig, use_container_width=True)

# =================== TAB 3: 头多样性分析 ===================
with tab3:
    st.header("📊 头多样性分析")

    diversity = visualizer.analyze_attention_diversity(text_type)

    col1, col2 = st.columns(2)
    with col1:
        st.metric("多样性分数", f"{diversity['diversity_score']:.2f}",
                 help="模式类型数 / 总头数，越高越好")
    with col2:
        st.metric("模式类型数", len(diversity['pattern_distribution']))

    st.divider()

    # Similarity matrix
    st.subheader("🔗 头间相似度矩阵")
    similarities = diversity['similarities']

    if similarities:
        # Build similarity matrix
        head_indices = sorted(set(
            int(k.split('_')[1]) for k in similarities.keys()
        ))
        n = len(head_indices)
        sim_matrix = np.eye(n)

        for key, val in similarities.items():
            # Key format: "head_0_vs_head_1" -> extract indices
            parts = key.split('_vs_head_')
            i = int(parts[0].replace('head_', ''))
            j = int(parts[1])
            if i < n and j < n:
                sim_matrix[i][j] = val
                sim_matrix[j][i] = val

        fig = go.Figure(data=go.Heatmap(
            z=sim_matrix,
            x=[f"头 {i}" for i in head_indices],
            y=[f"头 {i}" for i in head_indices],
            colorscale='RdBu',
            zmin=-1, zmax=1,
            hovertemplate='%{x} ↔ %{y}<br>相似度: %{z:.3f}<extra></extra>'
        ))
        fig.update_layout(
            title="头间余弦相似度",
            height=500,
            xaxis_tickangle=-45
        )
        st.plotly_chart(fig, use_container_width=True)

        # Most and least similar pairs
        st.divider()
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("最相似的注意力头对")
            sorted_sims = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
            for pair, sim in sorted_sims[:5]:
                st.write(f"- **{pair}**: {sim:.4f}")

        with col2:
            st.subheader("最不相似的注意力头对")
            for pair, sim in sorted_sims[-5:]:
                st.write(f"- **{pair}**: {sim:.4f}")
    else:
        st.info("需要至少 2 个注意力头才能进行多样性分析")

    st.divider()

    # Pattern distribution details
    st.subheader("📋 模式分布详情")
    pattern_dist = diversity['pattern_distribution']
    pattern_df = pd.DataFrame([
        {"模式类型": k, "头数量": v, "占比": f"{v/n_heads*100:.1f}%"}
        for k, v in pattern_dist.items()
    ])
    st.dataframe(pattern_df, height=200)

    st.markdown("""
    **分析要点**：
    - **高多样性**意味着不同头学习了不同的注意力模式，信息捕获更丰富
    - 如果所有头的模式高度相似，可能存在冗余，考虑减少头数
    - 不同任务类型会引导模型学习不同的注意力策略
    """)

# =================== TAB 4: 分析报告 ===================
with tab4:
    st.header("📋 注意力分析报告")

    report = visualizer.create_attention_summary_report()
    st.markdown(report)

st.markdown("---")
st.caption("🔥 注意力模式分析工具 | 理解注意力机制")
