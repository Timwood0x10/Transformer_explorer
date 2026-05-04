"""
权重分析页面：分析模型权重分布、异常检测、权重演化和层间相关性
"""
import streamlit as st
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import torch
import torch.nn as nn
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

from utils.weight_analyzer import WeightAnalyzer


st.set_page_config(page_title="权重分析", page_icon="🔍", layout="wide")

st.title("🔍 模型权重分析")
st.markdown("### 深度分析模型权重分布、异常值检测、演化趋势和层间相关性")

# Sidebar configuration
with st.sidebar:
    st.divider()
    st.header("⚙️ 模型配置")

    d_model = st.slider("模型维度 (d_model)", 64, 1024, 256, step=64)
    n_layers = st.slider("层数", 1, 12, 4)

    st.divider()

    st.header("🔧 分析配置")
    n_evolution_steps = st.slider("演化步数", 5, 30, 10)


# Create a dummy model if BaseTransformer is not available
@st.cache_resource
def create_model(_d_model, _n_layers):
    """Create a simple model for weight analysis"""
    try:
        from utils.base_models import BaseTransformer


        return BaseTransformer(d_model=_d_model, n_heads=4, n_layers=_n_layers, vocab_size=1000)
    except (ImportError, Exception):
        # Fallback: simple sequential model
        layers = []
        for i in range(_n_layers):
            layers.extend([
                nn.Linear(_d_model, _d_model),
                nn.ReLU(),
            ])
        layers.append(nn.Linear(_d_model, 10))
        return nn.Sequential(*layers)


# Create analyzer
model = create_model(d_model, n_layers)
analyzer = WeightAnalyzer(model)

# Main tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 权重分布",
    "🔍 异常检测",
    "📈 权重演化",
    "🔗 层间相关性",
    "📋 分析报告"
])

# =================== TAB 1: 权重分布 ===================
with tab1:
    st.header("📊 权重分布分析")

    weight_stats = analyzer.analyze_weight_distribution()

    if weight_stats:
        # Summary metrics
        all_stds = [s.std for s in weight_stats]
        all_outliers = [s.outlier_ratio for s in weight_stats]
        all_deads = [s.dead_ratio for s in weight_stats]

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("分析层数", len(weight_stats))
        with col2:
            st.metric("平均标准差", f"{np.mean(all_stds):.6f}")
        with col3:
            st.metric("平均异常值比例", f"{np.mean(all_outliers)*100:.2f}%")

        st.divider()

        # Weight distribution histogram
        fig = analyzer.visualize_weight_distribution()
        st.plotly_chart(fig, use_container_width=True)

        st.divider()

        # Detailed stats table
        st.subheader("📋 各层权重统计")
        stats_data = []
        for s in weight_stats:
            stats_data.append({
                "层名称": s.layer_name,
                "均值": f"{s.mean:.6f}",
                "标准差": f"{s.std:.6f}",
                "偏度": f"{s.skewness:.4f}",
                "峰度": f"{s.kurtosis:.4f}",
                "异常值比例": f"{s.outlier_ratio:.4f}",
                "死权重比例": f"{s.dead_ratio:.4f}"
            })
        st.dataframe(pd.DataFrame(stats_data), height=300)
    else:
        st.info("未找到可分析的权重")

# =================== TAB 2: 异常检测 ===================
with tab2:
    st.header("🔍 权重异常检测")

    anomalies = analyzer.detect_weight_anomalies()

    if anomalies:
        st.error(f"⚠️ 检测到 {len(anomalies)} 层存在权重异常")

        for layer_name, issues in anomalies.items():
            with st.expander(f"🔴 {layer_name}"):
                for issue in issues:
                    st.warning(f"- {issue}")
    else:
        st.success("✅ 未检测到权重异常，所有层权重分布正常")

    st.divider()

    # Visualization of potential anomalies
    weight_stats = analyzer.analyze_weight_distribution()
    if weight_stats:
        fig = go.Figure()

        layer_names = [s.layer_name.split('.')[-1] for s in weight_stats]
        outlier_ratios = [s.outlier_ratio * 100 for s in weight_stats]
        dead_ratios = [s.dead_ratio * 100 for s in weight_stats]

        fig.add_trace(go.Bar(
            x=layer_names,
            y=outlier_ratios,
            name='异常值比例 (%)',
            marker_color='orange'
        ))
        fig.add_trace(go.Bar(
            x=layer_names,
            y=dead_ratios,
            name='死权重比例 (%)',
            marker_color='red'
        ))
        fig.add_hline(y=5, line_dash="dash", annotation_text="5% 阈值", line_color="gray")

        fig.update_layout(
            title="异常值与死权重分布",
            xaxis_title='层',
            yaxis_title='比例 (%)',
            barmode='group',
            height=400,
            xaxis_tickangle=-45
        )
        st.plotly_chart(fig, use_container_width=True)

# =================== TAB 3: 权重演化 ===================
with tab3:
    st.header("📈 权重演化分析")

    if st.button("🔄 模拟权重演化", type="primary", key="evolve_weights"):
        with st.spinner("🔄 正在模拟权重更新过程..."):
            analyzer.weight_history.clear()
            analyzer._save_initial_weights()

            for step in range(n_evolution_steps):
                # Simulate weight updates
                with torch.no_grad():
                    for param in model.parameters():
                        if param.requires_grad:
                            param.data += torch.randn_like(param.data) * 0.001 * (1 - step / n_evolution_steps)

                analyzer.record_weight_evolution(step)

            st.success(f"✅ 完成 {n_evolution_steps} 步权重演化模拟！")

    if analyzer.weight_history:
        # Select layer to visualize
        layer_names = list(analyzer.weight_history.keys())
        if layer_names:
            selected_layer = st.selectbox(
                "选择层",
                options=layer_names,
                format_func=lambda x: x
            )

            fig = analyzer.visualize_weight_evolution(selected_layer)
            st.plotly_chart(fig, use_container_width=True)

            st.divider()

            # Show evolution stats for all layers
            st.subheader("📋 各层演化统计")
            evo_data = []
            for name, history in analyzer.weight_history.items():
                if history:
                    first = history[0]
                    last = history[-1]
                    evo_data.append({
                        "层名称": name,
                        "初始均值": f"{first.mean:.6f}",
                        "最终均值": f"{last.mean:.6f}",
                        "初始标准差": f"{first.std:.6f}",
                        "最终标准差": f"{last.std:.6f}",
                        "更新幅度": f"{last.update_magnitude:.6f}"
                    })
            st.dataframe(pd.DataFrame(evo_data), height=300)
    else:
        st.info("👆 点击 '模拟权重演化' 按钮查看权重演化趋势")

# =================== TAB 4: 层间相关性 ===================
with tab4:
    st.header("🔗 层间权重相关性")

    with st.spinner("🔄 正在计算层间相关性..."):
        correlations = analyzer.analyze_weight_correlation()

    if correlations:
        layer_names = list(correlations.keys())
        if len(layer_names) > 1:
            # Build correlation matrix
            corr_matrix = []
            for name1 in layer_names:
                row = []
                for name2 in layer_names:
                    row.append(correlations[name1].get(name2, 0.0))
                corr_matrix.append(row)

            short_names = [n.split('.')[-1] for n in layer_names]

            fig = go.Figure(data=go.Heatmap(
                z=corr_matrix,
                x=short_names,
                y=short_names,
                colorscale='RdBu',
                zmin=-1, zmax=1,
                hovertemplate='%{x} ↔ %{y}<br>相关系数: %{z:.3f}<extra></extra>'
            ))
            fig.update_layout(
                title="层间权重相关性热力图",
                xaxis_title='层',
                yaxis_title='层',
                height=500,
                xaxis_tickangle=-45
            )
            st.plotly_chart(fig, use_container_width=True)

            # Find most correlated pairs
            st.subheader("📋 高相关性层对")
            pairs = []
            for i, name1 in enumerate(layer_names):
                for j, name2 in enumerate(layer_names):
                    if i < j:
                        corr = correlations[name1].get(name2, 0.0)
                        pairs.append({
                            "层1": name1,
                            "层2": name2,
                            "相关系数": f"{corr:.4f}"
                        })

            df_pairs = pd.DataFrame(pairs)
            df_pairs["相关系数"] = df_pairs["相关系数"].astype(float)
            df_pairs = df_pairs.reindex(df_pairs["相关系数"].abs().sort_values(ascending=False).index)
            st.dataframe(df_pairs.head(10), height=300)
        else:
            st.info("需要至少 2 层才能计算相关性")
    else:
        st.info("未找到可分析的权重")

# =================== TAB 5: 分析报告 ===================
with tab5:
    st.header("📋 权重分析报告")

    report = analyzer.generate_weight_report()
    st.markdown(report)

st.markdown("---")
st.caption("🔍 权重分析工具 | 深度分析模型权重状态")
