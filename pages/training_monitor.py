"""
训练监控页面：实时监控训练过程中的关键指标、层级健康和异常检测
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

from utils.training_monitor import TrainingMonitor, create_training_report


st.set_page_config(page_title="训练监控", page_icon="📊", layout="wide")

st.title("📊 训练动态监控")
st.markdown("### 实时监控训练过程中的关键指标、层级健康状况和异常检测")

# Sidebar configuration
with st.sidebar:
    st.divider()
    st.header("⚙️ 模型配置")

    n_layers = st.slider("层数", 1, 12, 4)
    learning_rate = st.slider("学习率", 1e-5, 1e-2, 1e-3, format="%.5f")
    window_size = st.slider("监控窗口", 10, 200, 50)

    st.divider()

    st.header("🔧 训练配置")
    n_steps = st.slider("模拟步数", 20, 200, 50)
    batch_size = st.slider("Batch Size", 4, 64, 16)


# Create a dummy model if BaseTransformer is not available
@st.cache_resource
def create_model(_n_layers):
    """Create a simple model for monitoring"""
    try:
        from utils.base_models import BaseTransformer


        return BaseTransformer(d_model=256, n_heads=4, n_layers=_n_layers, vocab_size=1000)
    except (ImportError, Exception):
        # Fallback: simple sequential model
        layers = []
        for i in range(_n_layers):
            layers.extend([
                nn.Linear(256, 256),
                nn.ReLU(),
            ])
        layers.append(nn.Linear(256, 10))
        return nn.Sequential(*layers)


# Create monitor and register hooks for layer-level tracking
model = create_model(n_layers)
monitor = TrainingMonitor(model, window_size=window_size)
monitor.register_hooks()

# Main tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 训练曲线",
    "🔥 层级健康",
    "⚠️ 异常检测",
    "📋 训练报告"
])

# =================== TAB 1: 训练曲线 ===================
with tab1:
    st.header("📊 训练曲线监控")

    if st.button("🚀 开始模拟训练", type="primary", key="start_training"):
        with st.spinner("🔄 正在模拟训练过程..."):
            monitor.metrics_history.clear()
            monitor.layer_metrics_history.clear()

            # Simulate training steps
            for step in range(n_steps):
                # Simulate loss decay with noise
                loss = 5.0 * np.exp(-step * 0.03) + np.random.normal(0, 0.05)
                lr = learning_rate * (0.99 ** step)

                # Simulate gradients by doing a forward/backward pass
                model.zero_grad()
                if isinstance(model, nn.Sequential):
                    x = torch.randn(batch_size, 256)
                    output = model(x)
                else:
                    # BaseTransformer expects Long tensor input_ids
                    x = torch.randint(0, 1000, (batch_size, 16))
                    output = model(x)
                loss_tensor = output.mean()
                loss_tensor.backward()

                metrics = monitor.step(step, 0, loss, lr, batch_size)

            st.success(f"✅ 完成 {n_steps} 步训练模拟！")

    # Display training curves
    if monitor.metrics_history:
        metrics_list = list(monitor.metrics_history)
        steps = [m.step for m in metrics_list]
        losses = [m.loss for m in metrics_list]
        grad_norms = [m.grad_norm for m in metrics_list]
        param_norms = [m.param_norm for m in metrics_list]

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("最终 Loss", f"{losses[-1]:.4f}",
                     delta=f"{losses[-1] - losses[0]:.4f}")
        with col2:
            st.metric("平均梯度范数", f"{np.mean(grad_norms):.4f}")
        with col3:
            st.metric("平均参数范数", f"{np.mean(param_norms):.4f}")

        st.divider()

        # Loss and gradient curves
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=steps, y=losses,
            mode='lines', name='Loss',
            line=dict(color='red', width=2)
        ))
        fig.update_layout(
            title="训练 Loss 曲线",
            xaxis_title='Step',
            yaxis_title='Loss',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)

        # Gradient norm curve
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=steps, y=grad_norms,
            mode='lines', name="梯度范数",
            line=dict(color='blue', width=2)
        ))
        fig2.update_layout(
            title="梯度范数变化",
            xaxis_title='Step',
            yaxis_title="梯度范数",
            height=350
        )
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("👆 点击 '开始模拟训练' 按钮查看训练曲线")

# =================== TAB 2: 层级健康 ===================
with tab2:
    st.header("🔥 层级健康状况")

    if monitor.layer_metrics_history:
        latest_metrics = monitor.layer_metrics_history[-1]

        layer_names = [m.layer_name for m in latest_metrics]
        update_ratios = [m.update_ratio for m in latest_metrics]
        sparsities = [m.activation_sparsity for m in latest_metrics]
        dead_ratios = [m.dead_neurons_ratio for m in latest_metrics]

        # --- Trend curves across training steps ---
        history = list(monitor.layer_metrics_history)
        if len(history) > 1:
            st.subheader("📈 层级指标变化趋势")

            # Collect per-layer time series
            all_layer_names = [m.layer_name for m in history[0]]
            steps = list(range(len(history)))

            # Build traces for each layer's update ratio over time
            fig_trend = go.Figure()
            for lname in all_layer_names:
                ratios = [next((m.update_ratio for m in h if m.layer_name == lname), None) for h in history]
                fig_trend.add_trace(go.Scatter(
                    x=steps, y=ratios, mode='lines', name=lname,
                ))
            fig_trend.update_layout(
                title="更新比例趋势",
                xaxis_title="训练步数", yaxis_title="更新比例",
                height=400, hovermode='x unified',
            )
            st.plotly_chart(fig_trend, use_container_width=True)

            # Gradient norm trend
            fig_grad = go.Figure()
            for lname in all_layer_names:
                grad_norms = [next((m.grad_norm for m in h if m.layer_name == lname), None) for h in history]
                fig_grad.add_trace(go.Scatter(
                    x=steps, y=grad_norms, mode='lines', name=lname,
                ))
            fig_grad.update_layout(
                title="梯度范数趋势",
                xaxis_title="训练步数", yaxis_title="梯度范数",
                height=400, yaxis_type='log', hovermode='x unified',
            )
            st.plotly_chart(fig_grad, use_container_width=True)

        st.divider()

        # Update ratio chart (latest snapshot)
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=layer_names,
            y=update_ratios,
            name="更新比例",
            marker_color='lightblue'
        ))
        fig.update_layout(
            title="各层更新比例",
            xaxis_title='层',
            yaxis_title="更新比例",
            height=400,
            xaxis_tickangle=-45
        )
        st.plotly_chart(fig, use_container_width=True)

        # Sparsity and dead neuron chart
        col1, col2 = st.columns(2)

        with col1:
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=layer_names,
                y=sparsities,
                name="激活稀疏性",
                marker_color='orange'
            ))
            fig.update_layout(
                title="激活稀疏性",
                xaxis_title='层',
                yaxis_title='稀疏性',
                height=350,
                xaxis_tickangle=-45
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=layer_names,
                y=dead_ratios,
                name="死神经元比例",
                marker_color='red'
            ))
            fig.update_layout(
                title="死神经元比例",
                xaxis_title='层',
                yaxis_title='比例',
                height=350,
                xaxis_tickangle=-45
            )
            st.plotly_chart(fig, use_container_width=True)

        # Layer health table
        st.subheader("📋 层级详细数据")
        layer_data = []
        for m in latest_metrics:
            layer_data.append({
                "层名称": m.layer_name,
                "梯度范数": f"{m.grad_norm:.6f}",
                "参数范数": f"{m.param_norm:.4f}",
                "更新比例": f"{m.update_ratio:.6f}",
                "激活稀疏性": f"{m.activation_sparsity:.4f}",
                "死神经元比例": f"{m.dead_neurons_ratio:.4f}"
            })
        st.dataframe(pd.DataFrame(layer_data), height=300)
    else:
        st.info("请先在 '训练曲线' 标签页中运行模拟训练")

# =================== TAB 3: 异常检测 ===================
with tab3:
    st.header("⚠️ 训练异常检测")

    if len(monitor.metrics_history) >= 10:
        anomalies = monitor.detect_anomalies()

        has_anomaly = any(anomalies.values())

        if has_anomaly:
            st.error("⚠️ 检测到以下训练异常：")
            for category, issues in anomalies.items():
                if issues:
                    st.subheader(f"🔴 {category}")
                    for issue in issues:
                        st.warning(f"- {issue}")
        else:
            st.success("✅ 未检测到明显异常，训练状态正常")

        st.divider()

        # Gradient norm distribution
        metrics_list = list(monitor.metrics_history)
        recent = metrics_list[-20:]

        fig = go.Figure()
        fig.add_trace(go.Box(
            y=[m.grad_norm for m in recent],
            name='梯度范数分布'
        ))
        fig.update_layout(
            title="近期梯度范数分布",
            height=350
        )
        st.plotly_chart(fig, use_container_width=True)

        # Loss trend analysis
        losses = [m.loss for m in recent]
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=list(range(len(losses))),
            y=losses,
            mode='lines+markers',
            name="近期 Loss",
            line=dict(color='red', width=2)
        ))
        fig2.add_hline(
            y=np.mean(losses), line_dash="dash",
            annotation_text=f"均值: {np.mean(losses):.4f}"
        )
        fig2.update_layout(
            title="近期 Loss 趋势",
            xaxis_title="近期步数",
            yaxis_title='Loss',
            height=350
        )
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("需要至少 10 步训练数据才能进行异常检测，请先运行模拟训练")

# =================== TAB 4: 训练报告 ===================
with tab4:
    st.header("📋 训练报告")

    if monitor.metrics_history:
        report = create_training_report(monitor)
        st.markdown(report)

        st.divider()

        # Summary metrics
        summary = monitor.get_training_summary()
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("总步数", summary.get('total_steps', 0))
        with col2:
            st.metric("平均 Loss", f"{summary.get('avg_loss', 0):.4f}")
        with col3:
            trend = summary.get('loss_trend', 'unknown')
            trend_text = "下降" if trend == "decreasing" else "上升"
            st.metric("Loss 趋势", trend_text)
        with col4:
            stable = summary.get('training_stable', False)
            st.metric("训练稳定性", "稳定" if stable else "异常")
    else:
        st.info("请先运行模拟训练以生成报告")

st.markdown("---")
st.caption("📊 训练监控工具 | 实时追踪训练动态")
