"""
Token 旅程可视化：追踪 Token 在 Transformer 中的完整流程
==========================================================

交互式可视化一个 Token 从输入到输出的完整旅程：
1. 输入 Embedding
2. 逐层 Transformer 处理
3. 残差连接的修正效果
4. 最终 Logits 和预测

作者：Transformer Explorer Team
日期：2025-12-29
"""

import streamlit as st
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import torch
import torch.nn.functional as F
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from typing import Dict, List, Tuple

from utils.token_tracker import (
    TrackedTransformer, 
    TokenTracker, 
    create_simple_vocab,
    TokenJourney,
    LayerState
)

# 页面配置
st.set_page_config(
    page_title="Token 旅程追踪器",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS 样式
st.markdown("""
<style>
    .big-font {
        font-size: 24px !important;
        font-weight: bold;
    }
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
    }
    .token-highlight {
        background-color: #ffeb3b;
        padding: 5px 10px;
        border-radius: 5px;
        font-weight: bold;
        font-size: 18px;
    }
    .layer-box {
        border: 2px solid #1f77b4;
        border-radius: 8px;
        padding: 10px;
        margin: 5px 0;
        background-color: #e3f2fd;
    }
</style>
""", unsafe_allow_html=True)

# 标题
st.title("🚀 Token 旅程追踪器")
st.markdown("### 追踪一个 Token 如何在 Transformer 中逐层变换，最终成为预测")

# ==========================================
# 侧边栏：模型配置和输入
# ==========================================
with st.sidebar:
    st.header("⚙️ 模型配置")
    
    # 模型参数
    vocab_size = st.select_slider(
        "词表大小",
        options=[100, 300, 500, 1000, 2000],
        value=500,
        help="词表越大，模型越复杂"
    )
    
    d_model = st.select_slider(
        "模型维度 (d_model)",
        options=[64, 128, 256, 512],
        value=256,
        help="隐藏状态的维度"
    )
    
    n_heads = st.selectbox(
        "注意力头数",
        options=[2, 4, 8, 16],
        index=2,
        help="多头注意力的头数"
    )
    
    n_layers = st.slider(
        "层数",
        min_value=2,
        max_value=12,
        value=4,
        help="Transformer 层数"
    )
    
    st.divider()
    
    st.header("📝 输入文本")
    
    # 预设示例
    example_texts = {
        "示例1: AI学习": "I love learning AI models",
        "示例2: 编程": "The quick brown fox jumps",
        "示例3: 问答": "What is the capital city",
        "自定义": ""
    }
    
    selected_example = st.selectbox(
        "选择示例或自定义",
        options=list(example_texts.keys())
    )
    
    if selected_example == "自定义":
        input_text = st.text_input(
            "输入文本（空格分隔）",
            "I love learning AI models",
            help="输入要分析的文本"
        )
    else:
        input_text = example_texts[selected_example]
        st.info(f"📄 文本: {input_text}")
    
    tokens = input_text.split()
    
    st.divider()
    
    st.header("🎯 选择 Token")
    
    # 用户选择要追踪的 token
    token_position = st.selectbox(
        "选择要追踪的 Token 位置",
        options=list(range(len(tokens))),
        format_func=lambda x: f"位置 {x}: '{tokens[x]}'",
        help="选择序列中的哪个 token 进行追踪"
    )
    
    st.markdown(f"**追踪的 Token**: <span class='token-highlight'>{tokens[token_position]}</span>", 
                unsafe_allow_html=True)
    
    st.divider()
    
    # 追踪按钮
    track_button = st.button("🚀 开始追踪", type="primary", use_container_width=True)

# ==========================================
# 缓存模型和词表
# ==========================================
@st.cache_resource
def get_model_and_vocab(_vocab_size, _d_model, _n_heads, _n_layers):
    """创建并缓存模型"""
    vocab = create_simple_vocab(_vocab_size)
    model = TrackedTransformer(_vocab_size, _d_model, _n_heads, _n_layers)
    tracker = TokenTracker(model, vocab)
    return tracker, vocab

# ==========================================
# 主界面
# ==========================================

# 获取模型
with st.spinner("🔧 初始化模型..."):
    tracker, vocab = get_model_and_vocab(vocab_size, d_model, n_heads, n_layers)

# 显示模型信息
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("词表大小", f"{vocab_size:,}")
with col2:
    st.metric("模型维度", d_model)
with col3:
    st.metric("注意力头数", n_heads)
with col4:
    st.metric("层数", n_layers)

st.divider()

# 如果点击追踪按钮或者已有追踪结果
if track_button or 'journey' in st.session_state:
    
    if track_button:
        # 执行追踪
        with st.spinner(f"🔍 正在追踪 Token '{tokens[token_position]}'..."):
            try:
                journey = tracker.track_token_journey(input_text, token_position, return_top_k=20)
                st.session_state.journey = journey
                st.success(f"✅ 追踪完成！Token '{journey.token_text}' 的旅程已记录")
            except Exception as e:
                st.error(f"❌ 追踪失败: {str(e)}")
                st.stop()
    
    journey = st.session_state.journey
    
    # 创建标签页
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 总览", 
        "🔄 逐层追踪", 
        "➕ 残差修正",
        "🎯 注意力分析",
        "📈 Logits & 预测"
    ])
    
    # ==========================================
    # Tab 1: 总览
    # ==========================================
    with tab1:
        st.header("📊 Token 旅程总览")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("🎫 Token 信息")
            
            info_df = pd.DataFrame([
                {"属性": "Token 文本", "值": journey.token_text},
                {"属性": "Token ID", "值": str(journey.token_id)},
                {"属性": "位置", "值": f"{journey.token_position}/{len(tokens)-1}"},
                {"属性": "嵌入范数", "值": f"{np.linalg.norm(journey.embedding):.4f}"},
                {"属性": "最终范数", "值": f"{np.linalg.norm(journey.final_hidden):.4f}"},
            ])
            
            st.dataframe(info_df, hide_index=True, use_container_width=True)
            
            # 显示完整序列
            st.subheader("📝 完整序列")
            sequence_html = " ".join([
                f"<span class='token-highlight'>{token}</span>" if i == token_position 
                else f"<span>{token}</span>"
                for i, token in enumerate(tokens)
            ])
            st.markdown(sequence_html, unsafe_allow_html=True)
        
        with col2:
            st.subheader("🏆 Top-5 预测结果")
            
            pred_data = []
            for rank, (token_id, token_text, prob) in enumerate(journey.top_k_predictions[:5], 1):
                pred_data.append({
                    "排名": rank,
                    "Token": token_text,
                    "ID": token_id,
                    "概率": f"{prob:.2%}",
                    "概率值": prob
                })
            
            pred_df = pd.DataFrame(pred_data)
            
            st.dataframe(
                pred_df[["排名", "Token", "ID", "概率"]],
                hide_index=True,
                use_container_width=True
            )
            
            # 概率条形图
            fig = px.bar(
                pred_df, 
                x="Token", 
                y="概率值",
                title="Top-5 预测概率",
                text="概率"
            )
            fig.update_traces(texttemplate='%{text}', textposition='outside')
            fig.update_layout(height=300, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
    
    # ==========================================
    # Tab 2: 逐层追踪
    # ==========================================
    with tab2:
        st.header("🔄 逐层追踪 Token 的变换")
        
        st.markdown(f"""
        追踪 Token **'{journey.token_text}'** 在 {len(journey.layer_states)} 层 Transformer 中的演化过程。
        每一层都会对 Token 的隐藏状态进行修改和优化。
        """)
        
        # 层选择器
        selected_layer = st.selectbox(
            "选择要查看的层",
            options=list(range(len(journey.layer_states))),
            format_func=lambda x: f"Layer {x}",
            key="layer_selector"
        )
        
        layer_state = journey.layer_states[selected_layer]
        
        st.divider()
        
        # 显示该层的详细信息
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader(f"📊 Layer {selected_layer} 统计信息")
            
            metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
            with metrics_col1:
                st.metric("输入范数", f"{layer_state.norm_input:.4f}")
            with metrics_col2:
                st.metric("输出范数", f"{layer_state.norm_output:.4f}")
            with metrics_col3:
                delta_color = "normal" if abs(layer_state.norm_change) < 0.1 else "inverse"
                st.metric("变化", f"{layer_state.norm_change:+.4f}", delta_color=delta_color)
            
            # 隐藏状态向量可视化
            st.subheader("🎨 隐藏状态向量")
            
            viz_option = st.radio(
                "选择可视化方式",
                ["热力图", "折线图", "直方图"],
                horizontal=True,
                key=f"viz_layer_{selected_layer}"
            )
            
            if viz_option == "热力图":
                # 热力图显示向量
                data_to_show = np.vstack([
                    layer_state.input_hidden,
                    layer_state.output_hidden
                ])
                
                fig = go.Figure(data=go.Heatmap(
                    z=data_to_show,
                    y=["输入", "输出"],
                    colorscale='RdBu',
                    zmid=0,
                    colorbar=dict(title="值")
                ))
                fig.update_layout(
                    title=f"Layer {selected_layer} 隐藏状态对比",
                    xaxis_title="维度",
                    height=300
                )
                st.plotly_chart(fig, use_container_width=True)
            
            elif viz_option == "折线图":
                # 折线图对比
                dims = np.arange(len(layer_state.input_hidden))
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=dims, y=layer_state.input_hidden,
                    mode='lines', name='输入',
                    line=dict(color='blue', width=2)
                ))
                fig.add_trace(go.Scatter(
                    x=dims, y=layer_state.output_hidden,
                    mode='lines', name='输出',
                    line=dict(color='red', width=2)
                ))
                fig.update_layout(
                    title=f"Layer {selected_layer} 隐藏状态演化",
                    xaxis_title="维度索引",
                    yaxis_title="激活值",
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
            
            else:  # 直方图
                fig = go.Figure()
                fig.add_trace(go.Histogram(
                    x=layer_state.input_hidden,
                    name='输入分布',
                    opacity=0.7,
                    nbinsx=50
                ))
                fig.add_trace(go.Histogram(
                    x=layer_state.output_hidden,
                    name='输出分布',
                    opacity=0.7,
                    nbinsx=50
                ))
                fig.update_layout(
                    title=f"Layer {selected_layer} 激活值分布",
                    xaxis_title="激活值",
                    yaxis_title="频次",
                    barmode='overlay',
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("🔍 详细数据")
            
            # 显示关键向量的统计
            stats_data = []
            
            if layer_state.attn_query is not None:
                stats_data.append({
                    "向量": "Query (Q)",
                    "范数": f"{np.linalg.norm(layer_state.attn_query):.4f}",
                    "均值": f"{np.mean(layer_state.attn_query):.4f}",
                    "标准差": f"{np.std(layer_state.attn_query):.4f}"
                })
            
            if layer_state.attn_key is not None:
                stats_data.append({
                    "向量": "Key (K)",
                    "范数": f"{np.linalg.norm(layer_state.attn_key):.4f}",
                    "均值": f"{np.mean(layer_state.attn_key):.4f}",
                    "标准差": f"{np.std(layer_state.attn_key):.4f}"
                })
            
            if layer_state.attn_value is not None:
                stats_data.append({
                    "向量": "Value (V)",
                    "范数": f"{np.linalg.norm(layer_state.attn_value):.4f}",
                    "均值": f"{np.mean(layer_state.attn_value):.4f}",
                    "标准差": f"{np.std(layer_state.attn_value):.4f}"
                })
            
            if layer_state.attn_output is not None:
                stats_data.append({
                    "向量": "Attention 输出",
                    "范数": f"{np.linalg.norm(layer_state.attn_output):.4f}",
                    "均值": f"{np.mean(layer_state.attn_output):.4f}",
                    "标准差": f"{np.std(layer_state.attn_output):.4f}"
                })
            
            if layer_state.ffn_output is not None:
                stats_data.append({
                    "向量": "FFN 输出",
                    "范数": f"{np.linalg.norm(layer_state.ffn_output):.4f}",
                    "均值": f"{np.mean(layer_state.ffn_output):.4f}",
                    "标准差": f"{np.std(layer_state.ffn_output):.4f}"
                })
            
            if stats_data:
                st.dataframe(
                    pd.DataFrame(stats_data),
                    hide_index=True,
                    use_container_width=True
                )
            
            # FFN 中间层可视化
            if layer_state.ffn_intermediate is not None:
                st.subheader("🧬 FFN 维度扩展")
                st.info(f"FFN 将维度从 {d_model} 扩展到 {len(layer_state.ffn_intermediate)}")
                
                # 显示前10个和后10个维度
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=list(range(20)),
                    y=np.concatenate([
                        layer_state.ffn_intermediate[:10],
                        layer_state.ffn_intermediate[-10:]
                    ]),
                    marker_color='lightblue'
                ))
                fig.update_layout(
                    title="FFN 中间激活（前10+后10维）",
                    xaxis_title="维度",
                    yaxis_title="激活值",
                    height=250
                )
                st.plotly_chart(fig, use_container_width=True)
        
        st.divider()
        
        # 所有层的演化趋势
        st.subheader("📈 跨层演化趋势")
        
        layer_norms = [ls.norm_output for ls in journey.layer_states]
        layer_indices = list(range(len(journey.layer_states)))
        
        col1, col2 = st.columns(2)
        
        with col1:
            # 范数变化
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=layer_indices,
                y=layer_norms,
                mode='lines+markers',
                marker=dict(size=10),
                line=dict(width=3, color='blue')
            ))
            fig.update_layout(
                title="隐藏状态范数的演化",
                xaxis_title="层数",
                yaxis_title="L2 范数",
                height=300
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # 变化量
            norm_changes = [ls.norm_change for ls in journey.layer_states]
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=layer_indices,
                y=norm_changes,
                marker_color=['red' if x < 0 else 'green' for x in norm_changes]
            ))
            fig.update_layout(
                title="每层的范数变化",
                xaxis_title="层数",
                yaxis_title="变化量",
                height=300
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # ==========================================
    # Tab 3: 残差修正 - 核心功能！
    # ==========================================
    with tab3:
        st.header("➕ 残差连接：修正而非替换")
        
        st.markdown("""
        ### 🎯 核心问题：残差连接如何工作？
        
        **传统网络**: `output = Transform(input)` - 完全替换
        
        **残差网络**: `output = input + Transform(input)` - 在原有基础上修正
        
        这个标签页将直观地展示每一层如何**逐步修正**而非**完全替换**隐藏状态。
        """)
        
        st.divider()
        
        # 选择要分析的层
        selected_res_layer = st.selectbox(
            "选择要详细分析的层",
            options=list(range(len(journey.layer_states))),
            format_func=lambda x: f"Layer {x}",
            key="residual_layer_selector"
        )
        
        layer_state = journey.layer_states[selected_res_layer]
        
        # === 第一个残差连接分析 ===
        st.subheader(f"🔵 Layer {selected_res_layer} - Attention 残差连接")
        
        col1, col2 = st.columns([3, 2])
        
        with col1:
            if layer_state.residual_delta_1 is not None:
                # 可视化残差修正过程
                st.markdown("**公式**: `输出 = 输入 + Attention(输入)`")
                
                # 创建三个向量的对比图
                fig = go.Figure()
                
                dims = np.arange(min(50, len(layer_state.input_hidden)))  # 只显示前50维
                
                # 输入（蓝色）
                fig.add_trace(go.Scatter(
                    x=dims,
                    y=layer_state.input_hidden[:len(dims)],
                    mode='lines',
                    name='输入 (x)',
                    line=dict(color='blue', width=2)
                ))
                
                # Attention 输出（绿色）
                if layer_state.before_residual_1 is not None:
                    fig.add_trace(go.Scatter(
                        x=dims,
                        y=layer_state.before_residual_1[:len(dims)],
                        mode='lines',
                        name='Attention(x)',
                        line=dict(color='green', width=2, dash='dash')
                    ))
                
                # 残差后（红色）
                if layer_state.after_residual_1 is not None:
                    fig.add_trace(go.Scatter(
                        x=dims,
                        y=layer_state.after_residual_1[:len(dims)],
                        mode='lines',
                        name='输出 (x + Attention)',
                        line=dict(color='red', width=3)
                    ))
                
                fig.update_layout(
                    title="Attention 残差连接可视化（前50维）",
                    xaxis_title="维度索引",
                    yaxis_title="激活值",
                    height=400,
                    hovermode='x unified'
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # 残差贡献的热力图
                st.markdown("**残差修正量** (Delta):")
                
                # 重塑为二维以便显示
                delta_reshaped = layer_state.residual_delta_1[:100].reshape(10, 10)
                
                fig = go.Figure(data=go.Heatmap(
                    z=delta_reshaped,
                    colorscale='RdBu',
                    zmid=0,
                    colorbar=dict(title="修正量")
                ))
                fig.update_layout(
                    title="残差修正量热力图（前100维）",
                    height=300
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("### 📊 统计数据")
            
            # 计算关键指标
            input_norm = np.linalg.norm(layer_state.input_hidden)
            attn_norm = np.linalg.norm(layer_state.before_residual_1) if layer_state.before_residual_1 is not None else 0
            delta_norm = np.linalg.norm(layer_state.residual_delta_1)
            output_norm = np.linalg.norm(layer_state.after_residual_1) if layer_state.after_residual_1 is not None else 0
            
            # 修正比例
            correction_ratio = (delta_norm / input_norm) * 100 if input_norm > 0 else 0
            
            st.metric("输入范数", f"{input_norm:.4f}")
            st.metric("Attention 输出范数", f"{attn_norm:.4f}")
            st.metric("残差修正量范数", f"{delta_norm:.4f}", 
                     help="残差连接带来的修正大小")
            st.metric("修正比例", f"{correction_ratio:.2f}%",
                     help="修正量相对于输入的比例")
            st.metric("最终输出范数", f"{output_norm:.4f}")
            
            # 关键观察
            st.markdown("---")
            st.markdown("### 💡 关键观察")
            
            if correction_ratio < 10:
                st.success("✅ 小幅修正 - 保持了原始信息")
            elif correction_ratio < 50:
                st.info("📊 适度修正 - 平衡了新旧信息")
            else:
                st.warning("⚠️ 大幅修正 - 引入了较多新信息")
            
            # 余弦相似度
            if layer_state.input_hidden is not None and layer_state.after_residual_1 is not None:
                cos_sim = np.dot(layer_state.input_hidden, layer_state.after_residual_1) / (
                    np.linalg.norm(layer_state.input_hidden) * np.linalg.norm(layer_state.after_residual_1)
                )
                st.metric("输入-输出相似度", f"{cos_sim:.4f}",
                         help="余弦相似度，越接近1说明保留了越多原始信息")
        
        st.divider()
        
        # === 第二个残差连接分析 ===
        st.subheader(f"🟢 Layer {selected_res_layer} - FFN 残差连接")
        
        col1, col2 = st.columns([3, 2])
        
        with col1:
            if layer_state.residual_delta_2 is not None:
                st.markdown("**公式**: `输出 = Norm(x + Attention) + FFN(Norm(x + Attention))`")
                
                # 可视化第二个残差
                fig = go.Figure()
                
                dims = np.arange(min(50, len(layer_state.input_hidden)))
                
                # 第一次残差后的状态作为"输入"
                if layer_state.after_residual_1 is not None:
                    # 这里实际上应该是 norm 之后的，但我们用 after_residual_1 近似
                    baseline = layer_state.after_residual_1[:len(dims)]
                    fig.add_trace(go.Scatter(
                        x=dims,
                        y=baseline,
                        mode='lines',
                        name='Norm(x + Attention)',
                        line=dict(color='blue', width=2)
                    ))
                
                # FFN 输出
                if layer_state.ffn_output is not None:
                    fig.add_trace(go.Scatter(
                        x=dims,
                        y=layer_state.ffn_output[:len(dims)],
                        mode='lines',
                        name='FFN 输出',
                        line=dict(color='green', width=2, dash='dash')
                    ))
                
                # 第二次残差后
                if layer_state.after_residual_2 is not None:
                    fig.add_trace(go.Scatter(
                        x=dims,
                        y=layer_state.after_residual_2[:len(dims)],
                        mode='lines',
                        name='最终输出',
                        line=dict(color='red', width=3)
                    ))
                
                fig.update_layout(
                    title="FFN 残差连接可视化（前50维）",
                    xaxis_title="维度索引",
                    yaxis_title="激活值",
                    height=400,
                    hovermode='x unified'
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("### 📊 统计数据")
            
            ffn_output_norm = np.linalg.norm(layer_state.ffn_output) if layer_state.ffn_output is not None else 0
            delta_norm_2 = np.linalg.norm(layer_state.residual_delta_2)
            final_output_norm = np.linalg.norm(layer_state.after_residual_2) if layer_state.after_residual_2 is not None else 0
            
            st.metric("FFN 输出范数", f"{ffn_output_norm:.4f}")
            st.metric("残差修正量范数", f"{delta_norm_2:.4f}")
            st.metric("最终输出范数", f"{final_output_norm:.4f}")
            
            # 两次残差对比
            st.markdown("---")
            st.markdown("### 🔄 两次残差对比")
            
            comparison_df = pd.DataFrame([
                {"残差": "Attention 残差", "修正量": delta_norm},
                {"残差": "FFN 残差", "修正量": delta_norm_2}
            ])
            
            fig = px.bar(comparison_df, x="残差", y="修正量", 
                        color="残差",
                        title="两次残差贡献对比")
            fig.update_layout(height=250, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        st.divider()
        
        # === 所有层的残差效果总览 ===
        st.subheader("🌐 所有层的残差贡献总览")
        
        st.markdown("""
        下图展示了每一层的残差连接对隐藏状态的修正程度。
        这帮助我们理解：
        - 哪些层的修正更激进
        - 残差连接如何帮助信息流动
        - 深层网络如何避免梯度消失
        """)
        
        # 收集所有层的残差数据
        residual_data = []
        for ls in journey.layer_states:
            layer_data = {"层": f"Layer {ls.layer_idx}"}
            
            if ls.residual_delta_1 is not None:
                layer_data["Attention 残差"] = np.linalg.norm(ls.residual_delta_1)
            
            if ls.residual_delta_2 is not None:
                layer_data["FFN 残差"] = np.linalg.norm(ls.residual_delta_2)
            
            residual_data.append(layer_data)
        
        residual_df = pd.DataFrame(residual_data)
        
        # 堆叠柱状图
        fig = go.Figure()
        
        if "Attention 残差" in residual_df.columns:
            fig.add_trace(go.Bar(
                name='Attention 残差',
                x=residual_df['层'],
                y=residual_df['Attention 残差'],
                marker_color='lightblue'
            ))
        
        if "FFN 残差" in residual_df.columns:
            fig.add_trace(go.Bar(
                name='FFN 残差',
                x=residual_df['层'],
                y=residual_df['FFN 残差'],
                marker_color='lightcoral'
            ))
        
        fig.update_layout(
            title="每层的残差贡献（堆叠图）",
            xaxis_title="层",
            yaxis_title="残差修正量（L2 范数）",
            barmode='stack',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # 累积效应
        st.subheader("📈 残差的累积效应")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # 计算从输入到当前层的累积变化
            cumulative_changes = []
            initial_state = journey.embedding
            
            for ls in journey.layer_states:
                if ls.output_hidden is not None:
                    total_change = np.linalg.norm(ls.output_hidden - initial_state)
                    cumulative_changes.append(total_change)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=list(range(len(cumulative_changes))),
                y=cumulative_changes,
                mode='lines+markers',
                fill='tozeroy',
                line=dict(color='purple', width=3)
            ))
            fig.update_layout(
                title="从初始嵌入的累积变化",
                xaxis_title="层数",
                yaxis_title="累积变化量",
                height=300
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # 相对输入的变化比例
            relative_changes = []
            for ls in journey.layer_states:
                if ls.output_hidden is not None and ls.input_hidden is not None:
                    rel_change = (np.linalg.norm(ls.output_hidden) / np.linalg.norm(ls.input_hidden) - 1) * 100
                    relative_changes.append(rel_change)
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=list(range(len(relative_changes))),
                y=relative_changes,
                marker_color=['green' if x > 0 else 'red' for x in relative_changes]
            ))
            fig.update_layout(
                title="每层的相对变化（%）",
                xaxis_title="层数",
                yaxis_title="变化百分比",
                height=300
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # 关键洞察
        st.markdown("---")
        st.success("""
        ### ✨ 关键洞察
        
        1. **保持信息**: 残差连接确保原始信息不会丢失，每层只是"微调"表示
        2. **梯度高速公路**: 梯度可以直接通过残差路径反向传播，避免梯度消失
        3. **增量学习**: 网络学习的是"修正量"而非"新状态"，训练更稳定
        4. **深度可扩展性**: 有了残差连接，可以训练几百层的超深网络
        """)
    
    # ==========================================
    # Tab 4: 注意力分析
    # ==========================================
    with tab4:
        st.header("🎯 注意力权重分析")
        
        st.markdown(f"""
        查看 Token **'{journey.token_text}'** 在每一层对其他 Token 的注意力分布。
        注意力机制决定了如何聚合上下文信息。
        """)
        
        # 选择层
        selected_attn_layer = st.selectbox(
            "选择要查看的层",
            options=list(range(len(journey.layer_states))),
            format_func=lambda x: f"Layer {x}",
            key="attention_layer_selector"
        )
        
        layer_state = journey.layer_states[selected_attn_layer]
        
        st.divider()
        
        if layer_state.attn_weights is not None:
            col1, col2 = st.columns([3, 2])
            
            with col1:
                st.subheader(f"📊 Layer {selected_attn_layer} 注意力权重")
                
                # 注意力权重条形图
                fig = go.Figure()
                
                fig.add_trace(go.Bar(
                    x=tokens,
                    y=layer_state.attn_weights,
                    marker_color=['red' if i == token_position else 'lightblue' 
                                 for i in range(len(tokens))],
                    text=[f"{w:.3f}" for w in layer_state.attn_weights],
                    textposition='auto'
                ))
                
                fig.update_layout(
                    title=f"'{journey.token_text}' 对其他 Token 的注意力",
                    xaxis_title="Token",
                    yaxis_title="注意力权重",
                    height=400,
                    showlegend=False
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # 注意力热力图（如果有多层数据）
                st.subheader("🔥 跨层注意力模式")
                
                # 收集所有层的注意力权重
                attn_matrix = []
                layer_names = []
                
                for ls in journey.layer_states:
                    if ls.attn_weights is not None:
                        attn_matrix.append(ls.attn_weights)
                        layer_names.append(f"L{ls.layer_idx}")
                
                if attn_matrix:
                    attn_matrix = np.array(attn_matrix)
                    
                    fig = go.Figure(data=go.Heatmap(
                        z=attn_matrix,
                        x=tokens,
                        y=layer_names,
                        colorscale='Blues',
                        colorbar=dict(title="注意力")
                    ))
                    
                    fig.update_layout(
                        title="所有层的注意力模式",
                        xaxis_title="Token 位置",
                        yaxis_title="层",
                        height=400
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.subheader("📈 注意力统计")
                
                # 关键统计
                max_attn_idx = np.argmax(layer_state.attn_weights)
                max_attn_token = tokens[max_attn_idx]
                max_attn_weight = layer_state.attn_weights[max_attn_idx]
                
                self_attn = layer_state.attn_weights[token_position]
                
                st.metric("最关注的 Token", max_attn_token)
                st.metric("最大注意力权重", f"{max_attn_weight:.4f}")
                st.metric("自注意力权重", f"{self_attn:.4f}",
                         help="Token 对自己的注意力")
                
                # 注意力分布分析
                st.markdown("---")
                st.markdown("### 🔍 分布特征")
                
                # 计算熵（衡量注意力的分散程度）
                attn_entropy = -np.sum(layer_state.attn_weights * np.log(layer_state.attn_weights + 1e-10))
                max_entropy = np.log(len(layer_state.attn_weights))
                normalized_entropy = attn_entropy / max_entropy
                
                st.metric("注意力熵", f"{normalized_entropy:.4f}",
                         help="越接近1越分散，越接近0越集中")
                
                if normalized_entropy > 0.8:
                    st.info("📊 注意力分散 - 均匀关注所有位置")
                elif normalized_entropy > 0.5:
                    st.info("🎯 注意力适中 - 有重点但不极端")
                else:
                    st.success("🔍 注意力集中 - 聚焦于少数关键位置")
                
                # Top-3 关注的 Token
                st.markdown("### 🏆 Top-3 关注")
                
                top_3_indices = np.argsort(layer_state.attn_weights)[-3:][::-1]
                
                for rank, idx in enumerate(top_3_indices, 1):
                    st.markdown(f"{rank}. **{tokens[idx]}** - {layer_state.attn_weights[idx]:.4f}")
                
                # 注意力向量可视化
                st.markdown("---")
                st.markdown("### 📐 注意力向量")
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=list(range(len(layer_state.attn_weights))),
                    y=layer_state.attn_weights,
                    mode='lines+markers',
                    fill='tozeroy',
                    marker=dict(size=10, color='blue')
                ))
                fig.update_layout(
                    title="注意力分布曲线",
                    xaxis_title="位置",
                    yaxis_title="权重",
                    height=250,
                    showlegend=False
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("⚠️ 该层没有记录注意力权重")
        
        st.divider()
        
        # 注意力演化分析
        st.subheader("🌊 注意力的跨层演化")
        
        st.markdown("""
        观察注意力模式如何在不同层中变化：
        - **浅层**: 通常关注局部和句法结构
        - **中层**: 关注语义关系
        - **深层**: 关注任务相关的特定模式
        """)
        
        # 对比不同层对同一个 token 的注意力
        if len(tokens) > 1:
            compare_token_idx = st.selectbox(
                "选择要对比的目标 Token",
                options=[i for i in range(len(tokens)) if i != token_position],
                format_func=lambda x: f"{tokens[x]} (位置 {x})"
            )
            
            # 收集所有层对该 token 的注意力
            attn_to_target = []
            
            for ls in journey.layer_states:
                if ls.attn_weights is not None:
                    attn_to_target.append(ls.attn_weights[compare_token_idx])
            
            if attn_to_target:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=list(range(len(attn_to_target))),
                    y=attn_to_target,
                    mode='lines+markers',
                    line=dict(width=3, color='green'),
                    marker=dict(size=10)
                ))
                fig.update_layout(
                    title=f"'{journey.token_text}' 对 '{tokens[compare_token_idx]}' 的注意力演化",
                    xaxis_title="层数",
                    yaxis_title="注意力权重",
                    height=350
                )
                st.plotly_chart(fig, use_container_width=True)
    
    # ==========================================
    # Tab 5: Logits 和预测
    # ==========================================
    with tab5:
        st.header("📈 从隐藏状态到预测")
        
        st.markdown(f"""
        展示 Token **'{journey.token_text}'** 的最终隐藏状态如何通过 **lm_head** 线性层
        转换为词表大小的 Logits，然后通过 Softmax 得到概率分布。
        
        **这是 Token 旅程的最后一步！**
        """)
        
        st.divider()
        
        # 最终隐藏状态
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("🎨 最终隐藏状态")
            
            # 可视化最终隐藏状态
            viz_type = st.radio(
                "可视化方式",
                ["热力图", "折线图", "分布图"],
                horizontal=True,
                key="final_hidden_viz"
            )
            
            if viz_type == "热力图":
                # 重塑为 2D
                hidden_2d = journey.final_hidden[:min(100, len(journey.final_hidden))].reshape(10, -1)
                
                fig = go.Figure(data=go.Heatmap(
                    z=hidden_2d,
                    colorscale='RdBu',
                    zmid=0,
                    colorbar=dict(title="激活值")
                ))
                fig.update_layout(
                    title="最终隐藏状态（前100维）",
                    height=350
                )
                st.plotly_chart(fig, use_container_width=True)
            
            elif viz_type == "折线图":
                dims = np.arange(len(journey.final_hidden))
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=dims,
                    y=journey.final_hidden,
                    mode='lines',
                    line=dict(color='purple', width=2)
                ))
                fig.update_layout(
                    title="最终隐藏状态向量",
                    xaxis_title="维度",
                    yaxis_title="值",
                    height=350
                )
                st.plotly_chart(fig, use_container_width=True)
            
            else:  # 分布图
                fig = go.Figure()
                fig.add_trace(go.Histogram(
                    x=journey.final_hidden,
                    nbinsx=50,
                    marker_color='purple'
                ))
                fig.update_layout(
                    title="最终隐藏状态分布",
                    xaxis_title="激活值",
                    yaxis_title="频次",
                    height=350
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("📊 隐藏状态统计")
            
            st.metric("维度", len(journey.final_hidden))
            st.metric("L2 范数", f"{np.linalg.norm(journey.final_hidden):.4f}")
            st.metric("均值", f"{np.mean(journey.final_hidden):.4f}")
            st.metric("标准差", f"{np.std(journey.final_hidden):.4f}")
            st.metric("最大值", f"{np.max(journey.final_hidden):.4f}")
            st.metric("最小值", f"{np.min(journey.final_hidden):.4f}")
        
        st.divider()
        
        # Logits 分析
        st.subheader("🎯 Logits 到概率的转换")
        
        col1, col2 = st.columns([3, 2])
        
        with col1:
            st.markdown("""
            **lm_head**: 线性层，将隐藏状态映射到词表
            
            `Logits = W @ hidden_state + b`
            
            然后通过 Softmax 转换为概率：
            
            `P(token_i) = exp(logit_i) / Σ exp(logit_j)`
            """)
            
            # Top-K Logits 可视化
            top_k = st.slider("显示 Top-K", 5, 50, 20, key="logits_topk")
            
            # 获取 top-k logits
            top_k_indices = np.argsort(journey.logits)[-top_k:][::-1]
            top_k_logits = journey.logits[top_k_indices]
            top_k_tokens = [vocab.get(idx, f"<{idx}>") for idx in top_k_indices]
            
            # Logits 条形图
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=top_k_tokens,
                y=top_k_logits,
                marker_color='skyblue',
                text=[f"{v:.2f}" for v in top_k_logits],
                textposition='auto'
            ))
            fig.update_layout(
                title=f"Top-{top_k} Logits",
                xaxis_title="Token",
                yaxis_title="Logit 值",
                height=400,
                xaxis_tickangle=-45
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Softmax 后的概率
            probs = F.softmax(torch.from_numpy(journey.logits), dim=0).numpy()
            top_k_probs = probs[top_k_indices]
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=top_k_tokens,
                y=top_k_probs,
                marker_color='lightcoral',
                text=[f"{v:.2%}" for v in top_k_probs],
                textposition='auto'
            ))
            fig.update_layout(
                title=f"Top-{top_k} 概率分布",
                xaxis_title="Token",
                yaxis_title="概率",
                height=400,
                xaxis_tickangle=-45
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("🏆 预测结果")
            
            st.markdown(f"**预测的下一个 Token**: **{journey.predicted_token}**")
            st.metric("置信度", f"{journey.prediction_probability:.2%}")
            
            st.markdown("---")
            st.markdown("### 📋 Top-10 预测")
            
            pred_table = []
            for rank, (token_id, token_text, prob) in enumerate(journey.top_k_predictions[:10], 1):
                pred_table.append({
                    "#": rank,
                    "Token": token_text,
                    "概率": f"{prob:.2%}"
                })
            
            st.dataframe(
                pd.DataFrame(pred_table),
                hide_index=True,
                use_container_width=True
            )
        
        st.divider()
        
        # Logits 分布分析
        st.subheader("📊 Logits 分布分析")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Logits 直方图
            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=journey.logits,
                nbinsx=100,
                marker_color='lightgreen'
            ))
            fig.update_layout(
                title="所有 Logits 的分布",
                xaxis_title="Logit 值",
                yaxis_title="频次",
                height=350
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # 统计
            st.markdown(f"""
            **Logits 统计**:
            - 均值: {np.mean(journey.logits):.4f}
            - 标准差: {np.std(journey.logits):.4f}
            - 最大值: {np.max(journey.logits):.4f}
            - 最小值: {np.min(journey.logits):.4f}
            """)
        
        with col2:
            # 概率分布（整体）
            all_probs = F.softmax(torch.from_numpy(journey.logits), dim=0).numpy()
            
            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=all_probs,
                nbinsx=100,
                marker_color='lightpink'
            ))
            fig.update_layout(
                title="所有概率的分布",
                xaxis_title="概率",
                yaxis_title="频次",
                xaxis_type="log",
                height=350
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # 概率集中度
            top_10_prob_sum = np.sum(all_probs[np.argsort(journey.logits)[-10:]])
            st.markdown(f"""
            **概率集中度**:
            - Top-10 概率和: {top_10_prob_sum:.2%}
            - 有效词表大小: {1/np.sum(all_probs**2):.0f} 
            """)
            
            st.info("💡 有效词表大小（Perplexity）衡量预测的不确定性")
        
        # 完整旅程总结
        st.divider()
        st.success(f"""
        ### 🎉 Token 旅程完成！
        
        从 **'{journey.token_text}'** 到预测 **'{journey.predicted_token}'** 的完整流程：
        
        1. **输入嵌入** ({d_model}维) → 范数: {np.linalg.norm(journey.embedding):.4f}
        2. **{n_layers} 层 Transformer** → 逐层修正和提取特征
        3. **最终隐藏状态** ({d_model}维) → 范数: {np.linalg.norm(journey.final_hidden):.4f}
        4. **lm_head 投影** → {vocab_size} 个 Logits
        5. **Softmax** → 概率分布
        6. **预测** → {journey.predicted_token} (置信度: {journey.prediction_probability:.2%})
        
        **累积变化**: {np.linalg.norm(journey.final_hidden - journey.embedding):.4f}
        """)

else:
    # 未开始追踪时的说明
    st.info("👈 请在左侧配置模型参数并点击 '🚀 开始追踪' 按钮")
    
    st.markdown("""
    ## 🎯 功能介绍
    
    这个工具可以帮助你深入理解 Transformer 的工作原理：
    
    ### 1️⃣ **逐层追踪**
    - 查看 Token 在每一层的隐藏状态
    - 观察向量的变化和演化
    - 理解信息如何被逐步提取和转换
    
    ### 2️⃣ **残差连接分析**
    - **核心问题**: 残差连接如何"修正"而非"替换"？
    - 可视化每一层的残差贡献
    - 对比有无残差连接的差异
    - 理解梯度流的稳定性
    
    ### 3️⃣ **注意力权重**
    - 查看 Token 对序列中其他 Token 的注意力
    - 理解上下文信息的聚合
    
    ### 4️⃣ **Logits 分析**
    - 从最终隐藏状态到词表概率
    - 理解 lm_head 的作用
    - 查看预测的置信度
    
    ---
    
    **💡 提示**: 选择较小的模型（较少的层和维度）可以更快地看到结果
    """)

st.markdown("---")
st.caption("🚀 Token Journey Tracker | Transformer Explorer © 2025")
