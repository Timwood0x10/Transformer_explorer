"""
注意力模式可视化工具：展示Transformer注意力机制的工作原理
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List, Tuple, Optional


class AttentionVisualizer:
    """注意力模式可视化器"""
    
    def __init__(self, d_model: int = 512, n_heads: int = 8):
        """
        Args:
            d_model: 模型维度
            n_heads: 注意力头数
        """
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        
        # 创建示例模型
        self.model = self._create_sample_model()
        
        # 示例文本
        self.sample_texts = {
            "machine_translation": [
                "The cat sat on the mat",
                "猫 坐在 垫子 上"
            ],
            "text_summarization": [
                "The quick brown fox jumps over the lazy dog and runs away",
                "狐狸跳过懒狗"
            ],
            "question_answering": [
                "What is the capital of France? Paris is the capital of France",
                "法国 的 首都 是 什么"
            ]
        }
    
    def _create_sample_model(self) -> nn.Module:
        """创建示例Transformer模型"""
        class SimpleAttention(nn.Module):
            def __init__(self, d_model, n_heads):
                super().__init__()
                self.d_model = d_model
                self.n_heads = n_heads
                self.head_dim = d_model // n_heads
                
                self.q_proj = nn.Linear(d_model, d_model)
                self.k_proj = nn.Linear(d_model, d_model)
                self.v_proj = nn.Linear(d_model, d_model)
                self.out_proj = nn.Linear(d_model, d_model)
                
            def forward(self, x):
                B, L, D = x.shape
                
                # 计算Q, K, V
                Q = self.q_proj(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
                K = self.k_proj(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
                V = self.v_proj(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
                
                # 计算注意力分数
                scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_dim)
                attn_weights = F.softmax(scores, dim=-1)
                
                # 应用注意力
                attn_output = torch.matmul(attn_weights, V)
                attn_output = attn_output.transpose(1, 2).contiguous().view(B, L, D)
                
                # 输出投影
                output = self.out_proj(attn_output)
                
                return output, attn_weights
        
        return SimpleAttention(self.d_model, self.n_heads)
    
    def generate_attention_patterns(self, text_type: str = "machine_translation") -> Dict:
        """生成不同类型的注意力模式"""
        texts = self.sample_texts[text_type]
        
        # 模拟编码
        tokens = texts[0].split()
        seq_len = len(tokens)
        
        # 创建随机输入
        x = torch.randn(1, seq_len, self.d_model)
        
        # 获取注意力权重
        with torch.no_grad():
            output, attn_weights = self.model(x)
        
        # 提取不同头的注意力模式
        patterns = {}
        for head in range(self.n_heads):
            head_attn = attn_weights[0, head].cpu().numpy()
            
            # 分析注意力模式类型
            pattern_type = self._classify_attention_pattern(head_attn, text_type)
            
            patterns[f"head_{head}"] = {
                "weights": head_attn,
                "pattern_type": pattern_type,
                "description": self._get_pattern_description(pattern_type)
            }
        
        return {
            "tokens": tokens,
            "patterns": patterns,
            "text_type": text_type
        }
    
    def _classify_attention_pattern(self, attn_matrix: np.ndarray, text_type: str) -> str:
        """分类注意力模式类型"""
        # 计算注意力特征
        diagonal_strength = np.mean(np.diag(attn_matrix))
        max_off_diagonal = np.max(attn_matrix - np.diag(np.diag(attn_matrix)))
        entropy = -np.sum(attn_matrix * np.log(attn_matrix + 1e-8))
        
        if text_type == "machine_translation":
            if diagonal_strength > 0.5:
                return "diagonal_alignment"
            elif max_off_diagonal > 0.7:
                return "cross_attention"
            else:
                return "uniform_attention"
        
        elif text_type == "text_summarization":
            if entropy < 2.0:
                return "focused_attention"
            else:
                return "distributed_attention"
        
        elif text_type == "question_answering":
            if np.max(attn_matrix[:, 0]) > 0.8:  # 第一列（问题）被高度关注
                return "question_focus"
            else:
                return "answer_extraction"
        
        return "mixed_pattern"
    
    def _get_pattern_description(self, pattern_type: str) -> str:
        """获取模式描述"""
        descriptions = {
            "diagonal_alignment": "对齐模式 - 源语言和目标语言词序对应",
            "cross_attention": "交叉注意力 - 跨语言词对齐",
            "uniform_attention": "均匀注意力 - 所有位置平等关注",
            "focused_attention": "聚焦注意力 - 关注关键信息",
            "distributed_attention": "分布式注意力 - 信息分散在多处",
            "question_focus": "问题聚焦 - 高度关注问题部分",
            "answer_extraction": "答案提取 - 从上下文中提取答案",
            "mixed_pattern": "混合模式 - 多种注意力模式组合"
        }
        return descriptions.get(pattern_type, "未知模式")
    
    def visualize_attention_heatmap(self, head_idx: int = 0, text_type: str = "machine_translation") -> go.Figure:
        """可视化注意力热力图"""
        data = self.generate_attention_patterns(text_type)
        tokens = data["tokens"]
        patterns = data["patterns"]
        
        if f"head_{head_idx}" not in patterns:
            return go.Figure()
        
        attn_weights = patterns[f"head_{head_idx}"]["weights"]
        pattern_type = patterns[f"head_{head_idx}"]["pattern_type"]
        description = patterns[f"head_{head_idx}"]["description"]
        
        # 创建热力图
        fig = go.Figure(data=go.Heatmap(
            z=attn_weights,
            x=tokens,
            y=tokens,
            colorscale='Blues',
            showscale=True,
            hoverongaps=False,
            hovertemplate='从 %{y} 到 %{x}<br>注意力权重: %{z:.3f}<extra></extra>'
        ))
        
        fig.update_layout(
            title=f'注意力模式 - 头 {head_idx} ({pattern_type})<br>{description}',
            xaxis_title='目标位置',
            yaxis_title='源位置',
            height=500,
            width=600
        )
        
        return fig
    
    def visualize_multi_head_attention(self, text_type: str = "machine_translation") -> go.Figure:
        """可视化多头注意力"""
        data = self.generate_attention_patterns(text_type)
        tokens = data["tokens"]
        patterns = data["patterns"]
        
        # 创建子图
        n_cols = 4
        n_rows = (self.n_heads + n_cols - 1) // n_cols
        
        fig = go.Figure()
        
        for head in range(self.n_heads):
            if f"head_{head}" not in patterns:
                continue
                
            attn_weights = patterns[f"head_{head}"]["weights"]
            pattern_type = patterns[f"head_{head}"]["pattern_type"]
            
            row = head // n_cols + 1
            col = head % n_cols + 1
            
            fig.add_trace(go.Heatmap(
                z=attn_weights,
                x=tokens,
                y=tokens,
                colorscale='Blues',
                showscale=False,
                name=f'Head {head}',
                hovertemplate=f'头 {head}: %{{z:.3f}}<extra></extra>'
            ), row=row, col=col)
        
        fig.update_layout(
            title=f'多头注意力模式 - {text_type}',
            height=300*n_rows,
            width=800,
            showlegend=False
        )
        
        # 更新所有子图
        for i in range(1, self.n_heads + 1):
            fig.update_xaxes(title_text="目标位置", row=i, col=1)
            fig.update_yaxes(title_text="源位置", row=i, col=1)
        
        return fig
    
    def create_attention_animation(self, text_type: str = "machine_translation") -> go.Figure:
        """创建注意力演化动画"""
        data = self.generate_attention_patterns(text_type)
        tokens = data["tokens"]
        patterns = data["patterns"]
        
        # 创建动画帧
        frames = []
        for head in range(self.n_heads):
            if f"head_{head}" not in patterns:
                continue
                
            attn_weights = patterns[f"head_{head}"]["weights"]
            pattern_type = patterns[f"head_{head}"]["pattern_type"]
            
            frame = go.Frame(
                data=[go.Heatmap(
                    z=attn_weights,
                    x=tokens,
                    y=tokens,
                    colorscale='Blues',
                    showscale=True
                )],
                name=f'Head {head}'
            )
            frames.append(frame)
        
        # 创建初始图
        fig = go.Figure(
            data=[go.Heatmap(
                z=patterns["head_0"]["weights"],
                x=tokens,
                y=tokens,
                colorscale='Blues',
                showscale=True
            )],
            frames=frames
        )
        
        # 添加播放按钮
        fig.update_layout(
            title='注意力模式动画',
            height=500,
            width=600,
            updatemenus=[{
                'type': 'buttons',
                'showactive': False,
                'buttons': [
                    {
                        'label': '播放',
                        'method': 'animate',
                        'args': [None, {
                            'frame': {'duration': 1000, 'redraw': True},
                            'fromcurrent': True,
                            'transition': {'duration': 500}
                        }]
                    },
                    {
                        'label': '暂停',
                        'method': 'animate',
                        'args': [[None], {
                            'frame': {'duration': 0, 'redraw': False},
                            'mode': 'immediate',
                            'transition': {'duration': 0}
                        }]
                    }
                ]
            }]
        )
        
        return fig
    
    def analyze_attention_diversity(self, text_type: str = "machine_translation") -> Dict:
        """分析注意力头的多样性"""
        data = self.generate_attention_patterns(text_type)
        patterns = data["patterns"]
        
        # 计算头之间的相似度
        similarities = {}
        for i in range(self.n_heads):
            for j in range(i+1, self.n_heads):
                if f"head_{i}" in patterns and f"head_{j}" in patterns:
                    attn_i = patterns[f"head_{i}"]["weights"].flatten()
                    attn_j = patterns[f"head_{j}"]["weights"].flatten()
                    
                    # 计算余弦相似度
                    similarity = np.dot(attn_i, attn_j) / (np.linalg.norm(attn_i) * np.linalg.norm(attn_j) + 1e-8)
                    similarities[f"head_{i}_vs_head_{j}"] = similarity
        
        # 分析模式类型分布
        pattern_counts = {}
        for head_data in patterns.values():
            pattern_type = head_data["pattern_type"]
            pattern_counts[pattern_type] = pattern_counts.get(pattern_type, 0) + 1
        
        return {
            "similarities": similarities,
            "pattern_distribution": pattern_counts,
            "diversity_score": len(pattern_counts) / self.n_heads
        }
    
    def create_attention_summary_report(self) -> str:
        """生成注意力分析报告"""
        report = "# 注意力模式分析报告\n\n"
        
        # 分析不同任务类型的注意力模式
        for task_type in self.sample_texts.keys():
            report += f"## {task_type.replace('_', ' ').title()} 任务\n"
            
            diversity = self.analyze_attention_diversity(task_type)
            pattern_dist = diversity["pattern_distribution"]
            
            report += f"### 模式分布\n"
            for pattern, count in pattern_dist.items():
                report += f"- {pattern}: {count} 个头\n"
            
            report += f"### 多样性评分: {diversity['diversity_score']:.2f}\n"
            
            # 找出最相似的头对
            similarities = diversity["similarities"]
            if similarities:
                most_similar = max(similarities.items(), key=lambda x: x[1])
                report += f"### 最相似的头对: {most_similar[0]} (相似度: {most_similar[1]:.3f})\n"
            
            report += "\n"
        
        # 教学要点
        report += """
## 教学要点

### 🎯 注意力机制的核心概念
1. **多头注意力**: 不同的头学习不同的注意力模式
2. **任务特异性**: 不同任务需要不同的注意力策略
3. **模式多样性**: 高多样性通常意味着模型能捕获更丰富的信息

### 🔍 常见注意力模式
- **对齐模式**: 翻译任务中的词对齐
- **聚焦模式**: 摘要任务中的关键信息提取
- **问答模式**: 问答任务中的问题-答案匹配

### 💡 优化建议
- 如果所有头的模式相似，考虑增加头的多样性
- 如果注意力过于分散，可能需要调整温度参数
- 特定任务可以设计专门的注意力偏置
"""
        
        return report


if __name__ == "__main__":
    # 测试代码
    visualizer = AttentionVisualizer(d_model=512, n_heads=8)
    
    # 生成注意力模式
    patterns = visualizer.generate_attention_patterns("machine_translation")
    print(f"生成了 {len(patterns['patterns'])} 个注意力头的模式")
    
    # 分析多样性
    diversity = visualizer.analyze_attention_diversity("machine_translation")
    print(f"注意力多样性评分: {diversity['diversity_score']:.2f}")
    
    # 生成报告
    report = visualizer.create_attention_summary_report()
    print(report)
