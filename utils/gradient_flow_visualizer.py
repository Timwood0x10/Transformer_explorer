"""
梯度流可视化工具：理解和分析深度学习中的梯度传播机制
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List, Tuple, Optional
from collections import defaultdict


class GradientFlowVisualizer:
    """梯度流可视化器"""
    
    def __init__(self):
        """初始化梯度流可视化器"""
        self.gradient_history = defaultdict(list)
        self.activation_history = defaultdict(list)
        self.weight_history = defaultdict(list)
        
    def create_sample_networks(self) -> Dict[str, nn.Module]:
        """创建不同类型的示例网络"""
        networks = {}
        
        # 1. 简单的深度网络
        class DeepNetwork(nn.Module):
            def __init__(self, layer_sizes, activation='relu'):
                super().__init__()
                self.layers = nn.ModuleList()
                self.activation = activation
                
                for i in range(len(layer_sizes) - 1):
                    self.layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            
            def forward(self, x):
                for i, layer in enumerate(self.layers):
                    x = layer(x)
                    if i < len(self.layers) - 1:  # 不在最后一层激活
                        if self.activation == 'relu':
                            x = F.relu(x)
                        elif self.activation == 'tanh':
                            x = torch.tanh(x)
                        elif self.activation == 'sigmoid':
                            x = torch.sigmoid(x)
                return x
        
        # 2. 带残差连接的网络
        class ResidualNetwork(nn.Module):
            def __init__(self, layer_sizes):
                super().__init__()
                self.layers = nn.ModuleList()
                self.shortcuts = nn.ModuleList()
                
                for i in range(len(layer_sizes) - 1):
                    self.layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
                    if i > 0 and layer_sizes[i] == layer_sizes[i+1]:
                        self.shortcuts.append(nn.Identity())
                    else:
                        self.shortcuts.append(None)
            
            def forward(self, x):
                for i, layer in enumerate(self.layers):
                    out = layer(x)
                    if self.shortcuts[i] is not None:
                        out = out + x
                    x = F.relu(out)
                return x
        
        # 3. LSTM网络
        class LSTMLayer(nn.Module):
            def __init__(self, input_size, hidden_size, num_layers=2):
                super().__init__()
                self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
                self.output_layer = nn.Linear(hidden_size, hidden_size)
            
            def forward(self, x):
                lstm_out, _ = self.lstm(x)
                return self.output_layer(lstm_out)
        
        # 创建网络实例
        networks['deep_relu'] = DeepNetwork([512, 256, 128, 64, 32, 16, 8, 4, 2], 'relu')
        networks['deep_tanh'] = DeepNetwork([512, 256, 128, 64, 32, 16, 8, 4, 2], 'tanh')
        networks['deep_sigmoid'] = DeepNetwork([512, 256, 128, 64, 32, 16, 8, 4, 2], 'sigmoid')
        networks['residual'] = ResidualNetwork([512, 256, 256, 128, 128, 64, 64, 32])
        networks['lstm'] = LSTMLayer(512, 256, 3)
        
        return networks
    
    def analyze_gradient_flow(self, network: nn.Module, input_size: Tuple[int, int], 
                            num_batches: int = 10) -> Dict:
        """分析网络的梯度流"""
        network.train()
        
        gradient_stats = defaultdict(list)
        activation_stats = defaultdict(list)
        
        for batch in range(num_batches):
            # 生成随机输入
            x = torch.randn(input_size[0], input_size[1])
            target = torch.randn(input_size[0], 1)
            
            # 前向传播
            network.zero_grad()
            output = network(x)
            
            # 记录激活值
            self._record_activations(network, activation_stats, batch)
            
            # 计算损失并反向传播
            loss = F.mse_loss(output.mean(dim=1, keepdim=True), target)
            loss.backward()
            
            # 记录梯度
            self._record_gradients(network, gradient_stats, batch)
        
        # 计算统计信息
        gradient_analysis = self._compute_gradient_stats(gradient_stats)
        activation_analysis = self._compute_activation_stats(activation_stats)
        
        return {
            'gradient_stats': gradient_analysis,
            'activation_stats': activation_analysis,
            'network_type': type(network).__name__
        }
    
    def _record_activations(self, network: nn.Module, stats: Dict, batch: int):
        """记录激活值"""
        for name, module in network.named_modules():
            if isinstance(module, (nn.Linear, nn.LSTM)):
                if hasattr(module, 'output') and module.output is not None:
                    activation = module.output.detach()
                    stats[name].append({
                        'batch': batch,
                        'mean': activation.mean().item(),
                        'std': activation.std().item(),
                        'min': activation.min().item(),
                        'max': activation.max().item(),
                        'shape': activation.shape
                    })
    
    def _record_gradients(self, network: nn.Module, stats: Dict, batch: int):
        """记录梯度"""
        for name, param in network.named_parameters():
            if param.grad is not None:
                grad = param.grad.detach()
                stats[name].append({
                    'batch': batch,
                    'mean': grad.mean().item(),
                    'std': grad.std().item(),
                    'norm': grad.norm().item(),
                    'max': grad.max().item(),
                    'min': grad.min().item()
                })
    
    def _compute_gradient_stats(self, gradient_data: Dict) -> Dict:
        """计算梯度统计信息"""
        stats = {}
        
        for name, batches in gradient_data.items():
            if not batches:
                continue
            
            # 提取各批次的梯度范数
            norms = [b['norm'] for b in batches]
            means = [b['mean'] for b in batches]
            stds = [b['std'] for b in batches]
            
            stats[name] = {
                'avg_norm': np.mean(norms),
                'std_norm': np.std(norms),
                'min_norm': np.min(norms),
                'max_norm': np.max(norms),
                'avg_mean': np.mean(means),
                'avg_std': np.mean(stds),
                'gradient_health': self._assess_gradient_health(norms)
            }
        
        return stats
    
    def _compute_activation_stats(self, activation_data: Dict) -> Dict:
        """计算激活值统计信息"""
        stats = {}
        
        for name, batches in activation_data.items():
            if not batches:
                continue
            
            means = [b['mean'] for b in batches]
            stds = [b['std'] for b in batches]
            
            stats[name] = {
                'avg_mean': np.mean(means),
                'avg_std': np.std(stds),
                'activation_health': self._assess_activation_health(means, stds)
            }
        
        return stats
    
    def _assess_gradient_health(self, norms: List[float]) -> str:
        """评估梯度健康状况"""
        if len(norms) == 0:
            return "无数据"
        
        avg_norm = np.mean(norms)
        min_norm = np.min(norms)
        
        if min_norm < 1e-7:
            return "梯度消失"
        elif avg_norm > 10:
            return "梯度爆炸"
        elif np.std(norms) / avg_norm > 2:
            return "梯度不稳定"
        else:
            return "健康"
    
    def _assess_activation_health(self, means: List[float], stds: List[float]) -> str:
        """评估激活值健康状况"""
        if len(means) == 0:
            return "无数据"
        
        avg_mean = np.abs(np.mean(means))
        avg_std = np.mean(stds)
        
        if avg_mean < 0.01 and avg_std < 0.01:
            return "激活饱和"
        elif avg_std < 0.1:
            return "激活稀疏"
        else:
            return "正常"
    
    def visualize_gradient_flow(self, network_name: str = 'deep_relu') -> go.Figure:
        """可视化梯度流"""
        networks = self.create_sample_networks()
        
        if network_name not in networks:
            network_name = 'deep_relu'
        
        network = networks[network_name]
        
        # 分析梯度流
        input_size = (32, 512)  # batch_size=32, input_dim=512
        analysis = self.analyze_gradient_flow(network, input_size)
        
        # 提取梯度范数
        layer_names = []
        gradient_norms = []
        gradient_health = []
        
        for name, stats in analysis['gradient_stats'].items():
            if 'weight' in name:  # 只看权重参数
                layer_names.append(name.replace('.weight', ''))
                gradient_norms.append(stats['avg_norm'])
                gradient_health.append(stats['gradient_health'])
        
        # 创建可视化
        fig = go.Figure()
        
        # 添加梯度范数线
        fig.add_trace(go.Scatter(
            x=list(range(len(layer_names))),
            y=gradient_norms,
            mode='lines+markers',
            name='梯度范数',
            line=dict(width=3),
            marker=dict(size=8)
        ))
        
        # 添加健康状况标记
        health_colors = {'健康': 'green', '梯度消失': 'red', '梯度爆炸': 'orange', '梯度不稳定': 'purple'}
        for i, health in enumerate(gradient_health):
            if health != '健康':
                fig.add_annotation(
                    x=i, y=gradient_norms[i],
                    text=health,
                    showarrow=True,
                    arrowhead=2,
                    arrowsize=1,
                    arrowwidth=2,
                    arrowcolor=health_colors.get(health, 'black'),
                    ax=0, ay=-40
                )
        
        fig.update_layout(
            title=f'梯度流分析 - {network_name}',
            xaxis_title='网络层（从输入到输出）',
            yaxis_title='梯度范数（对数尺度）',
            yaxis_type='log',
            height=500,
            xaxis=dict(tickmode='array', tickvals=list(range(len(layer_names))), ticktext=layer_names)
        )
        
        return fig
    
    def compare_activation_functions(self) -> go.Figure:
        """对比不同激活函数的梯度流"""
        networks = self.create_sample_networks()
        input_size = (32, 512)
        
        fig = go.Figure()
        
        activation_functions = ['relu', 'tanh', 'sigmoid']
        colors = ['blue', 'red', 'green']
        
        for i, activation in enumerate(activation_functions):
            network_name = f'deep_{activation}'
            if network_name in networks:
                network = networks[network_name]
                analysis = self.analyze_gradient_flow(network, input_size)
                
                # 提取梯度范数
                gradient_norms = []
                for name, stats in analysis['gradient_stats'].items():
                    if 'weight' in name:
                        gradient_norms.append(stats['avg_norm'])
                
                fig.add_trace(go.Scatter(
                    x=list(range(len(gradient_norms))),
                    y=gradient_norms,
                    mode='lines+markers',
                    name=activation.upper(),
                    line=dict(color=colors[i], width=2),
                    marker=dict(size=6)
                ))
        
        fig.update_layout(
            title='不同激活函数的梯度流对比',
            xaxis_title='网络层深度',
            yaxis_title='梯度范数（对数尺度）',
            yaxis_type='log',
            height=500,
            legend=dict(x=0.02, y=0.98)
        )
        
        return fig
    
    def visualize_residual_connections(self) -> go.Figure:
        """可视化残差连接对梯度流的影响"""
        networks = self.create_sample_networks()
        input_size = (32, 512)
        
        fig = go.Figure()
        
        # 对比有残差连接和没有残差连接的网络
        network_types = ['deep_relu', 'residual']
        labels = ['无残差连接', '有残差连接']
        colors = ['red', 'green']
        
        for i, (network_type, label) in enumerate(zip(network_types, labels)):
            if network_type in networks:
                network = networks[network_type]
                analysis = self.analyze_gradient_flow(network, input_size)
                
                # 提取梯度范数
                gradient_norms = []
                for name, stats in analysis['gradient_stats'].items():
                    if 'weight' in name:
                        gradient_norms.append(stats['avg_norm'])
                
                fig.add_trace(go.Scatter(
                    x=list(range(len(gradient_norms))),
                    y=gradient_norms,
                    mode='lines+markers',
                    name=label,
                    line=dict(color=colors[i], width=2),
                    marker=dict(size=6)
                ))
        
        fig.update_layout(
            title='残差连接对梯度流的影响',
            xaxis_title='网络层深度',
            yaxis_title='梯度范数（对数尺度）',
            yaxis_type='log',
            height=500
        )
        
        return fig
    
    def create_gradient_flow_report(self) -> str:
        """生成梯度流分析报告"""
        networks = self.create_sample_networks()
        input_size = (32, 512)
        
        report = "# 梯度流分析报告\n\n"
        
        # 分析每种网络类型
        for network_name, network in networks.items():
            report += f"## {network_name.replace('_', ' ').title()}\n"
            
            analysis = self.analyze_gradient_flow(network, input_size)
            
            # 统计梯度健康状况
            health_counts = {}
            for stats in analysis['gradient_stats'].values():
                health = stats['gradient_health']
                health_counts[health] = health_counts.get(health, 0) + 1
            
            report += "### 梯度健康状况分布\n"
            for health, count in health_counts.items():
                report += f"- {health}: {count} 层\n"
            
            # 找出问题层
            problem_layers = []
            for name, stats in analysis['gradient_stats'].items():
                if stats['gradient_health'] != '健康':
                    problem_layers.append(f"{name}: {stats['gradient_health']}")
            
            if problem_layers:
                report += "### 问题层\n"
                for layer in problem_layers:
                    report += f"- {layer}\n"
            else:
                report += "### ✅ 所有层梯度健康\n"
            
            report += "\n"
        
        # 教学内容
        report += """
## 梯度流问题与解决方案

### 🔴 常见问题

#### 1. 梯度消失 (Vanishing Gradients)
**症状**: 浅层梯度接近零，深层网络难以训练
**原因**: 
- sigmoid/tanh激活函数的导数在输入绝对值大时接近0
- 网络过深，梯度连乘导致指数衰减
- 权重初始化不当

**解决方案**:
- 使用ReLU等非饱和激活函数
- 残差连接 (ResNet)
- 批归一化 (Batch Normalization)
- 合适的权重初始化 (Xavier/Kaiming)

#### 2. 梯度爆炸 (Exploding Gradients)
**症状**: 梯度值过大，训练不稳定
**原因**:
- 学习率过高
- 权重初始化方差过大
- RNN中的长时间依赖

**解决方案**:
- 梯度裁剪 (Gradient Clipping)
- 降低学习率
- 权重正则化
- LSTM/GRU结构

#### 3. 梯度不稳定 (Unstable Gradients)
**症状**: 梯度方差大，训练震荡
**原因**:
- 批大小过小
- 学习率调度不当
- 数据预处理问题

**解决方案**:
- 增大批大小
- 学习率预热 (Warmup)
- 梯度累积
- 自适应优化器 (Adam, RMSprop)

### 🎯 最佳实践

1. **激活函数选择**
   - 深度网络优先使用ReLU及其变体
   - 输出层根据任务选择合适的激活函数
   - 注意ReLU的"死亡"问题

2. **网络设计**
   - 深层网络考虑残差连接
   - 使用批归一化稳定训练
   - 合理的网络深度，避免过深

3. **训练技巧**
   - 监控梯度范数变化
   - 使用梯度裁剪防止爆炸
   - 适当的学习率调度

4. **诊断工具**
   - 定期检查梯度分布
   - 监控激活值范围
   - 使用TensorBoard等可视化工具
"""
        
        return report


if __name__ == "__main__":
    # 测试代码
    visualizer = GradientFlowVisualizer()
    
    # 创建示例网络
    networks = visualizer.create_sample_networks()
    print(f"创建了 {len(networks)} 种不同类型的网络")
    
    # 分析梯度流
    network = networks['deep_relu']
    analysis = visualizer.analyze_gradient_flow(network, (32, 512))
    print(f"分析了 {len(analysis['gradient_stats'])} 个参数的梯度")
    
    # 生成报告
    report = visualizer.create_gradient_flow_report()
    print(report)
