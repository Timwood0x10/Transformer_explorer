"""
参数初始化对比工具：演示不同初始化方法对模型训练的影响
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
from scipy import stats


class InitializationComparator:
    """参数初始化方法对比器"""
    
    def __init__(self, layer_sizes: List[int] = [512, 256, 128, 64, 10]):
        """
        Args:
            layer_sizes: 网络各层的大小
        """
        self.layer_sizes = layer_sizes
        self.init_methods = {
            "xavier_uniform": self._xavier_uniform_init,
            "xavier_normal": self._xavier_normal_init,
            "kaiming_uniform": self._kaiming_uniform_init,
            "kaiming_normal": self._kaiming_normal_init,
            "orthogonal": self._orthogonal_init,
            "lecun_normal": self._lecun_normal_init,
            "random_normal": self._random_normal_init,
            "random_uniform": self._random_uniform_init
        }
        
        # 初始化结果存储
        self.init_results = {}
        
    def _xavier_uniform_init(self, weight: torch.Tensor, gain: float = 1.0) -> torch.Tensor:
        """Xavier均匀初始化"""
        return nn.init.xavier_uniform_(weight, gain=gain)
    
    def _xavier_normal_init(self, weight: torch.Tensor, gain: float = 1.0) -> torch.Tensor:
        """Xavier正态初始化"""
        return nn.init.xavier_normal_(weight, gain=gain)
    
    def _kaiming_uniform_init(self, weight: torch.Tensor, a: float = 0, mode: str = 'fan_in', nonlinearity: str = 'relu') -> torch.Tensor:
        """Kaiming均匀初始化"""
        return nn.init.kaiming_uniform_(weight, a=a, mode=mode, nonlinearity=nonlinearity)
    
    def _kaiming_normal_init(self, weight: torch.Tensor, a: float = 0, mode: str = 'fan_in', nonlinearity: str = 'relu') -> torch.Tensor:
        """Kaiming正态初始化"""
        return nn.init.kaiming_normal_(weight, a=a, mode=mode, nonlinearity=nonlinearity)
    
    def _orthogonal_init(self, weight: torch.Tensor, gain: float = 1.0) -> torch.Tensor:
        """正交初始化"""
        return nn.init.orthogonal_(weight, gain=gain)
    
    def _lecun_normal_init(self, weight: torch.Tensor) -> torch.Tensor:
        """LeCun正态初始化"""
        fan_in = weight.size(1)
        std = np.sqrt(1.0 / fan_in)
        return nn.init.normal_(weight, 0, std)
    
    def _random_normal_init(self, weight: torch.Tensor, mean: float = 0.0, std: float = 0.02) -> torch.Tensor:
        """随机正态初始化"""
        return nn.init.normal_(weight, mean, std)
    
    def _random_uniform_init(self, weight: torch.Tensor, a: float = -0.05, b: float = 0.05) -> torch.Tensor:
        """随机均匀初始化"""
        return nn.init.uniform_(weight, a, b)
    
    def create_sample_network(self) -> nn.Module:
        """创建示例网络"""
        layers = []
        for i in range(len(self.layer_sizes) - 1):
            layers.append(nn.Linear(self.layer_sizes[i], self.layer_sizes[i+1]))
            if i < len(self.layer_sizes) - 2:  # 不在最后一层添加激活
                layers.append(nn.ReLU())
        
        return nn.Sequential(*layers)
    
    def analyze_initialization(self, init_method: str, n_samples: int = 1000) -> Dict:
        """分析特定初始化方法"""
        if init_method not in self.init_methods:
            raise ValueError(f"未知的初始化方法: {init_method}")
        
        # 创建网络
        net = self.create_sample_network()
        
        # 应用初始化
        init_func = self.init_methods[init_method]
        for module in net.modules():
            if isinstance(module, nn.Linear):
                init_func(module.weight)
                nn.init.zeros_(module.bias)
        
        # 生成随机输入
        x = torch.randn(n_samples, self.layer_sizes[0])
        
        # 前向传播分析
        activations = {}
        with torch.no_grad():
            current_input = x
            layer_idx = 0
            
            for i, module in enumerate(net.modules()):
                if isinstance(module, nn.Linear):
                    # 记录输入分布
                    activations[f'layer_{layer_idx}_input'] = {
                        'mean': current_input.mean().item(),
                        'std': current_input.std().item(),
                        'min': current_input.min().item(),
                        'max': current_input.max().item(),
                        'shape': current_input.shape
                    }
                    
                    # 前向传播
                    output = module(current_input)
                    
                    # 记录输出分布
                    activations[f'layer_{layer_idx}_output'] = {
                        'mean': output.mean().item(),
                        'std': output.std().item(),
                        'min': output.min().item(),
                        'max': output.max().item(),
                        'shape': output.shape
                    }
                    
                    # 记录权重分布
                    weight = module.weight
                    activations[f'layer_{layer_idx}_weight'] = {
                        'mean': weight.mean().item(),
                        'std': weight.std().item(),
                        'min': weight.min().item(),
                        'max': weight.max().item(),
                        'shape': weight.shape,
                        'frobenius_norm': weight.norm().item()
                    }
                    
                    current_input = output
                    layer_idx += 1
                
                elif isinstance(module, nn.ReLU):
                    current_input = module(current_input)
        
        # 计算梯度流（模拟）
        gradient_stats = self._analyze_gradient_flow(net, x)
        
        return {
            'method': init_method,
            'activations': activations,
            'gradient_stats': gradient_stats,
            'network': net
        }
    
    def _analyze_gradient_flow(self, net: nn.Module, x: torch.Tensor) -> Dict:
        """分析梯度流"""
        # 创建损失和反向传播
        output = net(x)
        loss = output.mean()
        loss.backward()
        
        gradient_stats = {}
        for name, param in net.named_parameters():
            if param.grad is not None:
                grad = param.grad
                gradient_stats[name] = {
                    'mean': grad.mean().item(),
                    'std': grad.std().item(),
                    'min': grad.min().item(),
                    'max': grad.max().item(),
                    'norm': grad.norm().item()
                }
        
        return gradient_stats
    
    def compare_all_initializations(self) -> Dict:
        """对比所有初始化方法"""
        results = {}
        
        for method in self.init_methods.keys():
            print(f"分析初始化方法: {method}")
            results[method] = self.analyze_initialization(method)
        
        self.init_results = results
        return results
    
    def visualize_weight_distributions(self) -> go.Figure:
        """可视化不同初始化方法的权重分布"""
        if not self.init_results:
            self.compare_all_initializations()
        
        fig = go.Figure()
        
        for method, result in self.init_results.items():
            # 获取第一层的权重
            weight_key = f'layer_0_weight'
            if weight_key in result['activations']:
                weight_stats = result['activations'][weight_key]
                
                # 生成模拟分布（基于统计信息）
                samples = np.random.normal(
                    weight_stats['mean'], 
                    weight_stats['std'], 
                    1000
                )
                
                fig.add_trace(go.Histogram(
                    x=samples,
                    name=method,
                    opacity=0.7,
                    nbinsx=50
                ))
        
        fig.update_layout(
            title='不同初始化方法的权重分布对比（第一层）',
            xaxis_title='权重值',
            yaxis_title='频次',
            barmode='overlay',
            height=500
        )
        
        return fig
    
    def visualize_activation_evolution(self) -> go.Figure:
        """可视化激活值在各层的演化"""
        if not self.init_results:
            self.compare_all_initializations()
        
        fig = go.Figure()
        
        # 选择几个代表性方法
        representative_methods = ['xavier_normal', 'kaiming_normal', 'random_normal']
        colors = ['blue', 'red', 'green']
        rgba_colors = ['rgba(0,0,255,0.2)', 'rgba(255,0,0,0.2)', 'rgba(0,128,0,0.2)']
        
        for i, method in enumerate(representative_methods):
            if method in self.init_results:
                result = self.init_results[method]
                
                # 收集各层的激活统计
                layers = []
                means = []
                stds = []
                
                for key, stats in result['activations'].items():
                    if 'output' in key:
                        layer_num = int(key.split('_')[1])
                        layers.append(layer_num)
                        means.append(stats['mean'])
                        stds.append(stats['std'])
                
                # 添加均值线
                fig.add_trace(go.Scatter(
                    x=layers,
                    y=means,
                    mode='lines+markers',
                    name=f'{method} (均值)',
                    line=dict(color=colors[i], width=2),
                    legendgroup=method
                ))
                
                # 添加标准差区域
                upper = np.array(means) + np.array(stds)
                lower = np.array(means) - np.array(stds)
                
                fig.add_trace(go.Scatter(
                    x=layers + layers[::-1],
                    y=list(upper) + list(lower[::-1]),
                    fill='toself',
                    fillcolor=rgba_colors[i],
                    line=dict(color='rgba(255,255,255,0)'),
                    hoverinfo="skip",
                    showlegend=False,
                    legendgroup=method
                ))
        
        fig.update_layout(
            title='激活值在各层的演化',
            xaxis_title='层数',
            yaxis_title='激活值',
            height=500
        )
        
        return fig
    
    def visualize_gradient_flow(self) -> go.Figure:
        """可视化梯度流"""
        if not self.init_results:
            self.compare_all_initializations()
        
        fig = go.Figure()
        
        for method, result in self.init_results.items():
            gradient_stats = result['gradient_stats']
            
            # 收集梯度范数
            layer_names = []
            grad_norms = []
            
            for name, stats in gradient_stats.items():
                layer_names.append(name)
                grad_norms.append(stats['norm'])
            
            fig.add_trace(go.Scatter(
                x=list(range(len(grad_norms))),
                y=grad_norms,
                mode='lines+markers',
                name=method,
                line=dict(width=2)
            ))
        
        fig.update_layout(
            title='不同初始化方法的梯度流对比',
            xaxis_title='层数（从输入到输出）',
            yaxis_title='梯度范数（对数尺度）',
            yaxis_type='log',
            height=500
        )
        
        return fig
    
    def create_initialization_report(self) -> str:
        """生成初始化对比报告"""
        if not self.init_results:
            self.compare_all_initializations()
        
        report = "# 参数初始化方法对比报告\n\n"
        
        # 方法介绍
        report += "## 初始化方法介绍\n\n"
        
        method_descriptions = {
            "xavier_uniform": "Xavier均匀初始化 - 适用于tanh激活函数",
            "xavier_normal": "Xavier正态初始化 - 适用于tanh激活函数",
            "kaiming_uniform": "Kaiming均匀初始化 - 适用于ReLU激活函数",
            "kaiming_normal": "Kaiming正态初始化 - 适用于ReLU激活函数",
            "orthogonal": "正交初始化 - 保持梯度范数稳定",
            "lecun_normal": "LeCun正态初始化 - 适用于SELU激活函数",
            "random_normal": "随机正态初始化 - 简单的基准方法",
            "random_uniform": "随机均匀初始化 - 简单的基准方法"
        }
        
        for method, desc in method_descriptions.items():
            report += f"### {method}\n{desc}\n\n"
        
        # 性能对比
        report += "## 性能对比分析\n\n"
        
        for method, result in self.init_results.items():
            report += f"### {method}\n"
            
            # 分析激活值稳定性
            output_means = []
            output_stds = []
            
            for key, stats in result['activations'].items():
                if 'output' in key:
                    output_means.append(stats['mean'])
                    output_stds.append(stats['std'])
            
            # 检查是否有梯度消失或爆炸
            grad_norms = [stats['norm'] for stats in result['gradient_stats'].values()]
            min_grad_norm = min(grad_norms)
            max_grad_norm = max(grad_norms)
            
            report += f"- 激活值均值范围: [{min(output_means):.4f}, {max(output_means):.4f}]\n"
            report += f"- 激活值标准差范围: [{min(output_stds):.4f}, {max(output_stds):.4f}]\n"
            report += f"- 梯度范数范围: [{min_grad_norm:.6f}, {max_grad_norm:.6f}]\n"
            
            # 判断问题
            if min_grad_norm < 1e-6:
                report += "- ⚠️ 存在梯度消失风险\n"
            elif max_grad_norm > 10:
                report += "- ⚠️ 存在梯度爆炸风险\n"
            else:
                report += "- ✅ 梯度流相对稳定\n"
            
            report += "\n"
        
        # 推荐建议
        report += "## 推荐建议\n\n"
        report += """
### 🎯 根据激活函数选择初始化方法
- **ReLU/LeakyReLU**: 推荐使用Kaiming初始化
- **tanh/sigmoid**: 推荐使用Xavier初始化  
- **SELU**: 推荐使用LeCun初始化

### 🚀 特殊场景推荐
- **深度网络**: 正交初始化有助于保持梯度稳定
- **RNN/LSTM**: 正交初始化对循环门很重要
- **生成模型**: 有时需要更保守的初始化方差

### ⚠️ 常见陷阱
- 避免使用过大的初始化方差
- 注意偏置项的初始化（通常设为0）
- 批归归化层可以缓解初始化问题
"""
        
        return report


if __name__ == "__main__":
    # 测试代码
    comparator = InitializationComparator([512, 256, 128, 64, 10])
    
    # 对比所有初始化方法
    results = comparator.compare_all_initializations()
    print(f"对比了 {len(results)} 种初始化方法")
    
    # 生成报告
    report = comparator.create_initialization_report()
    print(report)
