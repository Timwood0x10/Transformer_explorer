"""
模型权重分析工具：分析权重分布、异常值和演化趋势
"""
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import plotly.graph_objects as go
import plotly.express as px
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


@dataclass
class WeightStats:
    """权重统计信息"""
    layer_name: str
    mean: float
    std: float
    min: float
    max: float
    median: float
    skewness: float
    kurtosis: float
    outlier_ratio: float
    dead_ratio: float  # 接近零的权重比例


@dataclass
class WeightEvolution:
    """权重演化数据"""
    layer_name: str
    step: int
    mean: float
    std: float
    norm: float
    update_magnitude: float


class WeightAnalyzer:
    """模型权重分析器"""
    
    def __init__(self, model: nn.Module):
        """
        Args:
            model: 要分析的模型
        """
        self.model = model
        self.weight_history: Dict[str, List[WeightEvolution]] = {}
        self.initial_weights: Dict[str, torch.Tensor] = {}
        
        # 保存初始权重
        self._save_initial_weights()
    
    def _save_initial_weights(self):
        """保存模型初始权重"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.initial_weights[name] = param.data.clone()
    
    def analyze_weight_distribution(self) -> List[WeightStats]:
        """分析当前权重分布"""
        weight_stats = []
        
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.dim() >= 2:  # 只分析权重矩阵
                weights = param.data.flatten().cpu().numpy()
                
                # 基本统计
                mean = np.mean(weights)
                std = np.std(weights)
                min_val = np.min(weights)
                max_val = np.max(weights)
                median = np.median(weights)
                
                # 高阶统计
                skewness = stats.skew(weights)
                kurtosis = stats.kurtosis(weights)
                
                # 异常值检测（3σ原则）
                outlier_mask = np.abs(weights - mean) > 3 * std
                outlier_ratio = np.mean(outlier_mask)
                
                # 死权重检测（接近零）
                dead_mask = np.abs(weights) < 1e-6
                dead_ratio = np.mean(dead_mask)
                
                weight_stats.append(WeightStats(
                    layer_name=name,
                    mean=mean,
                    std=std,
                    min=min_val,
                    max=max_val,
                    median=median,
                    skewness=skewness,
                    kurtosis=kurtosis,
                    outlier_ratio=outlier_ratio,
                    dead_ratio=dead_ratio
                ))
        
        return weight_stats
    
    def record_weight_evolution(self, step: int):
        """记录权重演化"""
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.dim() >= 2:
                weights = param.data
                
                if name not in self.weight_history:
                    self.weight_history[name] = []
                
                # 计算更新幅度
                if name in self.initial_weights:
                    update_magnitude = (weights - self.initial_weights[name]).norm().item()
                else:
                    update_magnitude = 0
                
                evolution = WeightEvolution(
                    layer_name=name,
                    step=step,
                    mean=weights.mean().item(),
                    std=weights.std().item(),
                    norm=weights.norm().item(),
                    update_magnitude=update_magnitude
                )
                
                self.weight_history[name].append(evolution)
    
    def detect_weight_anomalies(self) -> Dict[str, List[str]]:
        """检测权重异常"""
        anomalies = {}
        weight_stats = self.analyze_weight_distribution()
        
        for stats in weight_stats:
            layer_anomalies = []
            
            # 检测权重消失
            if abs(stats.mean) < 1e-6 and stats.std < 1e-6:
                layer_anomalies.append("权重消失 - 层可能未参与训练")
            
            # 检测权重爆炸
            if stats.std > 10 or abs(stats.mean) > 10:
                layer_anomalies.append("权重爆炸 - 可能存在梯度爆炸问题")
            
            # 检测异常值过多
            if stats.outlier_ratio > 0.05:
                layer_anomalies.append("异常值过多 - 可能存在数值不稳定")
            
            # 检测死权重过多
            if stats.dead_ratio > 0.5:
                layer_anomalies.append("死权重过多 - 层可能过于稀疏")
            
            # 检测分布偏斜
            if abs(stats.skewness) > 2:
                layer_anomalies.append("权重分布严重偏斜")
            
            # 检测峰度异常
            if abs(stats.kurtosis) > 10:
                layer_anomalies.append("权重峰度异常 - 可能存在极端值")
            
            if layer_anomalies:
                anomalies[stats.layer_name] = layer_anomalies
        
        return anomalies
    
    def compare_initialization_methods(self, layer_name: str, 
                                     init_methods: List[str]) -> Dict:
        """比较不同初始化方法的效果"""
        if layer_name not in self.initial_weights:
            return {}
        
        original_weights = self.initial_weights[layer_name]
        results = {}
        
        for method in init_methods:
            # 复制原始权重
            test_weights = original_weights.clone()
            
            # 应用不同的初始化
            if method == "xavier_uniform":
                nn.init.xavier_uniform_(test_weights)
            elif method == "xavier_normal":
                nn.init.xavier_normal_(test_weights)
            elif method == "kaiming_uniform":
                nn.init.kaiming_uniform_(test_weights, nonlinearity='relu')
            elif method == "kaiming_normal":
                nn.init.kaiming_normal_(test_weights, nonlinearity='relu')
            elif method == "orthogonal":
                nn.init.orthogonal_(test_weights)
            else:
                continue
            
            # 分析初始化后的分布
            weights_np = test_weights.flatten().cpu().numpy()
            results[method] = {
                'mean': np.mean(weights_np),
                'std': np.std(weights_np),
                'frobenius_norm': test_weights.norm().item(),
                'condition_number': self._estimate_condition_number(test_weights)
            }
        
        return results
    
    def _estimate_condition_number(self, weight_matrix: torch.Tensor) -> float:
        """估算矩阵条件数"""
        try:
            # 对于大矩阵，使用随机采样估算
            if weight_matrix.numel() > 10000:
                # 随机选择部分行和列
                m, n = weight_matrix.shape
                sample_size = min(100, min(m, n))
                rows_idx = torch.randperm(m)[:sample_size]
                cols_idx = torch.randperm(n)[:sample_size]
                sampled = weight_matrix[rows_idx][:, cols_idx]
            else:
                sampled = weight_matrix
            
            # 计算奇异值
            singular_values = torch.linalg.svdvals(sampled.float())
            if len(singular_values) > 0 and singular_values[-1] > 1e-10:
                return (singular_values[0] / singular_values[-1]).item()
            else:
                return float('inf')
        except:
            return float('inf')
    
    def analyze_weight_correlation(self) -> Dict[str, Dict[str, float]]:
        """分析层间权重相关性"""
        layer_weights = {}
        
        # 收集各层的权重
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.dim() >= 2:
                # 展平权重向量
                layer_weights[name] = param.data.flatten().cpu().numpy()
        
        # 计算相关性矩阵
        layer_names = list(layer_weights.keys())
        correlations = {}
        
        for i, name1 in enumerate(layer_names):
            correlations[name1] = {}
            for j, name2 in enumerate(layer_names):
                if i <= j:  # Only compute upper triangle
                    w1, w2 = layer_weights[name1], layer_weights[name2]
                    # corrcoef requires equal-length arrays; truncate to min length
                    min_len = min(len(w1), len(w2))
                    if min_len < 2:
                        corr = 0.0
                    else:
                        corr = np.corrcoef(w1[:min_len], w2[:min_len])[0, 1]
                    correlations[name1][name2] = corr if not np.isnan(corr) else 0.0
                else:
                    correlations[name1][name2] = correlations[name2][name1]
        
        return correlations
    
    def visualize_weight_distribution(self, layer_name: str = None) -> go.Figure:
        """可视化权重分布"""
        weight_stats = self.analyze_weight_distribution()
        
        if layer_name:
            # 单层分布
            stats = next((s for s in weight_stats if s.layer_name == layer_name), None)
            if not stats:
                return go.Figure()
            
            param = next((p for n, p in self.model.named_parameters() 
                         if n == layer_name and p.requires_grad), None)
            if param is None:
                return go.Figure()
            
            weights = param.data.flatten().cpu().numpy()
            
            fig = go.Figure()
            
            # 直方图
            fig.add_trace(go.Histogram(
                x=weights,
                nbinsx=100,
                name='权重分布',
                histnorm='probability density'
            ))
            
            # 添加统计线
            fig.add_vline(x=stats.mean, line_dash="dash", line_color="red", 
                         annotation_text=f"均值: {stats.mean:.4f}")
            fig.add_vline(x=stats.median, line_dash="dash", line_color="green",
                         annotation_text=f"中位数: {stats.median:.4f}")
            
            fig.update_layout(
                title=f'{layer_name} 权重分布',
                xaxis_title='权重值',
                yaxis_title='密度',
                height=400
            )
            
        else:
            # 多层分布对比
            fig = go.Figure()
            
            for stats in weight_stats[:10]:  # 只显示前10层
                param = next((p for n, p in self.model.named_parameters() 
                            if n == stats.layer_name and p.requires_grad), None)
                if param is not None:
                    weights = param.data.flatten().cpu().numpy()
                    
                    fig.add_trace(go.Histogram(
                        x=weights,
                        nbinsx=50,
                        name=stats.layer_name.split('.')[-1],
                        histnorm='probability density',
                        opacity=0.7
                    ))
            
            fig.update_layout(
                title='各层权重分布对比',
                xaxis_title='权重值',
                yaxis_title='密度',
                height=500,
                barmode='overlay'
            )
        
        return fig
    
    def visualize_weight_evolution(self, layer_name: str) -> go.Figure:
        """可视化权重演化"""
        if layer_name not in self.weight_history:
            return go.Figure()
        
        history = self.weight_history[layer_name]
        steps = [h.step for h in history]
        means = [h.mean for h in history]
        stds = [h.std for h in history]
        norms = [h.norm for h in history]
        
        fig = go.Figure()
        
        # 均值演化
        fig.add_trace(go.Scatter(
            x=steps, y=means,
            mode='lines+markers',
            name='均值',
            line=dict(color='blue')
        ))
        
        # 标准差演化
        fig.add_trace(go.Scatter(
            x=steps, y=stds,
            mode='lines+markers',
            name='标准差',
            yaxis='y2',
            line=dict(color='red')
        ))
        
        # 范数演化
        fig.add_trace(go.Scatter(
            x=steps, y=norms,
            mode='lines+markers',
            name='Frobenius范数',
            yaxis='y3',
            line=dict(color='green')
        ))
        
        fig.update_layout(
            title=f'{layer_name} 权重演化',
            xaxis_title='训练步数',
            yaxis=dict(title='均值', side='left'),
            yaxis2=dict(title='标准差', side='right', overlaying='y'),
            yaxis3=dict(title='范数', side='right', overlaying='y', position=0.85),
            height=400
        )
        
        return fig
    
    def generate_weight_report(self) -> str:
        """生成权重分析报告"""
        weight_stats = self.analyze_weight_distribution()
        anomalies = self.detect_weight_anomalies()
        correlations = self.analyze_weight_correlation()
        
        report = """
# 模型权重分析报告

## 📊 权重统计摘要
"""
        
        # 整体统计
        all_means = [s.mean for s in weight_stats]
        all_stds = [s.std for s in weight_stats]
        all_outliers = [s.outlier_ratio for s in weight_stats]
        all_deads = [s.dead_ratio for s in weight_stats]
        
        report += f"""
- 平均权重范围: [{np.min(all_means):.6f}, {np.max(all_means):.6f}]
- 平均标准差: {np.mean(all_stds):.6f}
- 平均异常值比例: {np.mean(all_outliers)*100:.2f}%
- 平均死权重比例: {np.mean(all_deads)*100:.2f}%
"""
        
        # 异常报告
        if anomalies:
            report += "\n## ⚠️ 权重异常\n"
            for layer_name, issues in anomalies.items():
                report += f"\n### {layer_name}:\n"
                for issue in issues:
                    report += f"- {issue}\n"
        else:
            report += "\n## ✅ 未检测到权重异常\n"
        
        # 相关性分析
        report += "\n## 🔄 层间相关性分析\n"
        
        # 找出相关性最高和最低的层对
        high_corr_pairs = []
        low_corr_pairs = []
        
        for i, layer1 in enumerate(correlations):
            for j, layer2 in enumerate(correlations[layer1]):
                if i < j:  # 避免重复
                    corr = correlations[layer1][layer2]
                    if corr > 0.5:
                        high_corr_pairs.append((layer1, layer2, corr))
                    elif corr < -0.5:
                        low_corr_pairs.append((layer1, layer2, corr))
        
        if high_corr_pairs:
            report += "\n### 高相关性层对 (>0.5):\n"
            for layer1, layer2, corr in sorted(high_corr_pairs, key=lambda x: x[2], reverse=True)[:5]:
                report += f"- {layer1} ↔ {layer2}: {corr:.3f}\n"
        
        if low_corr_pairs:
            report += "\n### 低相关性层对 (<-0.5):\n"
            for layer1, layer2, corr in sorted(low_corr_pairs, key=lambda x: x[2])[:5]:
                report += f"- {layer1} ↔ {layer2}: {corr:.3f}\n"
        
        # 优化建议
        report += "\n## 💡 优化建议\n"
        
        if np.mean(all_outliers) > 0.05:
            report += "- 考虑使用梯度裁剪来减少异常值\n"
        
        if np.mean(all_deads) > 0.3:
            report += "- 考虑增加学习率或检查权重初始化\n"
        
        if np.mean(all_stds) > 5:
            report += "- 权重方差较大，建议使用更保守的初始化方法\n"
        
        if np.mean(all_stds) < 0.01:
            report += "- 权重方差过小，可能需要增加学习率\n"
        
        return report


if __name__ == "__main__":
    # 测试代码
    from utils.model_profiler import create_sample_transformer
    
    model = create_sample_transformer()
    analyzer = WeightAnalyzer(model)
    
    # 分析权重
    stats = analyzer.analyze_weight_distribution()
    print(f"分析了 {len(stats)} 层的权重分布")
    
    # 检测异常
    anomalies = analyzer.detect_weight_anomalies()
    print(f"发现 {len(anomalies)} 层存在异常")
    
    # 模拟训练过程
    for step in range(10):
        # 模拟权重更新
        with torch.no_grad():
            for param in model.parameters():
                if param.requires_grad:
                    param.data += torch.randn_like(param.data) * 0.001
        
        analyzer.record_weight_evolution(step)
    
    # 生成报告
    report = analyzer.generate_weight_report()
    print(report)
