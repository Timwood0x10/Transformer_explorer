"""
Model profiling tool: analyze Transformer model structure, parameter hotspots, and performance bottlenecks
"""
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import time

from utils.base_models import BaseTransformer


@dataclass
class LayerProfile:
    """单层性能分析数据"""
    name: str
    params: int
    flops: int
    memory_mb: float
    forward_time_ms: float
    backward_time_ms: float
    param_ratio: float
    flops_ratio: float


@dataclass
class ParameterHotspot:
    """参数热点分析"""
    layer_name: str
    grad_norm: float
    param_norm: float
    update_ratio: float  # 梯度更新幅度
    is_hotspot: bool


class TransformerProfiler:
    """Transformer 模型性能分析器"""
    
    def __init__(self, model: nn.Module, input_size: Tuple[int, int, int]):
        """
        Args:
            model: Transformer 模型
            input_size: (batch_size, seq_len, d_model)
        """
        self.model = model
        self.input_size = input_size
        self.layer_profiles: List[LayerProfile] = []
        self.hotspots: List[ParameterHotspot] = []
        
    def count_parameters(self, module: nn.Module = None) -> Dict[str, int]:
        """统计模型参数量"""
        if module is None:
            module = self.model
            
        param_counts = {}
        total = 0
        
        for name, param in module.named_parameters():
            if param.requires_grad:
                count = param.numel()
                param_counts[name] = count
                total += count
        
        param_counts['total'] = total
        return param_counts
    
    def estimate_flops(self, batch_size: int, seq_len: int, d_model: int, 
                       n_heads: int, n_layers: int, vocab_size: int) -> Dict[str, int]:
        """估算 FLOPs (浮点运算次数)"""
        
        # Self-Attention FLOPs
        # Q, K, V 投影: 3 * (seq_len * d_model * d_model)
        qkv_proj = 3 * seq_len * d_model * d_model
        
        # Q @ K^T: seq_len * seq_len * d_model
        attention_scores = seq_len * seq_len * d_model
        
        # Softmax: 约 5 * seq_len * seq_len (exp, sum, div)
        softmax = 5 * seq_len * seq_len
        
        # Attention @ V: seq_len * seq_len * d_model
        attention_output = seq_len * seq_len * d_model
        
        # Output 投影: seq_len * d_model * d_model
        output_proj = seq_len * d_model * d_model
        
        attention_flops = qkv_proj + attention_scores + softmax + attention_output + output_proj
        
        # FFN FLOPs
        # W1: seq_len * d_model * (4 * d_model)
        # W2: seq_len * (4 * d_model) * d_model
        ffn_flops = seq_len * d_model * (4 * d_model) * 2
        
        # 单层总 FLOPs
        layer_flops = attention_flops + ffn_flops
        
        # 所有层
        total_flops = n_layers * layer_flops * batch_size
        
        return {
            'attention_per_layer': attention_flops,
            'ffn_per_layer': ffn_flops,
            'total_per_layer': layer_flops,
            'total_model': total_flops,
            'qkv_projection': qkv_proj,
            'attention_matrix': attention_scores + attention_output,
            'ffn_total': ffn_flops
        }
    
    def estimate_memory(self, batch_size: int, seq_len: int, d_model: int, 
                       n_layers: int, dtype: torch.dtype = torch.float32) -> Dict[str, float]:
        """估算显存占用 (MB)"""
        
        bytes_per_element = 4 if dtype == torch.float32 else 2  # FP32: 4 bytes, FP16/BF16: 2 bytes
        
        # 1. 模型参数显存
        param_counts = self.count_parameters()
        param_memory = param_counts['total'] * bytes_per_element / (1024**2)
        
        # 2. 激活值显存 (单层)
        # Embedding: batch * seq_len * d_model
        embedding = batch_size * seq_len * d_model * bytes_per_element / (1024**2)
        
        # Attention:
        # - Q, K, V: 3 * batch * seq_len * d_model
        # - Attention weights: batch * n_heads * seq_len * seq_len
        # - Attention output: batch * seq_len * d_model
        attention_activation = (
            3 * batch_size * seq_len * d_model +  # Q, K, V
            batch_size * 8 * seq_len * seq_len +  # attention weights (假设 8 heads)
            batch_size * seq_len * d_model        # output
        ) * bytes_per_element / (1024**2)
        
        # FFN:
        # - Hidden: batch * seq_len * (4 * d_model)
        # - Output: batch * seq_len * d_model
        ffn_activation = (
            batch_size * seq_len * 4 * d_model +
            batch_size * seq_len * d_model
        ) * bytes_per_element / (1024**2)
        
        # 单层激活显存
        layer_activation = attention_activation + ffn_activation
        
        # 3. 梯度显存 (训练时，约等于参数量)
        gradient_memory = param_memory
        
        # 4. 优化器状态 (AdamW: 2倍参数量，m 和 v)
        optimizer_memory = 2 * param_memory
        
        return {
            'parameters_mb': param_memory,
            'activation_per_layer_mb': layer_activation,
            'activation_total_mb': layer_activation * n_layers,
            'gradients_mb': gradient_memory,
            'optimizer_mb': optimizer_memory,
            'total_training_mb': param_memory + layer_activation * n_layers + gradient_memory + optimizer_memory,
            'total_inference_mb': param_memory + layer_activation * n_layers
        }
    
    def profile_layers(self, batch_size: int = 8, seq_len: int = 128, 
                      d_model: int = 512, n_heads: int = 8, 
                      n_layers: int = 6, vocab_size: int = 50000) -> List[LayerProfile]:
        """Analyze computation and parameter counts per layer."""
        
        # Estimate FLOPs and parameters for each component
        flops_data = self.estimate_flops(batch_size, seq_len, d_model, n_heads, n_layers, vocab_size)
        
        profiles = []
        
        # 1. Embedding layer
        embedding_params = d_model * vocab_size
        embedding_flops = batch_size * seq_len * d_model
        profiles.append(LayerProfile(
            name="Embedding",
            params=embedding_params,
            flops=embedding_flops,
            memory_mb=embedding_params * 4 / (1024**2),
            forward_time_ms=0,
            backward_time_ms=0,
            param_ratio=0,
            flops_ratio=0
        ))
        
        # 2. 每个 Transformer 层
        for i in range(n_layers):
            # Self-Attention
            attn_params = 4 * d_model * d_model  # Q, K, V, O 投影
            attn_flops = flops_data['attention_per_layer'] * batch_size
            profiles.append(LayerProfile(
                name=f"Layer{i+1}_Attention",
                params=attn_params,
                flops=attn_flops,
                memory_mb=attn_params * 4 / (1024**2),
                forward_time_ms=0,
                backward_time_ms=0,
                param_ratio=0,
                flops_ratio=0
            ))
            
            # FFN
            ffn_params = 2 * d_model * 4 * d_model  # W1, W2
            ffn_flops = flops_data['ffn_per_layer'] * batch_size
            profiles.append(LayerProfile(
                name=f"Layer{i+1}_FFN",
                params=ffn_params,
                flops=ffn_flops,
                memory_mb=ffn_params * 4 / (1024**2),
                forward_time_ms=0,
                backward_time_ms=0,
                param_ratio=0,
                flops_ratio=0
            ))
        
        # 3. 输出层
        output_params = d_model * vocab_size
        output_flops = batch_size * seq_len * d_model * vocab_size
        profiles.append(LayerProfile(
            name="Output_Projection",
            params=output_params,
            flops=output_flops,
            memory_mb=output_params * 4 / (1024**2),
            forward_time_ms=0,
            backward_time_ms=0,
            param_ratio=0,
            flops_ratio=0
        ))
        
        # 计算比例
        total_params = sum(p.params for p in profiles)
        total_flops = sum(p.flops for p in profiles)
        
        for profile in profiles:
            profile.param_ratio = profile.params / total_params * 100
            profile.flops_ratio = profile.flops / total_flops * 100
        
        self.layer_profiles = profiles
        return profiles
    
    def analyze_gradient_hotspots(self, model: nn.Module, 
                                  threshold: float = 0.1) -> List[ParameterHotspot]:
        """分析梯度热点（哪些参数更新最频繁）"""
        
        hotspots = []
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                param_norm = param.data.norm().item()
                
                # 计算更新比例
                if param_norm > 1e-8:
                    update_ratio = grad_norm / param_norm
                else:
                    update_ratio = 0
                
                # 判断是否为热点
                is_hotspot = update_ratio > threshold
                
                hotspots.append(ParameterHotspot(
                    layer_name=name,
                    grad_norm=grad_norm,
                    param_norm=param_norm,
                    update_ratio=update_ratio,
                    is_hotspot=is_hotspot
                ))
        
        # 按更新比例排序
        hotspots.sort(key=lambda x: x.update_ratio, reverse=True)
        self.hotspots = hotspots
        return hotspots
    
    def get_attention_complexity_comparison(self, seq_lengths: List[int], 
                                           d_model: int = 512) -> Dict[str, List[float]]:
        """对比不同序列长度下的计算复杂度"""
        
        results = {
            'seq_lengths': seq_lengths,
            'transformer_flops': [],
            'linear_attention_flops': [],
            'mamba_flops': [],
            'transformer_memory': [],
            'mamba_memory': []
        }
        
        for L in seq_lengths:
            # Transformer: O(L^2 * d)
            transformer = L * L * d_model
            results['transformer_flops'].append(transformer)
            
            # Linear Attention: O(L * d^2)
            linear = L * d_model * d_model
            results['linear_attention_flops'].append(linear)
            
            # Mamba: O(L * d * N) where N is state dimension
            mamba = L * d_model * 16  # 假设 state_dim = 16
            results['mamba_flops'].append(mamba)
            
            # Memory: Transformer 需要存储 attention matrix
            transformer_mem = L * L * 4 / (1024**2)  # FP32
            mamba_mem = L * 16 * 4 / (1024**2)  # state only
            results['transformer_memory'].append(transformer_mem)
            results['mamba_memory'].append(mamba_mem)
        
        return results
    
    def simulate_training_step(self, batch_size: int = 8, seq_len: int = 128,
                              d_model: int = 512, n_layers: int = 6) -> Dict[str, float]:
        """模拟一次训练步的时间分解"""
        
        # 估算各阶段时间 (基于经验公式)
        base_time = seq_len * d_model * n_layers / 1e6  # 基准时间 (ms)
        
        return {
            'forward_ms': base_time * 1.0,
            'attention_ms': base_time * 0.4,
            'ffn_ms': base_time * 0.3,
            'backward_ms': base_time * 2.0,  # backward 通常是 forward 的 2 倍
            'optimizer_step_ms': base_time * 0.5,
            'total_ms': base_time * 3.5
        }


def create_sample_transformer(d_model: int = 512, n_heads: int = 8, 
                             n_layers: int = 6, vocab_size: int = 50000) -> nn.Module:
    """Create a sample Transformer model for analysis.

    Uses the shared BaseTransformer from utils.base_models, which provides
    the same output shape (B, S, vocab_size) and interface forward(input_ids) -> logits.
    """
    return BaseTransformer(d_model, n_heads, n_layers, vocab_size)


if __name__ == "__main__":
    # Test code
    model = create_sample_transformer()
    profiler = TransformerProfiler(model, (8, 128, 512))
    
    # 参数统计
    params = profiler.count_parameters()
    print(f"Total parameters: {params['total']:,}")
    
    # 层级分析
    profiles = profiler.profile_layers()
    for p in profiles[:5]:
        print(f"{p.name}: {p.params:,} params ({p.param_ratio:.1f}%), {p.flops:,} FLOPs ({p.flops_ratio:.1f}%)")
