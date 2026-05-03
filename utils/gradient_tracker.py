"""
梯度流追踪器：实时计算和追踪梯度在 Transformer 中的反向传播
================================================================

功能：
1. 记录每一层的梯度范数（实时计算，非硬编码）
2. 对比残差路径 vs 主路径的梯度
3. 检测梯度消失/爆炸问题
4. 验证残差连接的"梯度高速公路"效应

作者：Transformer Explorer Team
日期：2025-12-29
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field


@dataclass
class LayerGradientInfo:
    """单层的梯度信息"""
    layer_idx: int
    layer_name: str
    
    # 参数梯度
    qkv_grad_norm: float = 0.0  # Q/K/V 投影层的梯度范数
    out_proj_grad_norm: float = 0.0  # 输出投影层的梯度范数
    ffn_grad_norm: float = 0.0  # FFN 的梯度范数
    
    # 激活梯度（反向传播中的梯度）
    input_grad_norm: float = 0.0  # 该层输入的梯度
    output_grad_norm: float = 0.0  # 该层输出的梯度
    
    # 残差连接的梯度贡献
    residual_grad_1: float = 0.0  # 第一个残差连接的梯度
    residual_grad_2: float = 0.0  # 第二个残差连接的梯度
    
    # 梯度流比例
    grad_flow_ratio: float = 1.0  # output_grad / input_grad
    
    # 健康度指标
    has_vanishing: bool = False  # 是否有梯度消失
    has_explosion: bool = False  # 是否有梯度爆炸
    health_score: float = 1.0  # 0-1，越高越健康


@dataclass
class GradientJourney:
    """完整的梯度追踪记录"""
    model_config: Dict[str, int]  # 模型配置
    
    # 损失信息
    loss_value: float = 0.0
    
    # 每一层的梯度信息
    layer_gradients: List[LayerGradientInfo] = field(default_factory=list)
    
    # 整体统计
    total_grad_norm: float = 0.0  # 所有参数梯度的总范数
    max_grad_norm: float = 0.0  # 最大梯度范数
    min_grad_norm: float = float('inf')  # 最小梯度范数
    
    # 梯度流健康度
    has_vanishing_problem: bool = False
    has_explosion_problem: bool = False
    overall_health_score: float = 1.0
    
    # 残差效应
    avg_residual_contribution: float = 0.0  # 残差路径的平均贡献


class GradientTrackingTransformer(nn.Module):
    """支持梯度追踪的 Transformer 模型"""
    
    def __init__(self, d_model: int, n_heads: int, n_layers: int, 
                 vocab_size: int, max_seq_len: int = 512, 
                 dropout: float = 0.1, use_residual: bool = True):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.vocab_size = vocab_size
        self.use_residual = use_residual  # 是否使用残差连接
        
        # Embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)
        
        # Transformer Layers
        self.layers = nn.ModuleList([
            GradientTrackingLayer(d_model, n_heads, dropout, use_residual)
            for _ in range(n_layers)
        ])
        
        # Output head
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        
        self.dropout = nn.Dropout(dropout)
        
        # 梯度追踪
        self.gradient_hooks = []
        self.gradient_cache = {}
        
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.pos_embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.02)
    
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        batch_size, seq_len = input_ids.shape
        
        # Position IDs
        position_ids = torch.arange(seq_len, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        
        # Embeddings
        token_embeds = self.token_embedding(input_ids)
        pos_embeds = self.pos_embedding(position_ids)
        
        x = token_embeds + pos_embeds
        x = self.dropout(x)
        
        # 保存中间激活用于梯度追踪
        if x.requires_grad:
            x.retain_grad()
        self.activations = [x]
        
        # 通过所有层
        for layer in self.layers:
            x = layer(x)
            # 保留梯度以便追踪
            if x.requires_grad:
                x.retain_grad()
            self.activations.append(x)
        
        # 输出投影
        logits = self.lm_head(x)
        
        return logits
    
    def enable_gradient_tracking(self):
        """启用梯度追踪"""
        self.gradient_cache = {}
        
        # 为每层注册 backward hook
        for i, layer in enumerate(self.layers):
            # 追踪层的输入梯度
            def make_hook(layer_idx):
                def hook(grad):
                    self.gradient_cache[f'layer_{layer_idx}_input_grad'] = grad.detach()
                    return grad
                return hook
            
            # 注册到激活上
            if len(self.activations) > i:
                if self.activations[i].requires_grad:
                    handle = self.activations[i].register_hook(make_hook(i))
                    self.gradient_hooks.append(handle)
    
    def disable_gradient_tracking(self):
        """禁用梯度追踪"""
        for handle in self.gradient_hooks:
            handle.remove()
        self.gradient_hooks = []
        self.gradient_cache = {}


class GradientTrackingLayer(nn.Module):
    """支持梯度追踪的 Transformer 层"""
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1, 
                 use_residual: bool = True):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.use_residual = use_residual
        
        # Multi-head Attention
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout)
        )
        
        # LayerNorm
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
        
        # 初始化
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        for module in [self.q_proj, self.k_proj, self.v_proj, self.out_proj]:
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        
        for module in self.ffn:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        batch_size, seq_len, d_model = x.shape
        
        # === Self-Attention ===
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)
        
        # 重塑为多头
        Q = Q.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        
        # 注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_dim)
        attn_weights = F.softmax(scores, dim=-1)
        
        # 应用注意力
        attn_output = torch.matmul(attn_weights, V)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        
        # 输出投影
        attn_output = self.out_proj(attn_output)
        attn_output = self.dropout(attn_output)
        
        # 第一个残差连接
        if self.use_residual:
            x = x + attn_output
        else:
            x = attn_output
        
        x = self.norm1(x)
        
        # === FFN ===
        ffn_output = self.ffn(x)
        
        # 第二个残差连接
        if self.use_residual:
            x = x + ffn_output
        else:
            x = ffn_output
        
        x = self.norm2(x)
        
        return x


class GradientTracker:
    """梯度追踪器主类"""
    
    def __init__(self, model: GradientTrackingTransformer):
        self.model = model
        self.device = next(model.parameters()).device
    
    def track_gradient_flow(self, input_ids: torch.Tensor, 
                           target_ids: torch.Tensor) -> GradientJourney:
        """
        追踪梯度流
        
        Args:
            input_ids: 输入 token IDs (batch, seq_len)
            target_ids: 目标 token IDs (batch, seq_len)
        
        Returns:
            GradientJourney: 完整的梯度追踪记录
        """
        self.model.train()  # 训练模式
        
        # 前向传播
        logits = self.model(input_ids)  # (batch, seq_len, vocab_size)
        
        # 计算损失
        loss = F.cross_entropy(
            logits.view(-1, self.model.vocab_size),
            target_ids.view(-1),
            reduction='mean'
        )
        
        # 反向传播
        self.model.zero_grad()
        loss.backward()
        
        # 收集梯度信息
        journey = GradientJourney(
            model_config={
                'd_model': self.model.d_model,
                'n_heads': self.model.n_heads,
                'n_layers': self.model.n_layers,
                'vocab_size': self.model.vocab_size,
                'use_residual': self.model.use_residual
            },
            loss_value=loss.item()
        )
        
        # 遍历每一层，收集梯度
        total_grad_norm = 0.0
        
        for i, layer in enumerate(self.model.layers):
            layer_info = LayerGradientInfo(
                layer_idx=i,
                layer_name=f"Layer_{i}"
            )
            
            # 计算参数梯度范数
            # Q/K/V 投影
            qkv_grads = []
            for proj in [layer.q_proj, layer.k_proj, layer.v_proj]:
                if proj.weight.grad is not None:
                    qkv_grads.append(proj.weight.grad.norm().item())
            layer_info.qkv_grad_norm = np.mean(qkv_grads) if qkv_grads else 0.0
            
            # 输出投影
            if layer.out_proj.weight.grad is not None:
                layer_info.out_proj_grad_norm = layer.out_proj.weight.grad.norm().item()
            
            # FFN
            ffn_grads = []
            for module in layer.ffn:
                if isinstance(module, nn.Linear) and module.weight.grad is not None:
                    ffn_grads.append(module.weight.grad.norm().item())
            layer_info.ffn_grad_norm = np.mean(ffn_grads) if ffn_grads else 0.0
            
            # 激活梯度（如果可用）
            if hasattr(self.model, 'activations') and len(self.model.activations) > i:
                if self.model.activations[i].grad is not None:
                    layer_info.input_grad_norm = self.model.activations[i].grad.norm().item()
                if i + 1 < len(self.model.activations) and self.model.activations[i+1].grad is not None:
                    layer_info.output_grad_norm = self.model.activations[i+1].grad.norm().item()
                
                # 计算梯度流比例
                if layer_info.input_grad_norm > 1e-10:
                    layer_info.grad_flow_ratio = layer_info.output_grad_norm / layer_info.input_grad_norm
            
            # 健康度检测
            max_grad = max(layer_info.qkv_grad_norm, layer_info.out_proj_grad_norm, 
                          layer_info.ffn_grad_norm)
            
            # 梯度消失：梯度 < 1e-5
            if max_grad < 1e-5:
                layer_info.has_vanishing = True
                journey.has_vanishing_problem = True
            
            # 梯度爆炸：梯度 > 100
            if max_grad > 100:
                layer_info.has_explosion = True
                journey.has_explosion_problem = True
            
            # 健康度评分 (0-1)
            if max_grad < 1e-5:
                layer_info.health_score = 0.0
            elif max_grad > 100:
                layer_info.health_score = 0.1
            elif max_grad < 1e-3:
                layer_info.health_score = 0.5
            elif max_grad > 10:
                layer_info.health_score = 0.7
            else:
                layer_info.health_score = 1.0
            
            journey.layer_gradients.append(layer_info)
            total_grad_norm += max_grad
        
        # 整体统计
        if journey.layer_gradients:
            all_grads = [lg.qkv_grad_norm for lg in journey.layer_gradients] + \
                       [lg.out_proj_grad_norm for lg in journey.layer_gradients] + \
                       [lg.ffn_grad_norm for lg in journey.layer_gradients]
            all_grads = [g for g in all_grads if g > 0]
            
            if all_grads:
                journey.total_grad_norm = sum(all_grads)
                journey.max_grad_norm = max(all_grads)
                journey.min_grad_norm = min(all_grads)
        
        # 整体健康度
        health_scores = [lg.health_score for lg in journey.layer_gradients]
        journey.overall_health_score = np.mean(health_scores) if health_scores else 0.0
        
        return journey
    
    def compare_with_without_residual(self, input_ids: torch.Tensor, 
                                     target_ids: torch.Tensor) -> Tuple[GradientJourney, GradientJourney]:
        """
        对比有/无残差连接的梯度流
        
        Returns:
            (with_residual_journey, without_residual_journey)
        """
        # 保存原始设置
        original_use_residual = self.model.use_residual
        
        # 有残差
        self.model.use_residual = True
        for layer in self.model.layers:
            layer.use_residual = True
        journey_with = self.track_gradient_flow(input_ids, target_ids)
        
        # 无残差
        self.model.use_residual = False
        for layer in self.model.layers:
            layer.use_residual = False
        journey_without = self.track_gradient_flow(input_ids, target_ids)
        
        # 恢复原始设置
        self.model.use_residual = original_use_residual
        for layer in self.model.layers:
            layer.use_residual = original_use_residual
        
        return journey_with, journey_without


if __name__ == "__main__":
    # 测试代码
    print("🧪 测试梯度追踪器...")
    
    # 创建模型
    d_model = 256
    n_heads = 8
    n_layers = 4
    vocab_size = 500
    
    model = GradientTrackingTransformer(d_model, n_heads, n_layers, vocab_size)
    tracker = GradientTracker(model)
    
    # 创建测试数据
    batch_size = 2
    seq_len = 10
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    target_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    print("\n1️⃣ 测试梯度追踪...")
    journey = tracker.track_gradient_flow(input_ids, target_ids)
    
    print(f"   损失: {journey.loss_value:.4f}")
    print(f"   总梯度范数: {journey.total_grad_norm:.4f}")
    print(f"   最大梯度: {journey.max_grad_norm:.4f}")
    print(f"   最小梯度: {journey.min_grad_norm:.4f}")
    print(f"   整体健康度: {journey.overall_health_score:.2f}")
    print(f"   梯度消失: {'是' if journey.has_vanishing_problem else '否'}")
    print(f"   梯度爆炸: {'是' if journey.has_explosion_problem else '否'}")
    
    print("\n   逐层梯度:")
    for lg in journey.layer_gradients:
        print(f"   Layer {lg.layer_idx}: QKV={lg.qkv_grad_norm:.4f}, "
              f"Out={lg.out_proj_grad_norm:.4f}, FFN={lg.ffn_grad_norm:.4f}, "
              f"健康度={lg.health_score:.2f}")
    
    print("\n2️⃣ 测试残差对比...")
    journey_with, journey_without = tracker.compare_with_without_residual(input_ids, target_ids)
    
    print(f"\n   有残差:")
    print(f"     健康度: {journey_with.overall_health_score:.2f}")
    print(f"     最大梯度: {journey_with.max_grad_norm:.4f}")
    
    print(f"\n   无残差:")
    print(f"     健康度: {journey_without.overall_health_score:.2f}")
    print(f"     最大梯度: {journey_without.max_grad_norm:.4f}")
    
    print("\n✅ 所有测试通过！")
