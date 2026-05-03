"""
Token 追踪器：追踪一个 Token 在 Transformer 中的完整旅程
============================================================

功能：
1. 记录 Token 从输入到输出的每一层隐藏状态
2. 可视化残差连接的"修正"效果
3. 分析每一层对 Token 表示的影响
4. 展示最终的 Logits 和预测结果

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
class LayerState:
    """单层的隐藏状态记录"""
    layer_idx: int
    layer_name: str
    
    # 输入状态
    input_hidden: np.ndarray  # (d_model,)
    
    # Attention 相关
    attn_query: Optional[np.ndarray] = None  # (d_model,)
    attn_key: Optional[np.ndarray] = None
    attn_value: Optional[np.ndarray] = None
    attn_output: Optional[np.ndarray] = None  # attention 输出
    attn_weights: Optional[np.ndarray] = None  # 对其他 token 的注意力权重
    
    # 残差连接前后
    before_residual_1: Optional[np.ndarray] = None  # attention 输出
    after_residual_1: Optional[np.ndarray] = None   # x + attention
    residual_delta_1: Optional[np.ndarray] = None   # 残差修正量
    
    # FFN 相关
    ffn_intermediate: Optional[np.ndarray] = None  # FFN 中间层
    ffn_output: Optional[np.ndarray] = None        # FFN 输出
    
    # 第二个残差连接
    before_residual_2: Optional[np.ndarray] = None
    after_residual_2: Optional[np.ndarray] = None
    residual_delta_2: Optional[np.ndarray] = None
    
    # 最终输出
    output_hidden: np.ndarray = None  # 该层最终输出
    
    # 统计信息
    norm_input: float = 0.0
    norm_output: float = 0.0
    norm_change: float = 0.0  # 输出相比输入的变化幅度


@dataclass
class TokenJourney:
    """Token 的完整旅程记录"""
    token_id: int
    token_text: str
    token_position: int  # 在序列中的位置
    
    # 初始嵌入
    embedding: np.ndarray  # (d_model,)
    positional_encoding: Optional[np.ndarray] = None
    
    # 每一层的状态
    layer_states: List[LayerState] = field(default_factory=list)
    
    # 最终输出
    final_hidden: np.ndarray = None  # (d_model,)
    logits: np.ndarray = None  # (vocab_size,)
    
    # 预测结果
    top_k_predictions: List[Tuple[int, str, float]] = field(default_factory=list)
    predicted_token_id: int = -1
    predicted_token: str = ""
    prediction_probability: float = 0.0


class TrackedTransformerLayer(nn.Module):
    """可追踪的 Transformer 层"""
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        
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
        
        # LayerNorm (Post-LN style)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
        
        # 追踪开关
        self.tracking_enabled = False
        self.tracked_position = -1
        self.layer_cache = {}
    
    def enable_tracking(self, position: int):
        """启用追踪模式"""
        self.tracking_enabled = True
        self.tracked_position = position
        self.layer_cache = {}
    
    def disable_tracking(self):
        """禁用追踪模式"""
        self.tracking_enabled = False
        self.tracked_position = -1
        self.layer_cache = {}
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        前向传播，支持追踪
        Args:
            x: (batch, seq_len, d_model)
            mask: Optional attention mask
        Returns:
            output: (batch, seq_len, d_model)
        """
        batch_size, seq_len, d_model = x.shape
        
        # 如果启用追踪，记录输入
        if self.tracking_enabled and 0 <= self.tracked_position < seq_len:
            self.layer_cache['input'] = x[0, self.tracked_position].detach().cpu().numpy()
        
        # === Self-Attention ===
        # 计算 Q, K, V
        Q = self.q_proj(x)  # (batch, seq_len, d_model)
        K = self.k_proj(x)
        V = self.v_proj(x)
        
        # 追踪 QKV
        if self.tracking_enabled and 0 <= self.tracked_position < seq_len:
            self.layer_cache['q'] = Q[0, self.tracked_position].detach().cpu().numpy()
            self.layer_cache['k'] = K[0, self.tracked_position].detach().cpu().numpy()
            self.layer_cache['v'] = V[0, self.tracked_position].detach().cpu().numpy()
        
        # 重塑为多头形状
        Q = Q.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        
        # 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_dim)
        
        # 应用 mask（如果有）
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Softmax
        attn_weights = F.softmax(scores, dim=-1)  # (batch, n_heads, seq_len, seq_len)
        
        # 追踪注意力权重
        if self.tracking_enabled and 0 <= self.tracked_position < seq_len:
            # 平均所有头的注意力权重
            self.layer_cache['attn_weights'] = attn_weights[0, :, self.tracked_position, :].mean(dim=0).detach().cpu().numpy()
        
        # 应用注意力
        attn_output = torch.matmul(attn_weights, V)  # (batch, n_heads, seq_len, head_dim)
        
        # 合并多头
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        
        # 输出投影
        attn_output = self.out_proj(attn_output)
        attn_output = self.dropout(attn_output)
        
        # 追踪 attention 输出
        if self.tracking_enabled and 0 <= self.tracked_position < seq_len:
            self.layer_cache['attn_output'] = attn_output[0, self.tracked_position].detach().cpu().numpy()
            self.layer_cache['before_residual_1'] = attn_output[0, self.tracked_position].detach().cpu().numpy()
        
        # === 第一个残差连接 ===
        x_residual_1 = x + attn_output
        
        # 追踪残差后
        if self.tracking_enabled and 0 <= self.tracked_position < seq_len:
            after_res = x_residual_1[0, self.tracked_position].detach().cpu().numpy()
            self.layer_cache['after_residual_1'] = after_res
            self.layer_cache['residual_delta_1'] = after_res - self.layer_cache['input']
        
        # LayerNorm
        x_norm_1 = self.norm1(x_residual_1)
        
        # === FFN ===
        ffn_output = self.ffn(x_norm_1)
        
        # 追踪 FFN 中间层（第一个线性层后）
        if self.tracking_enabled and 0 <= self.tracked_position < seq_len:
            # 获取 FFN 中间激活
            with torch.no_grad():
                ffn_intermediate = self.ffn[0](x_norm_1)  # 第一个线性层
                ffn_intermediate = F.gelu(ffn_intermediate)  # GELU 激活
                self.layer_cache['ffn_intermediate'] = ffn_intermediate[0, self.tracked_position].cpu().numpy()
            
            self.layer_cache['ffn_output'] = ffn_output[0, self.tracked_position].detach().cpu().numpy()
            self.layer_cache['before_residual_2'] = ffn_output[0, self.tracked_position].detach().cpu().numpy()
        
        # === 第二个残差连接 ===
        x_residual_2 = x_norm_1 + ffn_output
        
        # 追踪残差后
        if self.tracking_enabled and 0 <= self.tracked_position < seq_len:
            after_res_2 = x_residual_2[0, self.tracked_position].detach().cpu().numpy()
            self.layer_cache['after_residual_2'] = after_res_2
            # 残差相对于 norm 后的输入
            norm_input = x_norm_1[0, self.tracked_position].detach().cpu().numpy()
            self.layer_cache['residual_delta_2'] = after_res_2 - norm_input
        
        # LayerNorm
        output = self.norm2(x_residual_2)
        
        # 追踪最终输出
        if self.tracking_enabled and 0 <= self.tracked_position < seq_len:
            self.layer_cache['output'] = output[0, self.tracked_position].detach().cpu().numpy()
        
        return output


class TrackedTransformer(nn.Module):
    """可追踪的完整 Transformer 模型"""
    
    def __init__(self, vocab_size: int, d_model: int, n_heads: int, 
                 n_layers: int, max_seq_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.max_seq_len = max_seq_len
        
        # Token Embedding
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        
        # Positional Encoding
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)
        
        # Transformer Layers
        self.layers = nn.ModuleList([
            TrackedTransformerLayer(d_model, n_heads, dropout)
            for _ in range(n_layers)
        ])
        
        # Output head
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        
        # 权重共享（optional，类似 GPT）
        # self.lm_head.weight = self.token_embedding.weight
        
        self.dropout = nn.Dropout(dropout)
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        # Token embedding
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)
        # Positional embedding
        nn.init.normal_(self.pos_embedding.weight, mean=0.0, std=0.02)
        # LM head
        nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.02)
        
        # 初始化所有层
        for layer in self.layers:
            for name, param in layer.named_parameters():
                if 'weight' in name and len(param.shape) >= 2:
                    nn.init.xavier_uniform_(param)
                elif 'bias' in name:
                    nn.init.zeros_(param)
    
    def forward(self, input_ids: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        前向传播
        Args:
            input_ids: (batch, seq_len)
            mask: Optional attention mask
        Returns:
            logits: (batch, seq_len, vocab_size)
        """
        batch_size, seq_len = input_ids.shape
        
        # Position IDs
        position_ids = torch.arange(seq_len, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        
        # Embeddings
        token_embeds = self.token_embedding(input_ids)  # (batch, seq_len, d_model)
        pos_embeds = self.pos_embedding(position_ids)   # (batch, seq_len, d_model)
        
        x = token_embeds + pos_embeds
        x = self.dropout(x)
        
        # 通过所有 Transformer 层
        for layer in self.layers:
            x = layer(x, mask)
        
        # 输出投影
        logits = self.lm_head(x)  # (batch, seq_len, vocab_size)
        
        return logits
    
    def enable_tracking_for_all_layers(self, position: int):
        """为所有层启用追踪"""
        for layer in self.layers:
            layer.enable_tracking(position)
    
    def disable_tracking_for_all_layers(self):
        """为所有层禁用追踪"""
        for layer in self.layers:
            layer.disable_tracking()
    
    def get_layer_caches(self) -> List[Dict]:
        """获取所有层的缓存"""
        return [layer.layer_cache for layer in self.layers]


class TokenTracker:
    """Token 追踪器主类"""
    
    def __init__(self, model: TrackedTransformer, tokenizer_vocab: Dict[int, str]):
        """
        Args:
            model: 可追踪的 Transformer 模型
            tokenizer_vocab: token ID 到文本的映射 {id: token_text}
        """
        self.model = model
        self.vocab = tokenizer_vocab
        self.reverse_vocab = {v: k for k, v in tokenizer_vocab.items()}
        self.device = next(model.parameters()).device
    
    def track_token_journey(self, input_text: str, token_position: int, 
                           return_top_k: int = 10) -> TokenJourney:
        """
        追踪一个 Token 的完整旅程
        
        Args:
            input_text: 输入文本（空格分隔的 tokens）
            token_position: 要追踪的 token 在序列中的位置
            return_top_k: 返回 top-k 预测结果
        
        Returns:
            TokenJourney: 完整的追踪记录
        """
        self.model.eval()
        
        # Tokenize（简单的空格分隔）
        tokens = input_text.split()
        
        if token_position >= len(tokens):
            raise ValueError(f"Position {token_position} out of range for sequence length {len(tokens)}")
        
        # 转换为 token IDs
        token_ids = []
        for token in tokens:
            if token in self.reverse_vocab:
                token_ids.append(self.reverse_vocab[token])
            else:
                # 使用 <UNK> token
                token_ids.append(0)
        
        input_ids = torch.tensor([token_ids], dtype=torch.long, device=self.device)
        
        # 获取当前 token
        tracked_token_id = token_ids[token_position]
        tracked_token_text = tokens[token_position]
        
        # 启用追踪
        self.model.enable_tracking_for_all_layers(token_position)
        
        with torch.no_grad():
            # 获取嵌入
            token_embed = self.model.token_embedding(input_ids[0, token_position]).cpu().numpy()
            pos_embed = self.model.pos_embedding(torch.tensor([token_position], device=self.device))[0].cpu().numpy()
            
            # 前向传播
            logits = self.model(input_ids)  # (1, seq_len, vocab_size)
            
            # 获取该 token 的 logits
            token_logits = logits[0, token_position].cpu().numpy()  # (vocab_size,)
            
            # 获取所有层的缓存
            layer_caches = self.model.get_layer_caches()
        
        # 禁用追踪
        self.model.disable_tracking_for_all_layers()
        
        # 构建 TokenJourney
        journey = TokenJourney(
            token_id=tracked_token_id,
            token_text=tracked_token_text,
            token_position=token_position,
            embedding=token_embed,
            positional_encoding=pos_embed
        )
        
        # 记录每一层的状态
        for layer_idx, cache in enumerate(layer_caches):
            if not cache:  # 空缓存，跳过
                continue
            
            layer_state = LayerState(
                layer_idx=layer_idx,
                layer_name=f"Layer_{layer_idx}",
                input_hidden=cache.get('input'),
                attn_query=cache.get('q'),
                attn_key=cache.get('k'),
                attn_value=cache.get('v'),
                attn_output=cache.get('attn_output'),
                attn_weights=cache.get('attn_weights'),
                before_residual_1=cache.get('before_residual_1'),
                after_residual_1=cache.get('after_residual_1'),
                residual_delta_1=cache.get('residual_delta_1'),
                ffn_intermediate=cache.get('ffn_intermediate'),
                ffn_output=cache.get('ffn_output'),
                before_residual_2=cache.get('before_residual_2'),
                after_residual_2=cache.get('after_residual_2'),
                residual_delta_2=cache.get('residual_delta_2'),
                output_hidden=cache.get('output')
            )
            
            # 计算统计信息
            if layer_state.input_hidden is not None:
                layer_state.norm_input = np.linalg.norm(layer_state.input_hidden)
            if layer_state.output_hidden is not None:
                layer_state.norm_output = np.linalg.norm(layer_state.output_hidden)
                if layer_state.input_hidden is not None:
                    layer_state.norm_change = layer_state.norm_output - layer_state.norm_input
            
            journey.layer_states.append(layer_state)
        
        # 记录最终输出
        if journey.layer_states:
            journey.final_hidden = journey.layer_states[-1].output_hidden
        
        journey.logits = token_logits
        
        # 获取 top-k 预测
        top_k_indices = np.argsort(token_logits)[-return_top_k:][::-1]
        probs = F.softmax(torch.from_numpy(token_logits), dim=0).numpy()
        
        for idx in top_k_indices:
            token_text = self.vocab.get(idx, f"<ID:{idx}>")
            journey.top_k_predictions.append((int(idx), token_text, float(probs[idx])))
        
        # 记录最高概率的预测
        if journey.top_k_predictions:
            journey.predicted_token_id = journey.top_k_predictions[0][0]
            journey.predicted_token = journey.top_k_predictions[0][1]
            journey.prediction_probability = journey.top_k_predictions[0][2]
        
        return journey
    
    def compare_residual_effects(self, journey: TokenJourney) -> Dict[str, Any]:
        """
        分析残差连接的效果
        
        Returns:
            Dict with analysis results
        """
        results = {
            'layer_residual_norms': [],
            'cumulative_change': [],
            'residual_importance': []
        }
        
        for layer_state in journey.layer_states:
            # 第一个残差连接的效果
            if layer_state.residual_delta_1 is not None:
                delta_norm = np.linalg.norm(layer_state.residual_delta_1)
                input_norm = layer_state.norm_input
                
                importance = delta_norm / (input_norm + 1e-8)  # 相对重要性
                
                results['layer_residual_norms'].append({
                    'layer': layer_state.layer_idx,
                    'residual_1_norm': float(delta_norm),
                    'residual_1_importance': float(importance)
                })
            
            # 第二个残差连接的效果
            if layer_state.residual_delta_2 is not None:
                delta_norm_2 = np.linalg.norm(layer_state.residual_delta_2)
                results['layer_residual_norms'][-1]['residual_2_norm'] = float(delta_norm_2)
        
        # 计算累积变化
        if journey.embedding is not None and journey.final_hidden is not None:
            total_change = np.linalg.norm(journey.final_hidden - journey.embedding)
            results['total_change_norm'] = float(total_change)
            results['embedding_norm'] = float(np.linalg.norm(journey.embedding))
            results['final_norm'] = float(np.linalg.norm(journey.final_hidden))
        
        return results


# 辅助函数：创建简单的词表
def create_simple_vocab(vocab_size: int = 1000) -> Dict[int, str]:
    """创建一个简单的词表用于演示"""
    vocab = {0: "<UNK>", 1: "<PAD>", 2: "<BOS>", 3: "<EOS>"}
    
    # 添加一些常见词
    common_words = [
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
        "I", "you", "he", "she", "it", "we", "they", "me", "him", "her",
        "my", "your", "his", "their", "our", "this", "that", "these", "those",
        "what", "which", "who", "when", "where", "why", "how",
        "in", "on", "at", "to", "for", "of", "with", "from", "by",
        "and", "or", "but", "not", "have", "has", "had", "do", "does", "did",
        "can", "could", "will", "would", "should", "may", "might", "must",
        "love", "like", "want", "need", "think", "know", "see", "look", "make",
        "go", "come", "take", "give", "get", "use", "find", "tell", "ask",
        "cat", "dog", "bird", "fish", "animal", "person", "man", "woman", "child",
        "learning", "AI", "models", "Transformer", "attention", "neural", "network",
        "computer", "data", "information", "system", "program", "code", "test"
    ]
    
    idx = 4
    for word in common_words:
        if idx >= vocab_size:
            break
        vocab[idx] = word
        idx += 1
    
    # 填充剩余的词表
    while idx < vocab_size:
        vocab[idx] = f"token_{idx}"
        idx += 1
    
    return vocab


if __name__ == "__main__":
    # 测试代码
    print("创建模型...")
    vocab_size = 500
    d_model = 256
    n_heads = 8
    n_layers = 4
    
    vocab = create_simple_vocab(vocab_size)
    model = TrackedTransformer(vocab_size, d_model, n_heads, n_layers)
    tracker = TokenTracker(model, vocab)
    
    print("追踪 Token 旅程...")
    input_text = "I love learning AI models"
    token_position = 2  # "learning"
    
    journey = tracker.track_token_journey(input_text, token_position)
    
    print(f"\n=== Token Journey for '{journey.token_text}' (Position {journey.token_position}) ===")
    print(f"Token ID: {journey.token_id}")
    print(f"Embedding norm: {np.linalg.norm(journey.embedding):.4f}")
    print(f"\nNumber of layers tracked: {len(journey.layer_states)}")
    
    for layer_state in journey.layer_states:
        print(f"\nLayer {layer_state.layer_idx}:")
        print(f"  Input norm: {layer_state.norm_input:.4f}")
        print(f"  Output norm: {layer_state.norm_output:.4f}")
        print(f"  Change: {layer_state.norm_change:+.4f}")
        if layer_state.residual_delta_1 is not None:
            print(f"  Residual 1 contribution: {np.linalg.norm(layer_state.residual_delta_1):.4f}")
        if layer_state.residual_delta_2 is not None:
            print(f"  Residual 2 contribution: {np.linalg.norm(layer_state.residual_delta_2):.4f}")
    
    print(f"\nFinal hidden norm: {np.linalg.norm(journey.final_hidden):.4f}")
    print(f"\nTop-5 Predictions:")
    for rank, (token_id, token_text, prob) in enumerate(journey.top_k_predictions[:5], 1):
        print(f"  {rank}. {token_text:15s} (ID: {token_id:4d}) - {prob:.4%}")
    
    print("\n✅ Token 追踪器测试成功！")
