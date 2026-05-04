"""
Shared Transformer Base Module
===============================

Provides reusable, well-documented base classes for Transformer components.
These classes contain NO tracking/hook/analysis logic -- they are pure model
building blocks that specialized modules (in token_tracker, gradient_tracker,
attnres_tracker, etc.) can inherit from or wrap.

Classes:
    MultiHeadAttention  - Standard multi-head self-attention with Q/K/V projections
    FeedForward         - Standard FFN with GELU activation
    TransformerLayer    - Standard Pre-LN Transformer layer (attention + FFN)
    BaseTransformer     - Complete Transformer with embedding, positional encoding,
                          layers, and output projection
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class MultiHeadAttention(nn.Module):
    """Standard multi-head self-attention mechanism.

    Projects input into Q (query), K (key), V (value) vectors, computes
    scaled dot-product attention across multiple heads, then projects the
    concatenated output back to the model dimension.

    Args:
        d_model: Model hidden dimension.
        n_heads: Number of attention heads. d_model must be divisible by n_heads.
        dropout: Dropout probability on attention weights.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        # Q, K, V projections
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)

        # Output projection
        self.out_proj = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

        self._init_weights()

    def _init_weights(self):
        """Initialize linear layers with xavier_uniform."""
        for module in [self.q_proj, self.k_proj, self.v_proj, self.out_proj]:
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute multi-head self-attention.

        Args:
            x: Input tensor of shape (batch, seq_len, d_model).
            mask: Optional attention mask. Shape broadcastable to
                  (batch, n_heads, seq_len, seq_len). A value of 0/False
                  masks out the corresponding position.

        Returns:
            Output tensor of shape (batch, seq_len, d_model).
        """
        B, S, D = x.shape

        # Project to Q, K, V
        Q = self.q_proj(x)  # (B, S, D)
        K = self.k_proj(x)
        V = self.v_proj(x)

        # Reshape to (B, n_heads, S, head_dim)
        Q = Q.view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, S, self.n_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attn_weights = F.softmax(scores, dim=-1)  # (B, n_heads, S, S)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        attn_output = torch.matmul(attn_weights, V)  # (B, n_heads, S, head_dim)

        # Concatenate heads: (B, S, D)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, S, D)

        # Output projection
        output = self.out_proj(attn_output)
        return output


class FeedForward(nn.Module):
    """Standard feed-forward network (FFN) with GELU activation.

    Architecture: Linear(d_model, 4*d_model) -> GELU -> Dropout ->
                  Linear(4*d_model, d_model) -> Dropout

    Args:
        d_model: Model hidden dimension.
        dropout: Dropout probability.
    """

    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )
        self._init_weights()

    def _init_weights(self):
        """Initialize linear layers with xavier_uniform."""
        for module in self.net:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch, seq_len, d_model).

        Returns:
            Output tensor of shape (batch, seq_len, d_model).
        """
        return self.net(x)


class TransformerLayer(nn.Module):
    """Standard Pre-LN Transformer layer.

    Uses Pre-LayerNorm (Pre-LN) placement: LayerNorm is applied before
    the attention and FFN sub-layers, which generally provides better
    training stability than Post-LN.

    Architecture:
        x_normed = LayerNorm(x)
        x = x + MultiHeadAttention(x_normed)
        x_normed = LayerNorm(x)
        x = x + FeedForward(x_normed)

    Args:
        d_model: Model hidden dimension.
        n_heads: Number of attention heads.
        dropout: Dropout probability.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attention = MultiHeadAttention(d_model, n_heads, dropout)

        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass with residual connections.

        Args:
            x: Input tensor of shape (batch, seq_len, d_model).
            mask: Optional attention mask.

        Returns:
            Output tensor of shape (batch, seq_len, d_model).
        """
        # Pre-LN Self-Attention with residual connection
        x = x + self.attention(self.norm1(x), mask)

        # Pre-LN FFN with residual connection
        x = x + self.ffn(self.norm2(x))

        return x


class BaseTransformer(nn.Module):
    """Complete Transformer model for language modeling.

    This is a clean, reusable base implementation with:
    - Token embedding
    - Learnable positional embedding
    - Stack of Pre-LN Transformer layers
    - Output projection (language model head)

    Args:
        d_model: Model hidden dimension.
        n_heads: Number of attention heads.
        n_layers: Number of Transformer layers.
        vocab_size: Vocabulary size.
        max_seq_len: Maximum sequence length for positional embeddings.
        dropout: Dropout probability.
    """

    def __init__(self, d_model: int, n_heads: int, n_layers: int,
                 vocab_size: int, max_seq_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len

        # Token embedding
        self.token_embedding = nn.Embedding(vocab_size, d_model)

        # Learnable positional embedding
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)

        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerLayer(d_model, n_heads, dropout)
            for _ in range(n_layers)
        ])

        # Output projection (language model head)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        self.dropout = nn.Dropout(dropout)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize all weights.

        - Embeddings: normal distribution with std=0.02
        - Linear layers: xavier_uniform (handled by sub-modules)
        - LM head: normal distribution with std=0.02
        """
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.pos_embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.02)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Forward pass producing logits over the vocabulary.

        Args:
            input_ids: Token IDs of shape (batch, seq_len).

        Returns:
            Logits of shape (batch, seq_len, vocab_size).
        """
        batch_size, seq_len = input_ids.shape

        # Build position indices: (1, seq_len) -> (batch, seq_len)
        position_ids = torch.arange(seq_len, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)

        # Embeddings: token + positional
        x = self.token_embedding(input_ids) + self.pos_embedding(position_ids)
        x = self.dropout(x)

        # Pass through all Transformer layers
        for layer in self.layers:
            x = layer(x)

        # Output projection
        logits = self.lm_head(x)  # (batch, seq_len, vocab_size)

        return logits
