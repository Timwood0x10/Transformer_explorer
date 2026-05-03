"""
Attention Residuals (Kimi) Tracker
====================================

Real-time computation engine for Kimi's Attention Residuals architecture.
All numerical results are computed on-the-fly using PyTorch — no mocked data.

Core capabilities:
1. PreNorm Dilution analysis: measure residual stream magnitude growth
2. Full AttnRes: depth-direction softmax attention over all predecessor layers
3. Block AttnRes: chunked version with cross-block attention
4. Gradient flow comparison: standard residual vs AttnRes

Reference: "Attention Residuals" (arXiv:2603.15031) by Moonshot AI (Kimi)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field


# ============================================================
# Data Classes
# ============================================================

@dataclass
class LayerNormStats:
    """Statistics for a single layer's normalization behavior."""
    layer_idx: int
    input_norm: float = 0.0       # L2 norm of input to the layer
    output_norm: float = 0.0      # L2 norm of output from the layer
    sublayer_output_norm: float = 0.0  # L2 norm of F(x) before residual add
    residual_contribution_ratio: float = 0.0  # ||x|| / ||x + F(x)||
    prenorm_scale: float = 0.0    # Scale factor applied by LayerNorm


@dataclass
class PrenormDilutionResult:
    """Result of PreNorm dilution analysis across all layers."""
    n_layers: int
    d_model: int
    layer_stats: List[LayerNormStats] = field(default_factory=list)
    # Aggregate metrics
    input_norms: List[float] = field(default_factory=list)
    output_norms: List[float] = field(default_factory=list)
    residual_ratios: List[float] = field(default_factory=list)
    # Dilution severity
    max_output_norm: float = 0.0
    norm_growth_rate: float = 0.0  # output_norm[-1] / output_norm[0]
    is_diluted: bool = False  # True if norm grows significantly


@dataclass
class AttnResWeightInfo:
    """Attention weight information for a single layer's depth attention."""
    layer_idx: int
    attention_weights: List[float] = field(default_factory=list)  # Weights to predecessor layers
    max_weight_idx: int = 0  # Which predecessor gets the most attention
    max_weight_val: float = 0.0
    entropy: float = 0.0  # Entropy of attention distribution
    is_uniform: bool = False  # True if weights are nearly uniform


@dataclass
class AttnResResult:
    """Result of Attention Residuals computation."""
    n_layers: int
    d_model: int
    block_size: int  # Block size for Block AttnRes (1 = Full AttnRes)
    weight_infos: List[AttnResWeightInfo] = field(default_factory=list)
    # Comparison with standard residual
    output_norms_attnres: List[float] = field(default_factory=list)
    output_norms_standard: List[float] = field(default_factory=list)
    # Gradient flow
    grad_norms_attnres: List[float] = field(default_factory=list)
    grad_norms_standard: List[float] = field(default_factory=list)
    # Metrics
    norm_boundedness_attnres: float = 0.0  # std of output norms (lower = more bounded)
    norm_boundedness_standard: float = 0.0
    grad_uniformity_attnres: float = 0.0  # coefficient of variation of grad norms
    grad_uniformity_standard: float = 0.0


@dataclass
class BlockAttnResResult:
    """Result of Block AttnRes analysis."""
    n_layers: int
    n_blocks: int
    block_size: int
    d_model: int
    # Per-block boundary analysis
    boundary_norms: List[float] = field(default_factory=list)
    boundary_norms_standard: List[float] = field(default_factory=list)
    # Cross-block attention weights (at each block boundary)
    cross_block_weights: List[List[float]] = field(default_factory=list)
    # Two-stage inference simulation
    precompute_flops_ratio: float = 0.0  # FLOPs saved by precomputation
    # Overall comparison
    output_norms: List[float] = field(default_factory=list)
    output_norms_standard: List[float] = field(default_factory=list)


# ============================================================
# Model Components
# ============================================================

class StandardTransformerBlock(nn.Module):
    """Standard Pre-LN Transformer block with fixed residual connections."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        # Attention
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
            nn.Dropout(dropout),
        )

        # LayerNorm (Pre-LN)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)
        self._init_weights()

    def _init_weights(self):
        for m in [self.q_proj, self.k_proj, self.v_proj, self.out_proj]:
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        for m in self.ffn:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """Forward pass, returns output and intermediate stats."""
        stats = {}
        residual = x
        input_norm = x.norm(dim=-1).mean().item()

        # Pre-LN + Attention
        x_normed = self.norm1(x)
        stats["prenorm1_scale"] = (x_normed.norm(dim=-1).mean() / (x.norm(dim=-1).mean() + 1e-8)).item()

        Q = self.q_proj(x_normed)
        K = self.k_proj(x_normed)
        V = self.v_proj(x_normed)

        B, S, D = x.shape
        Q = Q.view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, S, self.n_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_dim)
        attn_weights = F.softmax(scores, dim=-1)
        attn_out = torch.matmul(attn_weights, V)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, S, D)
        attn_out = self.out_proj(attn_out)
        attn_out = self.dropout(attn_out)

        stats["attn_output_norm"] = attn_out.norm(dim=-1).mean().item()

        # Residual add (fixed weight = 1)
        x = residual + attn_out
        residual = x

        # Pre-LN + FFN
        x_normed = self.norm2(x)
        stats["prenorm2_scale"] = (x_normed.norm(dim=-1).mean() / (x.norm(dim=-1).mean() + 1e-8)).item()

        ffn_out = self.ffn(x_normed)
        stats["ffn_output_norm"] = ffn_out.norm(dim=-1).mean().item()

        # Residual add (fixed weight = 1)
        x = residual + ffn_out

        stats["input_norm"] = input_norm
        stats["output_norm"] = x.norm(dim=-1).mean().item()
        stats["residual_ratio"] = input_norm / (x.norm(dim=-1).mean().item() + 1e-8)

        return x, stats


class AttnResTransformerBlock(nn.Module):
    """Transformer block with Attention Residuals (Full or Block).

    Instead of fixed residual x + F(x), uses depth-direction softmax attention
    to dynamically weight predecessor layer outputs.
    """

    def __init__(self, d_model: int, n_heads: int, layer_idx: int,
                 n_layers: int, block_size: int = 1, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.layer_idx = layer_idx
        self.n_layers = n_layers
        self.block_size = block_size  # 1 = Full AttnRes

        # Attention sublayer (same as standard)
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
            nn.Dropout(dropout),
        )

        # LayerNorm
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

        # === Attention Residuals specific parameters ===
        # Pseudo-query for depth attention (learnable, NOT input-dependent)
        # This enables two-stage inference optimization
        if block_size > 1:
            # Block AttnRes: attend to block boundary anchors
            n_keys = (layer_idx // block_size) + 1  # Number of block boundaries so far
        else:
            # Full AttnRes: attend to all predecessor layers
            n_keys = layer_idx + 1

        if n_keys > 0:
            self.depth_pseudo_q = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
            self.depth_k_proj = nn.Linear(d_model, d_model)
            self.depth_v_proj = nn.Linear(d_model, d_model)
            self.depth_out_proj = nn.Linear(d_model, d_model)
        else:
            self.depth_pseudo_q = None
            self.depth_k_proj = None
            self.depth_v_proj = None
            self.depth_out_proj = None

        self._init_weights()

    def _init_weights(self):
        for m in [self.q_proj, self.k_proj, self.v_proj, self.out_proj]:
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        for m in self.ffn:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # Depth attention init (small init for stability)
        if self.depth_k_proj is not None:
            nn.init.xavier_uniform_(self.depth_k_proj.weight, gain=0.02)
            nn.init.zeros_(self.depth_k_proj.bias)
            nn.init.xavier_uniform_(self.depth_v_proj.weight, gain=0.02)
            nn.init.zeros_(self.depth_v_proj.bias)
            nn.init.xavier_uniform_(self.depth_out_proj.weight, gain=0.02)
            nn.init.zeros_(self.depth_out_proj.bias)

    def compute_depth_attention(self, predecessor_states: List[torch.Tensor]) -> Tuple[torch.Tensor, List[float]]:
        """Compute depth-direction attention over predecessor states.

        Args:
            predecessor_states: List of hidden states from predecessor layers/blocks

        Returns:
            (aggregated_state, attention_weights)
        """
        if not predecessor_states or self.depth_pseudo_q is None:
            return None, []

        # Stack predecessor states: (n_predecessors, batch, seq, d_model)
        stacked = torch.stack(predecessor_states, dim=0)

        # Compute keys and values for each predecessor
        keys = self.depth_k_proj(stacked)  # (n_pred, B, S, D)
        values = self.depth_v_proj(stacked)  # (n_pred, B, S, D)

        # Pseudo-query (broadcast over batch and seq)
        # depth_pseudo_q shape: (1, 1, D) -> (B, S, D)
        B, S, D = keys.shape[1], keys.shape[2], keys.shape[3]
        q = self.depth_pseudo_q.expand(B, S, D)  # (B, S, D)

        # Keys for attention: (n_pred, B, S, D) -> transpose to (B, n_pred, S, D) -> mean over S -> (B, n_pred, D)
        # We use mean-pooled keys across sequence for depth attention
        keys_pooled = keys.transpose(0, 1).mean(dim=2)  # (B, n_pred, D)
        values_pooled = values.transpose(0, 1).mean(dim=2)  # (B, n_pred, D)

        # Attention scores: (B, S, D) @ (B, D, n_pred) -> (B, S, n_pred)
        scores = torch.bmm(q, keys_pooled.transpose(1, 2)) / np.sqrt(D)  # (B, S, n_pred)
        attn_weights = F.softmax(scores, dim=-1)  # (B, S, n_pred)

        # Weighted aggregation: (B, S, n_pred) @ (B, n_pred, D) -> (B, S, D)
        aggregated = torch.bmm(attn_weights, values_pooled)  # (B, S, D)

        # Output projection
        aggregated = self.depth_out_proj(aggregated)

        # Return mean attention weights for analysis
        mean_weights = attn_weights.mean(dim=(0, 1)).tolist()

        return aggregated, mean_weights

    def forward(self, x: torch.Tensor,
                predecessor_states: Optional[List[torch.Tensor]] = None) -> Tuple[torch.Tensor, Dict]:
        """Forward pass with attention residuals.

        Args:
            x: Current input (B, S, D)
            predecessor_states: Hidden states from predecessor layers (for AttnRes)

        Returns:
            (output, stats)
        """
        stats = {}
        input_norm = x.norm(dim=-1).mean().item()

        # === Depth Attention Residual ===
        attnres_state, depth_weights = self.compute_depth_attention(predecessor_states)
        stats["depth_attention_weights"] = depth_weights

        if attnres_state is not None:
            # Use attention-aggregated state as the residual stream input
            residual = attnres_state
            stats["attnres_norm"] = attnres_state.norm(dim=-1).mean().item()
        else:
            # Fallback to standard residual for the first layer
            residual = x
            stats["attnres_norm"] = input_norm

        # === Self-Attention ===
        x_normed = self.norm1(residual)
        Q = self.q_proj(x_normed)
        K = self.k_proj(x_normed)
        V = self.v_proj(x_normed)

        B, S, D = x.shape
        Q = Q.view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, S, self.n_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_dim)
        attn_w = F.softmax(scores, dim=-1)
        attn_out = torch.matmul(attn_w, V)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, S, D)
        attn_out = self.out_proj(attn_out)
        attn_out = self.dropout(attn_out)

        stats["attn_output_norm"] = attn_out.norm(dim=-1).mean().item()

        # Residual add within block (standard)
        x = residual + attn_out
        residual2 = x

        # === FFN ===
        x_normed = self.norm2(x)
        ffn_out = self.ffn(x_normed)
        stats["ffn_output_norm"] = ffn_out.norm(dim=-1).mean().item()

        x = residual2 + ffn_out

        stats["input_norm"] = input_norm
        stats["output_norm"] = x.norm(dim=-1).mean().item()

        return x, stats


# ============================================================
# Analysis Engine
# ============================================================

class AttnResTracker:
    """Main analysis engine for Attention Residuals.

    Provides real-time computation for:
    1. PreNorm dilution measurement
    2. Full AttnRes vs standard residual comparison
    3. Block AttnRes analysis
    4. Gradient flow comparison
    """

    def __init__(self, d_model: int = 256, n_heads: int = 4,
                 n_layers: int = 8, block_size: int = 1):
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.block_size = block_size
        self.device = torch.device("cpu")

    def _create_standard_model(self) -> nn.ModuleList:
        """Create standard Transformer layers."""
        layers = nn.ModuleList([
            StandardTransformerBlock(self.d_model, self.n_heads)
            for _ in range(self.n_layers)
        ])
        return layers

    def _create_attnres_model(self, block_size: int = 1) -> nn.ModuleList:
        """Create AttnRes Transformer layers."""
        layers = nn.ModuleList([
            AttnResTransformerBlock(
                self.d_model, self.n_heads, i, self.n_layers, block_size
            )
            for i in range(self.n_layers)
        ])
        return layers

    # ----------------------------------------------------------
    # 1. PreNorm Dilution Analysis
    # ----------------------------------------------------------
    def analyze_prenorm_dilution(self, batch_size: int = 2, seq_len: int = 16) -> PrenormDilutionResult:
        """Analyze PreNorm dilution: measure how residual stream magnitude grows.

        In standard Pre-LN Transformers, the residual stream norm grows
        monotonically with depth because each layer adds F(x) with fixed weight 1.
        This causes 'PreNorm Dilution' where deep layer contributions get diluted.

        Returns:
            PrenormDilutionResult with per-layer statistics
        """
        torch.manual_seed(42)
        layers = self._create_standard_model()
        layers.eval()

        # Random input
        x = torch.randn(batch_size, seq_len, self.d_model) * 0.02

        result = PrenormDilutionResult(
            n_layers=self.n_layers,
            d_model=self.d_model,
        )

        with torch.no_grad():
            for i, layer in enumerate(layers):
                x, stats = layer(x)

                layer_stat = LayerNormStats(
                    layer_idx=i,
                    input_norm=stats["input_norm"],
                    output_norm=stats["output_norm"],
                    sublayer_output_norm=stats.get("attn_output_norm", 0) + stats.get("ffn_output_norm", 0),
                    residual_contribution_ratio=stats["residual_ratio"],
                    prenorm_scale=stats.get("prenorm1_scale", 0),
                )
                result.layer_stats.append(layer_stat)
                result.input_norms.append(stats["input_norm"])
                result.output_norms.append(stats["output_norm"])
                result.residual_ratios.append(stats["residual_ratio"])

        # Aggregate metrics
        if result.output_norms:
            result.max_output_norm = max(result.output_norms)
            if result.output_norms[0] > 1e-8:
                result.norm_growth_rate = result.output_norms[-1] / result.output_norms[0]
            # Consider diluted if norm grows more than 2x
            result.is_diluted = result.norm_growth_rate > 2.0

        return result

    # ----------------------------------------------------------
    # 2. Full AttnRes vs Standard Residual
    # ----------------------------------------------------------
    def compare_attnres_vs_standard(self, batch_size: int = 2, seq_len: int = 16,
                                     block_size: int = 1) -> AttnResResult:
        """Compare Full/Block AttnRes with standard residual connections.

        Both models use the same random seed for fair comparison.
        Measures output norm boundedness and gradient flow uniformity.

        Returns:
            AttnResResult with comparison data
        """
        torch.manual_seed(42)

        # --- Standard model forward pass ---
        std_layers = self._create_standard_model()
        std_layers.eval()

        x_std = torch.randn(batch_size, seq_len, self.d_model) * 0.02
        std_output_norms = []

        with torch.no_grad():
            for layer in std_layers:
                x_std, stats = layer(x_std)
                std_output_norms.append(stats["output_norm"])

        # --- AttnRes model forward pass ---
        torch.manual_seed(42)
        attnres_layers = self._create_attnres_model(block_size=block_size)
        attnres_layers.eval()

        x_attnres = torch.randn(batch_size, seq_len, self.d_model) * 0.02
        predecessor_states = []
        attnres_output_norms = []
        all_weight_infos = []

        with torch.no_grad():
            for i, layer in enumerate(attnres_layers):
                # Collect predecessor states
                if block_size > 1:
                    # Block AttnRes: use block boundary states
                    preds = []
                    for b_start in range(0, i, block_size):
                        preds.append(predecessor_states[b_start])
                    if not preds:
                        preds = None
                else:
                    # Full AttnRes: use all predecessor layer outputs
                    preds = list(predecessor_states) if predecessor_states else None

                x_attnres, stats = layer(x_attnres, preds)
                predecessor_states.append(x_attnres.detach().clone())
                attnres_output_norms.append(stats["output_norm"])

                # Record depth attention weights
                weights = stats.get("depth_attention_weights", [])
                if weights:
                    entropy = self._compute_entropy(weights)
                    max_idx = int(np.argmax(weights))
                    weight_info = AttnResWeightInfo(
                        layer_idx=i,
                        attention_weights=weights,
                        max_weight_idx=max_idx,
                        max_weight_val=weights[max_idx],
                        entropy=entropy,
                        is_uniform=self._is_uniform(weights),
                    )
                    all_weight_infos.append(weight_info)

        # --- Gradient flow comparison ---
        std_grad_norms = self._compute_gradient_norms_std(batch_size, seq_len)
        attnres_grad_norms = self._compute_gradient_norms_attnres(batch_size, seq_len, block_size)

        # --- Compute metrics ---
        result = AttnResResult(
            n_layers=self.n_layers,
            d_model=self.d_model,
            block_size=block_size,
            weight_infos=all_weight_infos,
            output_norms_attnres=attnres_output_norms,
            output_norms_standard=std_output_norms,
            grad_norms_attnres=attnres_grad_norms,
            grad_norms_standard=std_grad_norms,
        )

        # Norm boundedness (std of output norms — lower = more bounded)
        if attnres_output_norms:
            result.norm_boundedness_attnres = float(np.std(attnres_output_norms))
        if std_output_norms:
            result.norm_boundedness_standard = float(np.std(std_output_norms))

        # Gradient uniformity (coefficient of variation — lower = more uniform)
        if attnres_grad_norms and np.mean(attnres_grad_norms) > 1e-10:
            result.grad_uniformity_attnres = float(np.std(attnres_grad_norms) / np.mean(attnres_grad_norms))
        if std_grad_norms and np.mean(std_grad_norms) > 1e-10:
            result.grad_uniformity_standard = float(np.std(std_grad_norms) / np.mean(std_grad_norms))

        return result

    # ----------------------------------------------------------
    # 3. Block AttnRes Analysis
    # ----------------------------------------------------------
    def analyze_block_attnres(self, batch_size: int = 2, seq_len: int = 16,
                               block_size: int = 4) -> BlockAttnResResult:
        """Analyze Block AttnRes with different block sizes.

        Block AttnRes divides layers into chunks. Within each chunk,
        standard residual is used. At chunk boundaries, depth-direction
        softmax attention aggregates previous chunk outputs.

        Args:
            block_size: Number of layers per block (e.g., 4 means 8 layers -> 2 blocks)

        Returns:
            BlockAttnResResult with per-block analysis
        """
        n_blocks = max(1, (self.n_layers + block_size - 1) // block_size)

        torch.manual_seed(42)

        # Standard model
        std_layers = self._create_standard_model()
        std_layers.eval()
        x_std = torch.randn(batch_size, seq_len, self.d_model) * 0.02
        std_norms = []
        with torch.no_grad():
            for layer in std_layers:
                x_std, stats = layer(x_std)
                std_norms.append(stats["output_norm"])

        # Block AttnRes model
        torch.manual_seed(42)
        attnres_layers = self._create_attnres_model(block_size=block_size)
        attnres_layers.eval()
        x_attnres = torch.randn(batch_size, seq_len, self.d_model) * 0.02

        block_boundary_states = []  # State at each block boundary
        attnres_norms = []
        boundary_norms = []
        boundary_norms_std = []
        cross_block_weights_all = []

        with torch.no_grad():
            for i, layer in enumerate(attnres_layers):
                # Determine predecessors (block boundary anchors)
                preds = list(block_boundary_states) if block_boundary_states else None

                x_attnres, stats = layer(x_attnres, preds)
                attnres_norms.append(stats["output_norm"])

                # At block boundary, save the state
                if (i + 1) % block_size == 0 or i == self.n_layers - 1:
                    block_boundary_states.append(x_attnres.detach().clone())
                    boundary_norms.append(stats["output_norm"])
                    # Corresponding standard norm
                    std_idx = min(i, len(std_norms) - 1)
                    boundary_norms_std.append(std_norms[std_idx])

                    # Record cross-block attention weights
                    weights = stats.get("depth_attention_weights", [])
                    cross_block_weights_all.append(weights)

        # Two-stage inference FLOPs analysis
        # Precomputation: pseudo-query is fixed, so all attention scores
        # can be computed in one batch before forward pass
        # Savings = (n_layers - 1) redundant attention computations avoided
        if n_blocks > 1:
            total_depth_attn_computations = self.n_layers  # Naive: compute at every layer
            precomputed_computations = n_blocks  # Smart: compute once per block boundary
            precompute_flops_ratio = precomputed_computations / total_depth_attn_computations
        else:
            precompute_flops_ratio = 1.0

        return BlockAttnResResult(
            n_layers=self.n_layers,
            n_blocks=n_blocks,
            block_size=block_size,
            d_model=self.d_model,
            boundary_norms=boundary_norms,
            boundary_norms_standard=boundary_norms_std,
            cross_block_weights=cross_block_weights_all,
            precompute_flops_ratio=precompute_flops_ratio,
            output_norms=attnres_norms,
            output_norms_standard=std_norms,
        )

    # ----------------------------------------------------------
    # 4. Gradient Flow Comparison
    # ----------------------------------------------------------
    def _compute_gradient_norms_std(self, batch_size: int, seq_len: int) -> List[float]:
        """Compute per-layer gradient norms for standard residual model."""
        torch.manual_seed(42)
        layers = self._create_standard_model()
        layers.train()

        x = torch.randn(batch_size, seq_len, self.d_model, requires_grad=True)
        target = torch.randn(batch_size, seq_len, self.d_model)

        # Forward
        for layer in layers:
            x, _ = layer(x)

        # Backward
        loss = F.mse_loss(x, target)
        loss.backward()

        # Collect gradient norms per layer
        grad_norms = []
        for layer in layers:
            grads = []
            for name, param in layer.named_parameters():
                if param.grad is not None:
                    grads.append(param.grad.norm().item())
            grad_norms.append(np.mean(grads) if grads else 0.0)

        return grad_norms

    def _compute_gradient_norms_attnres(self, batch_size: int, seq_len: int,
                                         block_size: int = 1) -> List[float]:
        """Compute per-layer gradient norms for AttnRes model."""
        torch.manual_seed(42)
        layers = self._create_attnres_model(block_size=block_size)
        layers.train()

        x = torch.randn(batch_size, seq_len, self.d_model, requires_grad=True)
        target = torch.randn(batch_size, seq_len, self.d_model)

        predecessor_states = []

        # Forward
        for i, layer in enumerate(layers):
            if block_size > 1:
                preds = []
                for b_start in range(0, i, block_size):
                    preds.append(predecessor_states[b_start])
                if not preds:
                    preds = None
            else:
                preds = list(predecessor_states) if predecessor_states else None

            x, _ = layer(x, preds)
            predecessor_states.append(x.detach())

        # Backward
        loss = F.mse_loss(x, target)
        loss.backward()

        # Collect gradient norms per layer
        grad_norms = []
        for layer in layers:
            grads = []
            for name, param in layer.named_parameters():
                if param.grad is not None:
                    grads.append(param.grad.norm().item())
            grad_norms.append(np.mean(grads) if grads else 0.0)

        return grad_norms

    # ----------------------------------------------------------
    # Utility methods
    # ----------------------------------------------------------
    @staticmethod
    def _compute_entropy(weights: List[float]) -> float:
        """Compute Shannon entropy of attention weights."""
        w = np.array(weights)
        w = w / (w.sum() + 1e-10)
        w = w[w > 1e-10]
        return float(-np.sum(w * np.log(w + 1e-10)))

    @staticmethod
    def _is_uniform(weights: List[float], threshold: float = 0.1) -> bool:
        """Check if attention weights are approximately uniform."""
        if not weights:
            return False
        w = np.array(weights)
        expected = 1.0 / len(w)
        return float(np.std(w - expected)) < threshold

    def scan_block_sizes(self, batch_size: int = 2, seq_len: int = 16,
                          block_sizes: Optional[List[int]] = None) -> Dict:
        """Scan performance across different block sizes.

        Returns a dictionary mapping block_size -> BlockAttnResResult
        """
        if block_sizes is None:
            block_sizes = [1, 2, 4, 8]
        # Filter to valid block sizes
        block_sizes = [bs for bs in block_sizes if 1 <= bs <= self.n_layers]

        results = {}
        for bs in block_sizes:
            results[bs] = self.analyze_block_attnres(batch_size, seq_len, bs)

        return results
