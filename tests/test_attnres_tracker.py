import sys
import os
import math

import pytest
import torch
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from utils.attnres_tracker import AttnResTracker, PrenormDilutionResult, AttnResResult, BlockAttnResResult

torch.manual_seed(42)


@pytest.fixture
def tracker():
    """Small tracker for fast tests."""
    return AttnResTracker(d_model=64, n_heads=4, n_layers=8, block_size=2)


# -------------------------------------------------------
# 1. PreNorm Dilution tests
# -------------------------------------------------------

class TestPrenormDilution:
    def test_prenorm_dilution_norm_growth(self, tracker):
        """Norm should grow with depth in a standard pre-LN transformer."""
        result = tracker.analyze_prenorm_dilution(batch_size=2, seq_len=8)
        assert result.norm_growth_rate > 1.0, (
            f"Expected norm_growth_rate > 1.0, got {result.norm_growth_rate}"
        )

    def test_prenorm_dilution_layer_count(self, tracker):
        """Number of layer stats should equal n_layers."""
        result = tracker.analyze_prenorm_dilution(batch_size=2, seq_len=8)
        assert len(result.layer_stats) == tracker.n_layers, (
            f"Expected {tracker.n_layers} layer stats, got {len(result.layer_stats)}"
        )

    def test_prenorm_dilution_residual_ratios(self, tracker):
        """All residual contribution ratios should be in (0, 1]."""
        result = tracker.analyze_prenorm_dilution(batch_size=2, seq_len=8)
        for ratio in result.residual_ratios:
            assert 0 < ratio <= 1.0, f"Residual ratio {ratio} not in (0, 1]"

    def test_prenorm_dilution_is_diluted(self, tracker):
        """With enough layers, the residual stream should be diluted."""
        result = tracker.analyze_prenorm_dilution(batch_size=2, seq_len=8)
        assert result.is_diluted is True, (
            "Expected is_diluted=True for 8-layer model"
        )


# -------------------------------------------------------
# 2. AttnRes vs Standard tests
# -------------------------------------------------------

class TestAttnResVsStandard:
    def test_attnres_vs_standard_norm_boundedness(self, tracker):
        """AttnRes output norm std should be smaller than standard's."""
        result = tracker.compare_attnres_vs_standard(batch_size=2, seq_len=8, block_size=1)
        assert result.norm_boundedness_attnres < result.norm_boundedness_standard, (
            f"AttnRes boundedness {result.norm_boundedness_attnres:.4f} "
            f"should be < standard {result.norm_boundedness_standard:.4f}"
        )

    def test_attnres_vs_standard_output_shapes(self, tracker):
        """Both output norm lists should have the same length (= n_layers)."""
        result = tracker.compare_attnres_vs_standard(batch_size=2, seq_len=8, block_size=1)
        assert len(result.output_norms_attnres) == len(result.output_norms_standard), (
            f"Mismatched lengths: attnres={len(result.output_norms_attnres)}, "
            f"standard={len(result.output_norms_standard)}"
        )

    def test_attnres_weight_infos(self, tracker):
        """weight_infos should be non-empty for layers > 0 (layers 1..n_layers-1 have predecessors)."""
        result = tracker.compare_attnres_vs_standard(batch_size=2, seq_len=8, block_size=1)
        assert len(result.weight_infos) > 0, "weight_infos should not be empty"

    def test_attnres_entropy_non_negative(self, tracker):
        """All entropy values in weight_infos should be >= 0."""
        result = tracker.compare_attnres_vs_standard(batch_size=2, seq_len=8, block_size=1)
        for wi in result.weight_infos:
            assert wi.entropy >= 0, f"Entropy {wi.entropy} for layer {wi.layer_idx} is negative"


# -------------------------------------------------------
# 3. Block AttnRes tests
# -------------------------------------------------------

class TestBlockAttnRes:
    def test_block_attnres_block_count(self, tracker):
        """Number of blocks should equal ceil(n_layers / block_size)."""
        block_size = 3
        expected_blocks = math.ceil(tracker.n_layers / block_size)
        result = tracker.analyze_block_attnres(batch_size=2, seq_len=8, block_size=block_size)
        assert result.n_blocks == expected_blocks, (
            f"Expected {expected_blocks} blocks, got {result.n_blocks}"
        )

    def test_block_attnres_precompute_ratio(self, tracker):
        """precompute_flops_ratio should be in (0, 1]."""
        block_size = 3
        result = tracker.analyze_block_attnres(batch_size=2, seq_len=8, block_size=block_size)
        assert 0 < result.precompute_flops_ratio <= 1.0, (
            f"precompute_flops_ratio {result.precompute_flops_ratio} not in (0, 1]"
        )


# -------------------------------------------------------
# 4. Scan block sizes tests
# -------------------------------------------------------

class TestScanBlockSizes:
    def test_scan_block_sizes(self, tracker):
        """scan_block_sizes should return a dict with block_size integer keys."""
        block_sizes = [1, 2, 4]
        results = tracker.scan_block_sizes(batch_size=2, seq_len=8, block_sizes=block_sizes)
        assert isinstance(results, dict), "Result should be a dict"
        for bs in block_sizes:
            assert bs in results, f"block_size {bs} missing from results"
            assert isinstance(results[bs], BlockAttnResResult), (
                f"Result for block_size {bs} should be a BlockAttnResResult"
            )

    def test_scan_smaller_block_more_stable(self, tracker):
        """Smaller block_size should produce more bounded (lower std) output norms."""
        results = tracker.scan_block_sizes(batch_size=2, seq_len=8, block_sizes=[1, 4])
        std_small = float(np.std(results[1].output_norms))
        std_large = float(np.std(results[4].output_norms))
        assert std_small <= std_large, (
            f"Smaller block_size std ({std_small:.4f}) should be <= "
            f"larger block_size std ({std_large:.4f})"
        )
