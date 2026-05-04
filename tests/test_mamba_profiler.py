"""
Tests for utils.mamba_profiler module.
"""
import sys
import os
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from utils.mamba_profiler import create_sample_mamba, MambaProfiler


@pytest.fixture(autouse=True)
def set_seed():
    torch.manual_seed(42)


@pytest.fixture
def mamba_profiler(sample_transformer_config):
    """Create a MambaProfiler with small test dimensions."""
    return MambaProfiler(
        d_model=sample_transformer_config['d_model'],
        d_state=16,
        d_conv=4,
        expand=2,
    )


class TestCreateSampleMamba:
    """Tests for create_sample_mamba."""

    def test_create_sample_mamba(self):
        """Returns dict with expected keys."""
        result = create_sample_mamba(d_model=128, d_state=16, d_conv=4, expand=2)

        assert isinstance(result, dict)
        for key in ['d_model', 'd_state', 'd_conv', 'expand', 'd_inner']:
            assert key in result, f"Missing key: {key}"

        assert result['d_model'] == 128
        assert result['d_state'] == 16
        assert result['d_conv'] == 4
        assert result['expand'] == 2
        assert result['d_inner'] == 128 * 2


class TestCountParameters:
    """Tests for MambaProfiler.count_parameters."""

    def test_count_parameters(self, mamba_profiler):
        """Total > 0 and has ssm_parameters key."""
        params = mamba_profiler.count_parameters()

        assert 'ssm_parameters' in params
        assert params['ssm_parameters'] > 0
        assert params['total_per_block'] > 0


class TestEstimateFlops:
    """Tests for MambaProfiler.estimate_flops."""

    def test_estimate_flops(self, mamba_profiler, sample_batch):
        """Returns dict with total_per_block and complexity='O(BLdN)'."""
        result = mamba_profiler.estimate_flops(
            batch_size=sample_batch['batch_size'],
            seq_len=sample_batch['seq_len'],
        )

        assert 'total_per_block' in result
        assert result['total_per_block'] > 0
        assert result['complexity'] == 'O(BLdN)'


class TestEstimateMemory:
    """Tests for MambaProfiler.estimate_memory."""

    def test_estimate_memory(self, mamba_profiler, sample_batch):
        """Returns dict with parameters_mb and values > 0."""
        result = mamba_profiler.estimate_memory(
            batch_size=sample_batch['batch_size'],
            seq_len=sample_batch['seq_len'],
            dtype=torch.float32,
        )

        assert 'parameters_mb' in result
        assert result['parameters_mb'] > 0
        assert result['total_training_mb'] > 0


class TestCompareWithTransformer:
    """Tests for MambaProfiler.compare_with_transformer."""

    def test_compare_with_transformer(self, mamba_profiler, sample_batch):
        """Returns dict with flops_speedup, mamba_flops, transformer_flops."""
        result = mamba_profiler.compare_with_transformer(
            batch_size=sample_batch['batch_size'],
            seq_len=sample_batch['seq_len'],
            n_heads=4,
        )

        assert 'flops_speedup' in result
        assert 'mamba_flops' in result
        assert 'transformer_flops' in result
        assert result['mamba_flops'] > 0
        assert result['transformer_flops'] > 0
        assert result['flops_speedup'] > 0

    def test_compare_transformer_longer_seq_faster(self, mamba_profiler):
        """At long seq_len, mamba should have flops_speedup > 1."""
        result = mamba_profiler.compare_with_transformer(
            batch_size=2,
            seq_len=2048,
            n_heads=4,
        )

        assert result['flops_speedup'] > 1.0, \
            f"Expected mamba speedup > 1 at long seq_len, got {result['flops_speedup']}"


class TestAnalyzeScaling:
    """Tests for MambaProfiler.analyze_scaling."""

    def test_analyze_scaling(self, mamba_profiler, sample_batch):
        """Returns dict with mamba_flops list and transformer_flops list."""
        seq_lengths = [64, 128, 256, 512]
        result = mamba_profiler.analyze_scaling(seq_lengths, batch_size=sample_batch['batch_size'])

        assert 'mamba_flops' in result
        assert 'transformer_flops' in result
        assert isinstance(result['mamba_flops'], list)
        assert isinstance(result['transformer_flops'], list)
        assert len(result['mamba_flops']) == len(seq_lengths)
        assert len(result['transformer_flops']) == len(seq_lengths)

    def test_analyze_scaling_mamba_linear(self, mamba_profiler, sample_batch):
        """Mamba FLOPs should grow linearly with seq_len."""
        seq_lengths = [64, 128, 256]
        result = mamba_profiler.analyze_scaling(seq_lengths, batch_size=sample_batch['batch_size'])

        mamba_flops = result['mamba_flops']

        # When seq_len doubles, mamba flops should also roughly double (linear)
        ratio_1 = mamba_flops[1] / mamba_flops[0]  # 128 / 64
        ratio_2 = mamba_flops[2] / mamba_flops[1]  # 256 / 128

        assert abs(ratio_1 - 2.0) < 0.5, f"Mamba FLOPs ratio (128/64): {ratio_1}, expected ~2.0"
        assert abs(ratio_2 - 2.0) < 0.5, f"Mamba FLOPs ratio (256/128): {ratio_2}, expected ~2.0"


class TestGetSelectiveScanAnalysis:
    """Tests for MambaProfiler.get_selective_scan_analysis."""

    def test_get_selective_scan_analysis(self, mamba_profiler):
        """Returns dict with key_innovation, advantages, complexity."""
        result = mamba_profiler.get_selective_scan_analysis()

        assert 'key_innovation' in result
        assert 'advantages' in result
        assert 'complexity' in result
        assert isinstance(result['advantages'], list)
        assert len(result['advantages']) > 0
        assert result['complexity'] == 'O(BLdN)'
