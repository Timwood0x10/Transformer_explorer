"""
Tests for utils.model_profiler module.
"""
import sys
import os
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from utils.model_profiler import create_sample_transformer, TransformerProfiler, LayerProfile


@pytest.fixture(autouse=True)
def set_seed():
    torch.manual_seed(42)


class TestCreateSampleTransformer:
    """Tests for create_sample_transformer."""

    def test_create_sample_transformer(self, sample_transformer_config, sample_batch):
        """Model can be created and forward pass works."""
        cfg = sample_transformer_config
        model = create_sample_transformer(
            d_model=cfg['d_model'],
            n_heads=cfg['n_heads'],
            n_layers=cfg['n_layers'],
            vocab_size=cfg['vocab_size'],
        )
        # Forward pass should produce output of shape (B, S, V)
        input_ids = torch.randint(0, cfg['vocab_size'], (sample_batch['batch_size'], sample_batch['seq_len']))
        output = model(input_ids)
        assert output.shape == (sample_batch['batch_size'], sample_batch['seq_len'], cfg['vocab_size'])


class TestCountParameters:
    """Tests for TransformerProfiler.count_parameters."""

    def test_count_parameters(self, sample_transformer_config):
        """Total parameter count > 0 and result contains 'total' key."""
        cfg = sample_transformer_config
        model = create_sample_transformer(**cfg)
        profiler = TransformerProfiler(model, (2, 16, cfg['d_model']))
        params = profiler.count_parameters()

        assert 'total' in params
        assert params['total'] > 0


class TestEstimateFlops:
    """Tests for TransformerProfiler.estimate_flops."""

    def test_estimate_flops(self, sample_transformer_config, sample_batch):
        """Returns dict with expected keys and all values > 0."""
        cfg = sample_transformer_config
        model = create_sample_transformer(**cfg)
        profiler = TransformerProfiler(model, (sample_batch['batch_size'], sample_batch['seq_len'], cfg['d_model']))

        result = profiler.estimate_flops(
            batch_size=sample_batch['batch_size'],
            seq_len=sample_batch['seq_len'],
            d_model=cfg['d_model'],
            n_heads=cfg['n_heads'],
            n_layers=cfg['n_layers'],
            vocab_size=cfg['vocab_size'],
        )

        # Check expected keys exist
        for key in ['attention_per_layer', 'ffn_per_layer', 'total_per_layer', 'total_model',
                     'qkv_projection', 'attention_matrix', 'ffn_total']:
            assert key in result, f"Missing key: {key}"

        # All flops values should be positive
        for key, val in result.items():
            assert val > 0, f"Expected {key} > 0, got {val}"


class TestEstimateMemory:
    """Tests for TransformerProfiler.estimate_memory."""

    def test_estimate_memory(self, sample_transformer_config, sample_batch):
        """Returns dict with expected memory keys and values > 0."""
        cfg = sample_transformer_config
        model = create_sample_transformer(**cfg)
        profiler = TransformerProfiler(model, (sample_batch['batch_size'], sample_batch['seq_len'], cfg['d_model']))

        result = profiler.estimate_memory(
            batch_size=sample_batch['batch_size'],
            seq_len=sample_batch['seq_len'],
            d_model=cfg['d_model'],
            n_layers=cfg['n_layers'],
            dtype=torch.float32,
        )

        for key in ['parameters_mb', 'activation_total_mb', 'total_training_mb']:
            assert key in result, f"Missing key: {key}"
            assert result[key] > 0, f"Expected {key} > 0, got {result[key]}"

    def test_estimate_memory_dtypes(self, sample_transformer_config, sample_batch):
        """FP16 should use less memory than FP32."""
        cfg = sample_transformer_config
        model = create_sample_transformer(**cfg)
        profiler = TransformerProfiler(model, (sample_batch['batch_size'], sample_batch['seq_len'], cfg['d_model']))

        mem_fp32 = profiler.estimate_memory(
            batch_size=sample_batch['batch_size'],
            seq_len=sample_batch['seq_len'],
            d_model=cfg['d_model'],
            n_layers=cfg['n_layers'],
            dtype=torch.float32,
        )
        mem_fp16 = profiler.estimate_memory(
            batch_size=sample_batch['batch_size'],
            seq_len=sample_batch['seq_len'],
            d_model=cfg['d_model'],
            n_layers=cfg['n_layers'],
            dtype=torch.float16,
        )

        assert mem_fp16['parameters_mb'] < mem_fp32['parameters_mb']
        assert mem_fp16['total_training_mb'] < mem_fp32['total_training_mb']


class TestProfileLayers:
    """Tests for TransformerProfiler.profile_layers."""

    def test_profile_layers(self, sample_transformer_config, sample_batch):
        """Returns list of LayerProfile with length matching n_layers * 2 + 2."""
        cfg = sample_transformer_config
        model = create_sample_transformer(**cfg)
        profiler = TransformerProfiler(model, (sample_batch['batch_size'], sample_batch['seq_len'], cfg['d_model']))

        profiles = profiler.profile_layers(
            batch_size=sample_batch['batch_size'],
            seq_len=sample_batch['seq_len'],
            d_model=cfg['d_model'],
            n_heads=cfg['n_heads'],
            n_layers=cfg['n_layers'],
        )

        assert isinstance(profiles, list)
        assert all(isinstance(p, LayerProfile) for p in profiles)
        # Each layer produces 2 profiles (Attention + FFN), plus Embedding + Output
        expected_count = cfg['n_layers'] * 2 + 2
        assert len(profiles) == expected_count

    def test_profile_layers_param_ratios(self, sample_transformer_config, sample_batch):
        """param_ratios (param_ratio) should sum to ~100 (percent)."""
        cfg = sample_transformer_config
        model = create_sample_transformer(**cfg)
        profiler = TransformerProfiler(model, (sample_batch['batch_size'], sample_batch['seq_len'], cfg['d_model']))

        profiles = profiler.profile_layers(
            batch_size=sample_batch['batch_size'],
            seq_len=sample_batch['seq_len'],
            d_model=cfg['d_model'],
            n_heads=cfg['n_heads'],
            n_layers=cfg['n_layers'],
        )

        total_ratio = sum(p.param_ratio for p in profiles)
        assert abs(total_ratio - 100.0) < 1.0, f"param_ratios sum to {total_ratio}, expected ~100.0"


class TestAttentionComplexityComparison:
    """Tests for TransformerProfiler.get_attention_complexity_comparison."""

    def test_attention_complexity_comparison(self, sample_transformer_config):
        """Transformer FLOPs grow as L^2, Mamba FLOPs grow as L."""
        cfg = sample_transformer_config
        model = create_sample_transformer(**cfg)
        profiler = TransformerProfiler(model, (2, 16, cfg['d_model']))

        seq_lengths = [64, 128, 256, 512]
        result = profiler.get_attention_complexity_comparison(seq_lengths, d_model=cfg['d_model'])

        transformer_flops = result['transformer_flops']
        mamba_flops = result['mamba_flops']

        # Transformer flops should quadruple when seq_len doubles (L^2 growth)
        # Compare flops at L=128 vs L=64
        ratio_transformer = transformer_flops[1] / transformer_flops[0]
        assert ratio_transformer > 3.0, f"Transformer FLOPs ratio: {ratio_transformer}, expected ~4.0 (L^2)"

        # Mamba flops should double when seq_len doubles (linear growth)
        ratio_mamba = mamba_flops[1] / mamba_flops[0]
        assert abs(ratio_mamba - 2.0) < 0.5, f"Mamba FLOPs ratio: {ratio_mamba}, expected ~2.0 (linear)"


class TestSimulateTrainingStep:
    """Tests for TransformerProfiler.simulate_training_step."""

    def test_simulate_training_step(self, sample_transformer_config, sample_batch):
        """Returns dict with forward_ms, backward_ms, total_ms, all > 0."""
        cfg = sample_transformer_config
        model = create_sample_transformer(**cfg)
        profiler = TransformerProfiler(model, (sample_batch['batch_size'], sample_batch['seq_len'], cfg['d_model']))

        result = profiler.simulate_training_step(
            batch_size=sample_batch['batch_size'],
            seq_len=sample_batch['seq_len'],
            d_model=cfg['d_model'],
            n_layers=cfg['n_layers'],
        )

        for key in ['forward_ms', 'backward_ms', 'total_ms']:
            assert key in result, f"Missing key: {key}"
            assert result[key] > 0, f"Expected {key} > 0, got {result[key]}"

    def test_simulate_training_step_backward_larger(self, sample_transformer_config, sample_batch):
        """backward_ms should be >= forward_ms (backward is typically ~2x forward)."""
        cfg = sample_transformer_config
        model = create_sample_transformer(**cfg)
        profiler = TransformerProfiler(model, (sample_batch['batch_size'], sample_batch['seq_len'], cfg['d_model']))

        result = profiler.simulate_training_step(
            batch_size=sample_batch['batch_size'],
            seq_len=sample_batch['seq_len'],
            d_model=cfg['d_model'],
            n_layers=cfg['n_layers'],
        )

        assert result['backward_ms'] >= result['forward_ms'], \
            f"backward_ms ({result['backward_ms']}) should be >= forward_ms ({result['forward_ms']})"
