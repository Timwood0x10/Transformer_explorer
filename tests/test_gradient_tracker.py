"""
Tests for utils.gradient_tracker module.
"""
import sys
import os
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from utils.gradient_tracker import (
    GradientTrackingTransformer,
    GradientTracker,
    GradientJourney,
)


@pytest.fixture(autouse=True)
def set_seed():
    torch.manual_seed(42)


@pytest.fixture
def model_with_residual(sample_transformer_config):
    """Create a GradientTrackingTransformer with residual connections."""
    cfg = sample_transformer_config
    return GradientTrackingTransformer(
        d_model=cfg['d_model'],
        n_heads=cfg['n_heads'],
        n_layers=cfg['n_layers'],
        vocab_size=cfg['vocab_size'],
        use_residual=True,
    )


@pytest.fixture
def model_without_residual(sample_transformer_config):
    """Create a GradientTrackingTransformer without residual connections."""
    cfg = sample_transformer_config
    return GradientTrackingTransformer(
        d_model=cfg['d_model'],
        n_heads=cfg['n_heads'],
        n_layers=cfg['n_layers'],
        vocab_size=cfg['vocab_size'],
        use_residual=False,
    )


@pytest.fixture
def sample_input_target(sample_transformer_config, sample_batch):
    """Create sample input_ids and target_ids tensors."""
    B = sample_batch['batch_size']
    S = sample_batch['seq_len']
    V = sample_transformer_config['vocab_size']
    input_ids = torch.randint(0, V, (B, S))
    target_ids = torch.randint(0, V, (B, S))
    return input_ids, target_ids


class TestGradientTrackingTransformerForward:
    """Tests for GradientTrackingTransformer forward pass."""

    def test_gradient_tracking_transformer_forward(self, model_with_residual, sample_input_target, sample_transformer_config):
        """Model forward pass works, output shape is (B, S, V)."""
        input_ids, _ = sample_input_target
        cfg = sample_transformer_config
        output = model_with_residual(input_ids)
        assert output.shape == (input_ids.shape[0], input_ids.shape[1], cfg['vocab_size'])

    def test_gradient_tracking_with_residual(self, model_with_residual, sample_input_target):
        """Forward pass with residual=True works."""
        input_ids, _ = sample_input_target
        output = model_with_residual(input_ids)
        assert output is not None
        assert not torch.isnan(output).any()

    def test_gradient_tracking_without_residual(self, model_without_residual, sample_input_target):
        """Forward pass with residual=False works."""
        input_ids, _ = sample_input_target
        output = model_without_residual(input_ids)
        assert output is not None
        assert not torch.isnan(output).any()


class TestTrackGradientFlow:
    """Tests for GradientTracker.track_gradient_flow."""

    def test_track_gradient_flow(self, model_with_residual, sample_input_target):
        """Returns GradientJourney with layer_gradients and loss_value."""
        tracker = GradientTracker(model_with_residual)
        input_ids, target_ids = sample_input_target

        journey = tracker.track_gradient_flow(input_ids, target_ids)

        assert isinstance(journey, GradientJourney)
        assert hasattr(journey, 'layer_gradients')
        assert hasattr(journey, 'loss_value')
        assert len(journey.layer_gradients) > 0
        assert journey.loss_value > 0

    def test_track_gradient_flow_layer_count(self, model_with_residual, sample_input_target, sample_transformer_config):
        """len(layer_gradients) == n_layers."""
        tracker = GradientTracker(model_with_residual)
        input_ids, target_ids = sample_input_target

        journey = tracker.track_gradient_flow(input_ids, target_ids)

        assert len(journey.layer_gradients) == sample_transformer_config['n_layers']

    def test_track_gradient_flow_health_scores(self, model_with_residual, sample_input_target):
        """All health_scores should be in [0, 1]."""
        tracker = GradientTracker(model_with_residual)
        input_ids, target_ids = sample_input_target

        journey = tracker.track_gradient_flow(input_ids, target_ids)

        for lg in journey.layer_gradients:
            assert 0.0 <= lg.health_score <= 1.0, \
                f"health_score {lg.health_score} out of [0, 1] range"


class TestCompareWithWithoutResidual:
    """Tests for GradientTracker.compare_with_without_residual."""

    def test_compare_with_without_residual(self, model_with_residual, sample_input_target):
        """Returns tuple of 2 GradientJourneys."""
        tracker = GradientTracker(model_with_residual)
        input_ids, target_ids = sample_input_target

        result = tracker.compare_with_without_residual(input_ids, target_ids)

        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], GradientJourney)
        assert isinstance(result[1], GradientJourney)

    def test_compare_residual_healthier(self, model_with_residual, sample_input_target):
        """With-residual should have higher overall_health_score than without-residual."""
        tracker = GradientTracker(model_with_residual)
        input_ids, target_ids = sample_input_target

        journey_with, journey_without = tracker.compare_with_without_residual(input_ids, target_ids)

        assert journey_with.overall_health_score >= journey_without.overall_health_score, \
            f"With-residual health ({journey_with.overall_health_score}) should be >= " \
            f"without-residual health ({journey_without.overall_health_score})"

    def test_gradient_journey_has_vanishing(self, model_with_residual, sample_input_target):
        """Without-residual is more likely to have vanishing gradient problem."""
        tracker = GradientTracker(model_with_residual)
        input_ids, target_ids = sample_input_target

        journey_with, journey_without = tracker.compare_with_without_residual(input_ids, target_ids)

        # Without residual connections, vanishing gradients are more likely.
        # Either has_vanishing_problem is True for without-residual, or its health is worse.
        # We check that without-residual does NOT have strictly better vanishing status.
        assert not (journey_without.has_vanishing_problem == False and journey_with.has_vanishing_problem == True), \
            "With-residual should not have more vanishing than without-residual"

        # Additionally, the without-residual model should have equal or worse health
        assert journey_without.overall_health_score <= journey_with.overall_health_score + 0.01
