import sys
import os

import pytest
import torch
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from utils.token_tracker import (
    create_simple_vocab,
    TrackedTransformer,
    TokenTracker,
    TokenJourney,
)

torch.manual_seed(42)


@pytest.fixture
def vocab():
    """Create a small vocabulary for testing."""
    return create_simple_vocab(vocab_size=100)


@pytest.fixture
def model(vocab):
    """Create a small TrackedTransformer for testing."""
    torch.manual_seed(42)
    return TrackedTransformer(
        vocab_size=len(vocab),
        d_model=64,
        n_heads=4,
        n_layers=4,
        max_seq_len=32,
    )


@pytest.fixture
def tracker(model, vocab):
    """Create a TokenTracker for testing."""
    return TokenTracker(model, vocab)


# -------------------------------------------------------
# 1. Vocab creation
# -------------------------------------------------------

class TestCreateSimpleVocab:
    def test_create_simple_vocab(self):
        """create_simple_vocab should return a dict of the requested size."""
        vocab_size = 200
        vocab = create_simple_vocab(vocab_size=vocab_size)
        assert isinstance(vocab, dict), "Vocab should be a dict"
        assert len(vocab) == vocab_size, f"Expected {vocab_size} entries, got {len(vocab)}"


# -------------------------------------------------------
# 2. TrackedTransformer tests
# -------------------------------------------------------

class TestTrackedTransformer:
    def test_tracked_transformer_forward(self, model, vocab):
        """Forward pass should produce logits of shape (B, S, V)."""
        model.eval()
        batch_size, seq_len = 2, 8
        input_ids = torch.randint(0, len(vocab), (batch_size, seq_len))
        with torch.no_grad():
            logits = model(input_ids)
        assert logits.shape == (batch_size, seq_len, len(vocab)), (
            f"Expected shape ({batch_size}, {seq_len}, {len(vocab)}), got {logits.shape}"
        )

    def test_tracked_transformer_enable_tracking(self, model):
        """enable_tracking_for_all_layers should not raise."""
        model.enable_tracking_for_all_layers(position=0)
        model.disable_tracking_for_all_layers()


# -------------------------------------------------------
# 3. TokenTracker tests
# -------------------------------------------------------

class TestTokenTracker:
    def test_token_tracker_init(self, tracker):
        """TokenTracker should be created without errors."""
        assert tracker.model is not None
        assert tracker.vocab is not None

    def test_track_token_journey(self, tracker):
        """track_token_journey should return a TokenJourney with layer_states."""
        # Use tokens that exist in the vocab (indices 0..3 are special, 4+ are words)
        input_text = "the a is"
        journey = tracker.track_token_journey(input_text, token_position=0, return_top_k=5)
        assert isinstance(journey, TokenJourney), "Should return a TokenJourney"
        assert len(journey.layer_states) > 0, "layer_states should not be empty"

    def test_track_token_journey_layer_count(self, tracker, model):
        """Number of layer_states should equal n_layers."""
        input_text = "the a is"
        journey = tracker.track_token_journey(input_text, token_position=0, return_top_k=5)
        assert len(journey.layer_states) == model.n_layers, (
            f"Expected {model.n_layers} layer_states, got {len(journey.layer_states)}"
        )

    def test_track_token_journey_has_logits(self, tracker, vocab):
        """Logits should not be None and have shape (1, vocab_size)."""
        input_text = "the a is"
        journey = tracker.track_token_journey(input_text, token_position=0, return_top_k=5)
        assert journey.logits is not None, "logits should not be None"
        assert journey.logits.shape == (len(vocab),), (
            f"Expected logits shape ({len(vocab)},), got {journey.logits.shape}"
        )

    def test_compare_residual_effects(self, tracker):
        """compare_residual_effects should return a dict with residual_norms key."""
        input_text = "the a is"
        journey = tracker.track_token_journey(input_text, token_position=0, return_top_k=5)
        result = tracker.compare_residual_effects(journey)
        assert isinstance(result, dict), "Should return a dict"
        assert 'layer_residual_norms' in result, "Result should contain 'layer_residual_norms' key"
