"""
Shared fixtures for model profiling tests.
"""
import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


@pytest.fixture
def sample_transformer_config():
    """Common transformer configuration for testing."""
    return {
        'd_model': 128,
        'n_heads': 4,
        'n_layers': 4,
        'vocab_size': 1000,
    }


@pytest.fixture
def sample_batch():
    """Common batch configuration for testing."""
    return {
        'batch_size': 2,
        'seq_len': 16,
    }
