import sys
import os

import pytest
import torch
import torch.nn as nn
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from utils.training_monitor import TrainingMonitor, TrainingMetrics

torch.manual_seed(42)


@pytest.fixture
def model():
    """Create a small model for training monitor tests."""
    torch.manual_seed(42)
    return nn.Sequential(
        nn.Linear(32, 64),
        nn.ReLU(),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, 10),
    )


@pytest.fixture
def monitor(model):
    """Create a TrainingMonitor for testing."""
    return TrainingMonitor(model, window_size=50)


# -------------------------------------------------------
# 1. Initialization
# -------------------------------------------------------

class TestTrainingMonitorInit:
    def test_training_monitor_init(self, monitor):
        """TrainingMonitor should be created without errors."""
        assert monitor.model is not None
        assert monitor.window_size == 50


# -------------------------------------------------------
# 2. Hook management
# -------------------------------------------------------

class TestHooks:
    def test_register_remove_hooks(self, monitor):
        """Registering and then removing hooks should not raise."""
        monitor.register_hooks()
        monitor.remove_hooks()
        assert len(monitor.hooks) == 0, "Hooks should be empty after removal"


# -------------------------------------------------------
# 3. Gradient and parameter norms
# -------------------------------------------------------

class TestNorms:
    def test_compute_grad_norm(self, monitor):
        """compute_grad_norm should return a float >= 0 (0 if no gradients yet)."""
        grad_norm = monitor.compute_grad_norm()
        assert isinstance(grad_norm, float), f"Should return float, got {type(grad_norm)}"
        assert grad_norm >= 0, f"Grad norm should be >= 0, got {grad_norm}"

    def test_compute_param_norm(self, monitor):
        """compute_param_norm should return a float > 0 for a non-empty model."""
        param_norm = monitor.compute_param_norm()
        assert isinstance(param_norm, float), f"Should return float, got {type(param_norm)}"
        assert param_norm > 0, f"Param norm should be > 0, got {param_norm}"


# -------------------------------------------------------
# 4. Step recording
# -------------------------------------------------------

class TestStep:
    def test_step_returns_metrics(self, monitor):
        """step() should return a TrainingMetrics with step, loss, grad_norm."""
        metrics = monitor.step(step=0, epoch=0, loss=1.5, learning_rate=0.001, batch_size=4)
        assert isinstance(metrics, TrainingMetrics), f"Should return TrainingMetrics, got {type(metrics)}"
        assert metrics.step == 0, f"Expected step=0, got {metrics.step}"
        assert metrics.loss == 1.5, f"Expected loss=1.5, got {metrics.loss}"
        assert isinstance(metrics.grad_norm, float), "grad_norm should be a float"

    def test_step_accumulates(self, monitor):
        """Calling step multiple times should accumulate history."""
        for i in range(5):
            monitor.step(step=i, epoch=0, loss=float(10 - i), learning_rate=0.001, batch_size=4)
        assert len(monitor.metrics_history) == 5, (
            f"Expected 5 entries in history, got {len(monitor.metrics_history)}"
        )


# -------------------------------------------------------
# 5. Training summary
# -------------------------------------------------------

class TestTrainingSummary:
    def test_get_training_summary(self, monitor):
        """get_training_summary should return a dict with expected keys."""
        for i in range(5):
            monitor.step(step=i, epoch=0, loss=float(10 - i), learning_rate=0.001, batch_size=4)
        summary = monitor.get_training_summary()
        assert isinstance(summary, dict), "Should return a dict"
        assert 'total_steps' in summary, "Should contain 'total_steps'"
        assert 'avg_loss' in summary, "Should contain 'avg_loss'"
        assert 'avg_grad_norm' in summary, "Should contain 'avg_grad_norm'"


# -------------------------------------------------------
# 6. Anomaly detection
# -------------------------------------------------------

class TestAnomalyDetection:
    def test_detect_anomalies(self, monitor):
        """detect_anomalies should return a dict."""
        for i in range(15):
            monitor.step(step=i, epoch=0, loss=float(10 - i * 0.5), learning_rate=0.001, batch_size=4)
        anomalies = monitor.detect_anomalies()
        assert isinstance(anomalies, dict), "Should return a dict"
