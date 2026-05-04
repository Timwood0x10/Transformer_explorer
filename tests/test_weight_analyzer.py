import sys
import os

import pytest
import torch
import torch.nn as nn
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from utils.weight_analyzer import WeightAnalyzer, WeightStats

torch.manual_seed(42)


@pytest.fixture
def model():
    """Create a small model with uniform layer sizes for weight analysis tests.

    All Linear layers have the same shape so that analyze_weight_correlation
    can compute np.corrcoef on equal-length flattened weight vectors.
    """
    torch.manual_seed(42)
    return nn.Sequential(
        nn.Linear(32, 32),
        nn.ReLU(),
        nn.Linear(32, 32),
        nn.ReLU(),
        nn.Linear(32, 32),
    )


@pytest.fixture
def analyzer(model):
    """Create a WeightAnalyzer for testing."""
    return WeightAnalyzer(model)


# -------------------------------------------------------
# 1. Weight distribution analysis
# -------------------------------------------------------

class TestAnalyzeWeightDistribution:
    def test_analyze_weight_distribution(self, analyzer):
        """Should return a list; each item should have layer_name, mean, std."""
        stats = analyzer.analyze_weight_distribution()
        assert isinstance(stats, list), "Should return a list"
        assert len(stats) > 0, "List should not be empty"
        for s in stats:
            assert isinstance(s, WeightStats), f"Each item should be WeightStats, got {type(s)}"
            assert hasattr(s, 'layer_name'), "Should have layer_name"
            assert hasattr(s, 'mean'), "Should have mean"
            assert hasattr(s, 'std'), "Should have std"

    def test_analyze_weight_distribution_stats_valid(self, analyzer):
        """Statistical invariants: std >= 0, min <= mean <= max."""
        stats = analyzer.analyze_weight_distribution()
        for s in stats:
            assert s.std >= 0, f"std {s.std} for {s.layer_name} is negative"
            assert s.min <= s.mean <= s.max, (
                f"For {s.layer_name}: min={s.min}, mean={s.mean}, max={s.max}"
            )


# -------------------------------------------------------
# 2. Anomaly detection
# -------------------------------------------------------

class TestDetectWeightAnomalies:
    def test_detect_weight_anomalies(self, analyzer):
        """Should return a dict with string keys (layer names)."""
        anomalies = analyzer.detect_weight_anomalies()
        assert isinstance(anomalies, dict), "Should return a dict"
        for key in anomalies:
            assert isinstance(key, str), f"Key {key} should be a string"


# -------------------------------------------------------
# 3. Weight correlation
# -------------------------------------------------------

class TestAnalyzeWeightCorrelation:
    def test_analyze_weight_correlation(self, analyzer):
        """Should return a nested dict (layer_name -> layer_name -> float)."""
        correlations = analyzer.analyze_weight_correlation()
        assert isinstance(correlations, dict), "Should return a dict"
        for outer_key, inner_dict in correlations.items():
            assert isinstance(outer_key, str), f"Outer key {outer_key} should be a string"
            assert isinstance(inner_dict, dict), f"Value for {outer_key} should be a dict"
            for inner_key, val in inner_dict.items():
                assert isinstance(inner_key, str), f"Inner key {inner_key} should be a string"
                assert isinstance(val, (int, float)), f"Correlation value should be numeric, got {type(val)}"


# -------------------------------------------------------
# 4. Weight report
# -------------------------------------------------------

class TestGenerateWeightReport:
    def test_generate_weight_report(self, analyzer):
        """Should return a non-empty string."""
        report = analyzer.generate_weight_report()
        assert isinstance(report, str), "Report should be a string"
        assert len(report) > 0, "Report should not be empty"


# -------------------------------------------------------
# 5. Weight evolution recording
# -------------------------------------------------------

class TestRecordWeightEvolution:
    def test_record_weight_evolution(self, analyzer):
        """Recording weight evolution at step 0 and step 1 should not raise."""
        analyzer.record_weight_evolution(step=0)
        analyzer.record_weight_evolution(step=1)
        # Verify history was recorded
        total_entries = sum(len(v) for v in analyzer.weight_history.values())
        assert total_entries > 0, "Weight history should have entries after recording"
