"""
Tests for utils.initialization_comparator module.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn

torch.manual_seed(42)

from utils.initialization_comparator import InitializationComparator


class TestInitializationComparator:
    """Tests for InitializationComparator class."""

    def test_init(self):
        """Can be created with a layer_sizes list."""
        layer_sizes = [64, 32, 16, 8]
        comparator = InitializationComparator(layer_sizes)
        assert comparator.layer_sizes == layer_sizes
        assert len(comparator.init_methods) == 8

    def test_create_sample_network(self):
        """Returns nn.Module and forward pass works."""
        layer_sizes = [32, 16, 8]
        comparator = InitializationComparator(layer_sizes)
        net = comparator.create_sample_network()
        assert isinstance(net, nn.Module)
        # Forward pass should work
        x = torch.randn(4, layer_sizes[0])
        output = net(x)
        assert output.shape == (4, layer_sizes[-1])

    def test_analyze_initialization(self):
        """Returns dict with expected keys: activations, gradient_stats, and method."""
        layer_sizes = [32, 16, 8]
        comparator = InitializationComparator(layer_sizes)
        result = comparator.analyze_initialization('xavier_normal', n_samples=100)
        assert isinstance(result, dict)
        assert 'activations' in result
        assert 'gradient_stats' in result
        assert 'method' in result
        assert result['method'] == 'xavier_normal'
        # Activations should contain per-layer stats
        assert 'layer_0_weight' in result['activations']
        assert 'layer_0_output' in result['activations']

    def test_analyze_initialization_all_methods(self):
        """Works for all 8 built-in initialization methods."""
        layer_sizes = [32, 16, 8]
        comparator = InitializationComparator(layer_sizes)
        methods = [
            'xavier_uniform', 'xavier_normal',
            'kaiming_uniform', 'kaiming_normal',
            'orthogonal', 'lecun_normal',
            'random_normal', 'random_uniform',
        ]
        for method in methods:
            result = comparator.analyze_initialization(method, n_samples=50)
            assert isinstance(result, dict)
            assert 'activations' in result
            assert 'gradient_stats' in result

    def test_compare_all_initializations(self):
        """Returns dict with all method names as keys."""
        layer_sizes = [32, 16, 8]
        comparator = InitializationComparator(layer_sizes)
        results = comparator.compare_all_initializations()
        assert isinstance(results, dict)
        for method in comparator.init_methods:
            assert method in results
            assert 'activations' in results[method]

    def test_create_initialization_report(self):
        """Returns non-empty string."""
        layer_sizes = [32, 16, 8]
        comparator = InitializationComparator(layer_sizes)
        report = comparator.create_initialization_report()
        assert isinstance(report, str)
        assert len(report) > 0
