"""
Tests for utils.gradient_flow_visualizer module.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn
import numpy as np

torch.manual_seed(42)

from utils.gradient_flow_visualizer import GradientFlowVisualizer


class TestGradientFlowVisualizer:
    """Tests for GradientFlowVisualizer class."""

    def test_init(self):
        """Can be created without errors."""
        visualizer = GradientFlowVisualizer()
        assert visualizer is not None

    def test_create_sample_networks(self):
        """Returns dict with expected network names."""
        visualizer = GradientFlowVisualizer()
        networks = visualizer.create_sample_networks()
        assert isinstance(networks, dict)
        expected_names = ['deep_relu', 'deep_tanh', 'deep_sigmoid', 'residual', 'lstm']
        for name in expected_names:
            assert name in networks, f"Missing network: {name}"
            assert isinstance(networks[name], nn.Module)

    def test_analyze_gradient_flow(self):
        """Returns dict with gradient_stats containing layer_grad_norms."""
        visualizer = GradientFlowVisualizer()
        networks = visualizer.create_sample_networks()
        network = networks['deep_relu']
        result = visualizer.analyze_gradient_flow(network, input_size=(8, 512), num_batches=2)
        assert isinstance(result, dict)
        assert 'gradient_stats' in result
        # Each gradient stat entry should have avg_norm
        for name, stats in result['gradient_stats'].items():
            assert 'avg_norm' in stats
            assert isinstance(stats['avg_norm'], float)

    def test_analyze_gradient_flow_residual_better(self):
        """Residual network should have healthier gradient flow than deep_relu."""
        torch.manual_seed(42)
        visualizer = GradientFlowVisualizer()
        networks = visualizer.create_sample_networks()

        relu_result = visualizer.analyze_gradient_flow(
            networks['deep_relu'], input_size=(8, 512), num_batches=3
        )
        residual_result = visualizer.analyze_gradient_flow(
            networks['residual'], input_size=(8, 512), num_batches=3
        )

        # Collect weight gradient norms for each network
        relu_norms = [
            stats['avg_norm']
            for name, stats in relu_result['gradient_stats'].items()
            if 'weight' in name
        ]
        residual_norms = [
            stats['avg_norm']
            for name, stats in residual_result['gradient_stats'].items()
            if 'weight' in name
        ]

        assert len(relu_norms) > 1
        assert len(residual_norms) > 1

        # Measure gradient flow health: ratio of minimum to maximum gradient norm.
        # A higher ratio means more uniform (healthier) gradient flow.
        relu_ratio = min(relu_norms) / (max(relu_norms) + 1e-8)
        residual_ratio = min(residual_norms) / (max(residual_norms) + 1e-8)

        # Residual network should maintain a better min/max ratio (more uniform)
        assert residual_ratio >= relu_ratio, (
            f"Residual min/max ratio ({residual_ratio:.4f}) should be >= "
            f"ReLU min/max ratio ({relu_ratio:.4f})"
        )

    def test_compare_activation_functions(self):
        """Returns plotly Figure."""
        visualizer = GradientFlowVisualizer()
        fig = visualizer.compare_activation_functions()
        # plotly Figure has a _data attribute with traces
        assert hasattr(fig, 'data')
        assert len(fig.data) > 0

    def test_visualize_residual_connections(self):
        """Returns plotly Figure."""
        visualizer = GradientFlowVisualizer()
        fig = visualizer.visualize_residual_connections()
        assert hasattr(fig, 'data')
        assert len(fig.data) > 0

    def test_create_gradient_flow_report(self):
        """Returns non-empty string."""
        visualizer = GradientFlowVisualizer()
        report = visualizer.create_gradient_flow_report()
        assert isinstance(report, str)
        assert len(report) > 0
