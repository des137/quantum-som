"""Pytest configuration and shared fixtures."""

import sys
from pathlib import Path

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest
import numpy as np

from qsom import QuantumBackend


@pytest.fixture
def backend():
    """Create a quantum backend for testing."""
    return QuantumBackend(use_simulator=True, shots=1)


@pytest.fixture
def random_seed():
    """Set random seed for reproducible tests."""
    np.random.seed(42)
    return 42


@pytest.fixture
def sample_data():
    """Generate sample data for SOM tests."""
    np.random.seed(42)
    return np.random.rand(20, 12)


def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "hardware: marks tests requiring quantum hardware"
    )
