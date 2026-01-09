"""Tests for the ClassicalShadow module."""

import sys
from pathlib import Path

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest
import numpy as np
from qiskit import QuantumCircuit
from qsom.backend import QuantumBackend
from qsom.shadows import ClassicalShadow


@pytest.fixture
def mock_backend():
    """Create a mock backend for testing."""
    return QuantumBackend(use_simulator=True, shots=1)


class TestClassicalShadow:
    """Tests for ClassicalShadow class."""

    def test_shadow_init(self, mock_backend):
        """Test shadow initialization."""
        shadow = ClassicalShadow(n_qubits=2, backend=mock_backend)
        assert shadow.n_qubits == 2
        assert shadow.backend == mock_backend

    def test_shadow_init_default_backend(self):
        """Test shadow initialization with default backend."""
        shadow = ClassicalShadow(n_qubits=2)
        assert shadow.n_qubits == 2
        assert shadow.backend is not None

    def test_random_bases_shape(self, mock_backend):
        """Test random bases generation shape."""
        shadow = ClassicalShadow(n_qubits=2, backend=mock_backend)
        bases = shadow._get_random_bases(10)
        assert bases.shape == (10, 2)

    def test_random_bases_values(self, mock_backend):
        """Test random bases are in valid range."""
        shadow = ClassicalShadow(n_qubits=3, backend=mock_backend)
        bases = shadow._get_random_bases(100)
        assert np.all(bases >= 0) and np.all(bases <= 2)

    def test_generate_shadow_dimensions(self, mock_backend):
        """Test shadow generation output dimensions."""
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)

        shadow_gen = ClassicalShadow(n_qubits=2, backend=mock_backend, shadow_size=5)
        samples = shadow_gen.generate(qc)

        assert len(samples) == 5
        for bases, outcomes in samples:
            assert len(bases) == 2
            assert len(outcomes) == 2

    def test_feature_vector_shape(self, mock_backend):
        """Test feature vector shape."""
        qc = QuantumCircuit(2)
        shadow_gen = ClassicalShadow(n_qubits=2, backend=mock_backend, shadow_size=5)
        samples = shadow_gen.generate(qc)
        vec = shadow_gen.shadow_to_feature_vector(samples)

        # 2 qubits * 3 bases * 2 outcomes = 12 features
        assert vec.shape == (12,)

    def test_feature_vector_normalized(self, mock_backend):
        """Test feature vector is properly normalized."""
        qc = QuantumCircuit(2)
        qc.h(0)
        shadow_gen = ClassicalShadow(n_qubits=2, backend=mock_backend, shadow_size=20)
        samples = shadow_gen.generate(qc)
        vec = shadow_gen.shadow_to_feature_vector(samples)

        # Sum should equal n_qubits (1.0 per qubit normalized across bases/outcomes)
        # Each qubit contributes 3 bases * 2 outcomes = 6 features, summing to 1.0
        assert np.isclose(vec.sum(), 2.0, atol=1e-6)  # 2 qubits

    def test_bases_to_string(self, mock_backend):
        """Test basis array to string conversion."""
        shadow = ClassicalShadow(n_qubits=3, backend=mock_backend)
        bases = np.array([0, 1, 2])  # X, Y, Z
        result = shadow._bases_to_string(bases)
        assert result == "XYZ"

    def test_inverse_channel_single_qubit(self, mock_backend):
        """Test single qubit inverse channel."""
        shadow = ClassicalShadow(n_qubits=1, backend=mock_backend)

        # Test Z basis, outcome 0 (|0>)
        result = shadow._inverse_channel_single_qubit('Z', 0)
        expected = 3 * np.array([[1, 0], [0, 0]]) - np.eye(2)
        assert result.shape == (2, 2)
        assert np.allclose(result, expected)

    def test_reconstruct_state_shape(self, mock_backend):
        """Test state reconstruction output shape."""
        qc = QuantumCircuit(2)
        qc.h(0)

        shadow_gen = ClassicalShadow(n_qubits=2, backend=mock_backend, shadow_size=10)
        samples = shadow_gen.generate(qc)
        rho = shadow_gen.reconstruct_state(samples)

        assert rho.shape == (4, 4)  # 2^2 x 2^2
