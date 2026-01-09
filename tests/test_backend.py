"""Tests for the QuantumBackend module."""

import sys
from pathlib import Path

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest
from qiskit import QuantumCircuit
from qsom.backend import QuantumBackend


class TestQuantumBackend:
    """Tests for QuantumBackend class."""

    def test_backend_init_simulator(self):
        """Test backend initialization with simulator."""
        backend = QuantumBackend(use_simulator=True)
        assert backend.backend is not None
        assert backend.use_simulator is True

    def test_backend_run_simple_circuit(self):
        """Test running a simple circuit."""
        backend = QuantumBackend(use_simulator=True, shots=10)
        qc = QuantumCircuit(1)
        qc.h(0)
        qc.measure_all()

        result = backend.run(qc)
        assert result is not None

    def test_backend_run_multiple_circuits(self):
        """Test running multiple circuits in batch."""
        backend = QuantumBackend(use_simulator=True, shots=10)

        circuits = []
        for _ in range(3):
            qc = QuantumCircuit(2)
            qc.h(0)
            qc.cx(0, 1)
            qc.measure_all()
            circuits.append(qc)

        result = backend.run(circuits)
        assert result is not None

    def test_backend_shots_configuration(self):
        """Test that shots are configured correctly."""
        shots = 100
        backend = QuantumBackend(use_simulator=True, shots=shots)
        assert backend.shots == shots

    def test_backend_default_fallback(self):
        """Test that backend falls back to simulator when no service provided."""
        backend = QuantumBackend()  # No service, should use simulator
        assert backend.backend is not None
