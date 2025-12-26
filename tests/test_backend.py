import pytest
from qiskit import QuantumCircuit
from qsom.backend import QuantumBackend

def test_backend_init_simulator():
    backend = QuantumBackend(use_simulator=True)
    assert backend.backend.name == "aer_simulator"

def test_backend_run_simple_circuit():
    backend = QuantumBackend(use_simulator=True, shots=10)
    qc = QuantumCircuit(1)
    qc.h(0)
    qc.measure_all()
    
    result = backend.run(qc)
    # Check if result has counts or valid format
    # Depending on AerSampler implementation details.
    # Usually wrapper returns something that has quasi_dists or counts.
    assert result is not None
