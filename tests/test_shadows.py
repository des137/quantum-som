import pytest
import numpy as np
from qiskit import QuantumCircuit
from qsom.backend import QuantumBackend
from qsom.shadows import ClassicalShadow

@pytest.fixture
def mock_backend():
    return QuantumBackend(use_simulator=True, shots=1)

def test_shadow_init(mock_backend):
    shadow = ClassicalShadow(n_qubits=2, backend=mock_backend)
    assert shadow.n_qubits == 2
    assert shadow.backend == mock_backend

def test_random_bases(mock_backend):
    shadow = ClassicalShadow(n_qubits=2, backend=mock_backend)
    bases = shadow._get_random_bases(10)
    assert bases.shape == (10, 2)
    assert np.all(bases >= 0) and np.all(bases <= 2)

def test_generate_shadow_dimensions(mock_backend):
    # This runs actual simulation, might be slow but OK for unit test on small scale
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    
    shadow_gen = ClassicalShadow(n_qubits=2, backend=mock_backend, shadow_size=5)
    samples = shadow_gen.generate(qc)
    
    assert len(samples) == 5
    for bases, outcomes in samples:
        assert len(bases) == 2
        assert len(outcomes) == 2

def test_feature_vector_shape(mock_backend):
    qc = QuantumCircuit(2)
    shadow_gen = ClassicalShadow(n_qubits=2, backend=mock_backend, shadow_size=5)
    samples = shadow_gen.generate(qc)
    vec = shadow_gen.shadow_to_feature_vector(samples)
    
    # 2 qubits * 3 bases * 2 outcomes = 12 features
    assert vec.shape == (12,)
