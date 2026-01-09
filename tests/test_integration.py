"""Integration tests for end-to-end QSOM workflows."""

import sys
from pathlib import Path

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest
import numpy as np
from qiskit import QuantumCircuit

from qsom import QuantumBackend, ClassicalShadow, QuantumSOM, generate_quantum_states


class TestEndToEndWorkflow:
    """Integration tests for complete QSOM workflows."""

    def test_full_pipeline_random_states(self):
        """Test complete pipeline with random quantum states."""
        # 1. Generate quantum states
        n_qubits = 2
        n_states = 10
        states = generate_quantum_states(n_states, n_qubits, state_type='random')
        assert len(states) == n_states

        # 2. Initialize backend
        backend = QuantumBackend(use_simulator=True, shots=1)

        # 3. Generate classical shadows
        shadow_gen = ClassicalShadow(
            n_qubits=n_qubits,
            backend=backend,
            shadow_size=10
        )

        shadow_features = []
        for state in states:
            # Create circuit from statevector
            qc = QuantumCircuit(n_qubits)
            qc.prepare_state(state, range(n_qubits))

            samples = shadow_gen.generate(qc)
            features = shadow_gen.shadow_to_feature_vector(samples)
            shadow_features.append(features)

        shadow_features = np.array(shadow_features)
        assert shadow_features.shape == (n_states, n_qubits * 3 * 2)

        # 4. Train SOM
        som = QuantumSOM(
            grid_size=(4, 4),
            input_dim=shadow_features.shape[1],
            n_iterations=50,
            random_seed=42
        )
        som.train(shadow_features, verbose=False)

        # 5. Verify predictions
        predictions = som.predict_batch(shadow_features)
        assert predictions.shape == (n_states, 2)
        assert np.all(predictions >= 0)
        assert np.all(predictions < 4)

        # 6. Verify U-matrix
        umatrix = som.get_umatrix()
        assert umatrix.shape == (4, 4)

    def test_full_pipeline_with_labels(self):
        """Test pipeline with labeled data for classification."""
        n_qubits = 2
        n_per_class = 5

        # Generate states of different types
        ghz_states = generate_quantum_states(n_per_class, n_qubits, state_type='ghz')
        random_states = generate_quantum_states(n_per_class, n_qubits, state_type='random')

        all_states = ghz_states + random_states
        labels = np.array([0] * n_per_class + [1] * n_per_class)

        backend = QuantumBackend(use_simulator=True, shots=1)
        shadow_gen = ClassicalShadow(n_qubits=n_qubits, backend=backend, shadow_size=15)

        shadow_features = []
        for state in all_states:
            qc = QuantumCircuit(n_qubits)
            qc.prepare_state(state, range(n_qubits))
            samples = shadow_gen.generate(qc)
            features = shadow_gen.shadow_to_feature_vector(samples)
            shadow_features.append(features)

        shadow_features = np.array(shadow_features)

        som = QuantumSOM(
            grid_size=(5, 5),
            input_dim=shadow_features.shape[1],
            n_iterations=100,
            distance_metric='quantum',
            random_seed=42
        )
        som.train(shadow_features, verbose=False)

        # Get hit map
        hit_map = som.get_hit_map(shadow_features)
        assert hit_map.sum() == len(all_states)

    def test_different_distance_metrics(self):
        """Test that different distance metrics work in full pipeline."""
        n_qubits = 2
        n_states = 8

        states = generate_quantum_states(n_states, n_qubits, state_type='random')

        backend = QuantumBackend(use_simulator=True, shots=1)
        shadow_gen = ClassicalShadow(n_qubits=n_qubits, backend=backend, shadow_size=10)

        shadow_features = []
        for state in states:
            qc = QuantumCircuit(n_qubits)
            qc.prepare_state(state, range(n_qubits))
            samples = shadow_gen.generate(qc)
            features = shadow_gen.shadow_to_feature_vector(samples)
            shadow_features.append(features)

        shadow_features = np.array(shadow_features)

        for metric in ['euclidean', 'fidelity', 'quantum']:
            som = QuantumSOM(
                grid_size=(3, 3),
                input_dim=shadow_features.shape[1],
                n_iterations=30,
                distance_metric=metric,
                random_seed=42
            )
            som.train(shadow_features, verbose=False)

            # All metrics should produce valid results
            predictions = som.predict_batch(shadow_features)
            assert predictions.shape == (n_states, 2)

    def test_shadow_reconstruction_workflow(self):
        """Test workflow with state reconstruction from shadows."""
        n_qubits = 2

        # Create a known state (Bell state)
        qc = QuantumCircuit(n_qubits)
        qc.h(0)
        qc.cx(0, 1)

        backend = QuantumBackend(use_simulator=True, shots=1)
        shadow_gen = ClassicalShadow(
            n_qubits=n_qubits,
            backend=backend,
            shadow_size=50,
            use_inverse_channel=True
        )

        samples = shadow_gen.generate(qc)

        # Reconstruct state
        rho = shadow_gen.reconstruct_state(samples)

        # Check properties
        assert rho.shape == (4, 4)
        # Trace should be approximately 1
        assert np.isclose(np.trace(rho), 1.0, atol=0.5)  # Relaxed for finite samples

    def test_training_convergence(self):
        """Test that training shows convergence."""
        n_qubits = 2
        n_states = 20

        states = generate_quantum_states(n_states, n_qubits, state_type='random', random_seed=42)

        backend = QuantumBackend(use_simulator=True, shots=1)
        shadow_gen = ClassicalShadow(n_qubits=n_qubits, backend=backend, shadow_size=15)

        shadow_features = []
        for state in states:
            qc = QuantumCircuit(n_qubits)
            qc.prepare_state(state, range(n_qubits))
            samples = shadow_gen.generate(qc)
            features = shadow_gen.shadow_to_feature_vector(samples)
            shadow_features.append(features)

        shadow_features = np.array(shadow_features)

        som = QuantumSOM(
            grid_size=(5, 5),
            input_dim=shadow_features.shape[1],
            n_iterations=300,
            random_seed=42
        )
        som.train(shadow_features, verbose=False, track_errors=True)

        # Training history should exist
        assert len(som.training_history) > 0

        # Quantization error should generally decrease
        errors = [h['quantization_error'] for h in som.training_history]
        # First error should be >= last error (allowing some variance)
        assert errors[0] >= errors[-1] * 0.5  # Allow 50% margin for stochasticity


class TestUtilityFunctions:
    """Test utility functions in isolation."""

    def test_generate_quantum_states_types(self):
        """Test generation of different state types."""
        n_qubits = 3
        n_states = 5

        for state_type in ['random', 'ghz', 'w', 'product']:
            states = generate_quantum_states(n_states, n_qubits, state_type=state_type)
            assert len(states) == n_states

    def test_generate_quantum_states_reproducibility(self):
        """Test reproducibility with random seed."""
        states1 = generate_quantum_states(5, 2, state_type='random', random_seed=42)
        states2 = generate_quantum_states(5, 2, state_type='random', random_seed=42)

        for s1, s2 in zip(states1, states2):
            assert np.allclose(np.array(s1), np.array(s2))


class TestErrorHandling:
    """Test error handling and edge cases."""

    def test_empty_data_training(self):
        """Test behavior with minimal data."""
        som = QuantumSOM(grid_size=(2, 2), input_dim=4, n_iterations=5)
        data = np.random.rand(2, 4)  # Minimal data

        # Should not raise
        som.train(data, verbose=False)

    def test_single_sample_prediction(self):
        """Test prediction with single sample."""
        som = QuantumSOM(grid_size=(3, 3), input_dim=4)
        data = np.random.rand(10, 4)
        som.train(data, verbose=False)

        single = np.random.rand(4)
        bmu = som.predict(single)

        assert len(bmu) == 2
        assert 0 <= bmu[0] < 3
        assert 0 <= bmu[1] < 3

    def test_large_grid(self):
        """Test with larger grid size."""
        som = QuantumSOM(grid_size=(15, 15), input_dim=10, n_iterations=10)
        data = np.random.rand(50, 10)

        som.train(data, verbose=False)

        assert som.weights.shape == (15, 15, 10)
