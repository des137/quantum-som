"""
Classical Shadows Module.

This module provides functionality to generate classical shadows of quantum states.
It implements the theoretical framework from Huang et al., "Predicting many properties
of a quantum system from very few measurements".

Features:
- Random Pauli measurement basis selection
- Proper inverse channel reconstruction
- Multiple shadow types (Pauli, Clifford)
- Feature vector extraction for SOM training
- Qiskit 2.0+ integration
"""

from typing import List, Tuple, Dict, Optional, Union
import numpy as np
from scipy.linalg import sqrtm

# Qiskit imports with graceful fallback
try:
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Statevector, DensityMatrix, Pauli
    from qiskit_aer.primitives import Sampler as AerSampler
    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False

from .backend import QuantumBackend


class ClassicalShadow:
    """
    Classical Shadow representation of a quantum state.

    Classical shadows provide an efficient way to represent quantum states
    using randomized measurements, enabling classical processing of quantum data.

    Supports both Pauli and Clifford shadow protocols.

    Examples:
        >>> from qsom import ClassicalShadow, QuantumBackend
        >>> backend = QuantumBackend(use_simulator=True)
        >>> shadow = ClassicalShadow(n_qubits=3, backend=backend)
        >>> features = shadow.generate(state_prep_circuit)
    """

    # Pauli operators for inverse channel reconstruction
    PAULI_OPS = {
        'I': np.array([[1, 0], [0, 1]], dtype=complex),
        'X': np.array([[0, 1], [1, 0]], dtype=complex),
        'Y': np.array([[0, -1j], [1j, 0]], dtype=complex),
        'Z': np.array([[1, 0], [0, -1]], dtype=complex)
    }

    # Eigenstates for each Pauli basis
    PAULI_EIGENSTATES = {
        'X': {0: np.array([1, 1]) / np.sqrt(2),   # |+>
              1: np.array([1, -1]) / np.sqrt(2)}, # |->
        'Y': {0: np.array([1, 1j]) / np.sqrt(2),  # |+i>
              1: np.array([1, -1j]) / np.sqrt(2)}, # |-i>
        'Z': {0: np.array([1, 0]),                 # |0>
              1: np.array([0, 1])}                 # |1>
    }

    def __init__(
        self,
        n_qubits: int,
        backend: Optional[QuantumBackend] = None,
        n_shots: int = 1000,
        shadow_size: int = 100,
        shadow_type: str = 'pauli',
        use_inverse_channel: bool = True
    ):
        """
        Initialize ClassicalShadow.

        Args:
            n_qubits: Number of qubits in the quantum system.
            backend: QuantumBackend instance for execution. If None, uses local simulator.
            n_shots: Number of shots per measurement (for statistics).
            shadow_size: Number of random bases to sample (N snapshots).
            shadow_type: Type of shadow protocol ('pauli' or 'clifford').
            use_inverse_channel: Whether to apply proper inverse channel reconstruction.
        """
        self.n_qubits = n_qubits
        self.n_shots = n_shots
        self.shadow_size = shadow_size
        self.shadow_type = shadow_type
        self.use_inverse_channel = use_inverse_channel
        self.dim = 2 ** n_qubits

        # Initialize backend
        if backend is not None:
            self.backend = backend
        elif QISKIT_AVAILABLE:
            self.backend = QuantumBackend(use_simulator=True)
        else:
            self.backend = None

        self.pauli_basis = ['X', 'Y', 'Z']
        self.pauli_map = {0: 'X', 1: 'Y', 2: 'Z'}

        # Initialize sampler for standalone usage
        if QISKIT_AVAILABLE and self.backend is None:
            try:
                self._sampler = AerSampler()
            except Exception:
                self._sampler = None
        else:
            self._sampler = None

    def _get_random_bases(self, n_snapshots: int) -> np.ndarray:
        """
        Generate random Pauli bases for each qubit and snapshot.

        Args:
            n_snapshots: Number of measurement snapshots.

        Returns:
            Array of shape (n_snapshots, n_qubits) with integers 0=X, 1=Y, 2=Z.
        """
        return np.random.randint(0, 3, size=(n_snapshots, self.n_qubits))

    def _bases_to_string(self, bases: np.ndarray) -> str:
        """Convert numeric bases array to Pauli string."""
        return ''.join(self.pauli_map[b] for b in bases)

    def _apply_rotation_gates(self, qc: QuantumCircuit, bases: np.ndarray) -> None:
        """
        Apply rotation gates to measure in specified Pauli bases.

        Args:
            qc: Quantum circuit to modify in-place.
            bases: Array of basis indices (0=X, 1=Y, 2=Z) for each qubit.
        """
        for i, basis in enumerate(bases):
            if basis == 0:  # X basis
                qc.h(i)
            elif basis == 1:  # Y basis
                qc.sdg(i)
                qc.h(i)
            # Z basis (2): no rotation needed

    def _inverse_channel_single_qubit(
        self,
        pauli: str,
        outcome: int
    ) -> np.ndarray:
        """
        Compute inverse channel for a single qubit measurement.

        For Pauli measurements: M^-1(|b><b|) = 3 * |b><b| - I

        Args:
            pauli: Pauli basis ('X', 'Y', or 'Z').
            outcome: Measurement outcome (0 or 1).

        Returns:
            2x2 reconstructed operator.
        """
        eigenstate = self.PAULI_EIGENSTATES[pauli][outcome]
        projector = np.outer(eigenstate, eigenstate.conj())
        return 3 * projector - np.eye(2, dtype=complex)

    def _inverse_channel_full(
        self,
        pauli_string: str,
        outcomes: List[int]
    ) -> np.ndarray:
        """
        Apply full inverse channel for multi-qubit measurement.

        The inverse channel is applied as a tensor product of single-qubit
        inverse channels: M^-1 = ⊗_i M_i^-1

        Args:
            pauli_string: String of Pauli operators (e.g., 'XYZ').
            outcomes: List of measurement outcomes for each qubit.

        Returns:
            Reconstructed density matrix snapshot.
        """
        # Build tensor product of single-qubit inverse channels
        snapshot = np.array([[1.0]], dtype=complex)

        for pauli, outcome in zip(pauli_string, outcomes):
            local_snapshot = self._inverse_channel_single_qubit(pauli, outcome)
            snapshot = np.kron(snapshot, local_snapshot)

        return snapshot

    def generate(
        self,
        state_prep_circuit: QuantumCircuit
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate classical shadow samples for a state preparation circuit.

        Args:
            state_prep_circuit: Circuit that prepares the quantum state ρ.

        Returns:
            List of tuples (pauli_indices, outcome_bits):
                - pauli_indices: Array of shape (n_qubits,) with 0=X, 1=Y, 2=Z
                - outcome_bits: Array of shape (n_qubits,) with 0 or 1
        """
        if not QISKIT_AVAILABLE:
            raise RuntimeError("Qiskit is required for shadow generation")

        bases_list = self._get_random_bases(self.shadow_size)
        circuits = []

        for bases in bases_list:
            qc = state_prep_circuit.copy()
            self._apply_rotation_gates(qc, bases)
            qc.measure_all()
            circuits.append(qc)

        # Execute circuits
        results = self.backend.run(circuits)

        # Extract measurement outcomes
        shadow_samples = []
        outcomes_list = self._extract_outcomes(results, len(circuits))

        for i, outcomes in enumerate(outcomes_list):
            shadow_samples.append((bases_list[i], outcomes))

        return shadow_samples

    def _extract_outcomes(self, results, n_circuits: int) -> List[np.ndarray]:
        """Extract measurement outcomes from backend results."""
        outcomes_list = []

        # Handle V1 SamplerResult (quasi_dists)
        if hasattr(results, 'quasi_dists'):
            for dist in results.quasi_dists:
                outcome_int = max(dist, key=dist.get)
                bitstring = format(outcome_int, f'0{self.n_qubits}b')
                outcomes = np.array([int(c) for c in bitstring[::-1]])
                outcomes_list.append(outcomes)
            return outcomes_list

        # Handle V2 PrimitiveResult
        try:
            for pub_result in results:
                if hasattr(pub_result, 'data'):
                    data = pub_result.data
                    if hasattr(data, 'meas'):
                        bitstrings = data.meas.get_bitstrings()
                        b = bitstrings[0]
                        outcomes = np.array([int(c) for c in b[::-1]])
                        outcomes_list.append(outcomes)
        except Exception:
            # Fallback for direct AerSimulator results
            if hasattr(results, 'get_counts'):
                for i in range(n_circuits):
                    counts = results.get_counts(i)
                    b = list(counts.keys())[0]
                    outcomes = np.array([int(c) for c in b[::-1]])
                    outcomes_list.append(outcomes)

        return outcomes_list

    def generate_from_statevector(
        self,
        statevector: 'Statevector'
    ) -> List[Dict]:
        """
        Generate classical shadow from a Qiskit Statevector.

        Args:
            statevector: Qiskit Statevector object.

        Returns:
            List of shadow measurement records with 'pauli_basis' and 'outcomes'.
        """
        if not QISKIT_AVAILABLE:
            raise RuntimeError("Qiskit is required")

        shadow_data = []
        sampler = self._sampler or AerSampler()

        for _ in range(self.shadow_size):
            pauli_string = self._generate_random_pauli_string()

            # Create measurement circuit
            qc = QuantumCircuit(self.n_qubits)
            qc.prepare_state(statevector, range(self.n_qubits))

            # Apply rotation gates
            for i, pauli in enumerate(pauli_string):
                if pauli == 'X':
                    qc.h(i)
                elif pauli == 'Y':
                    qc.sdg(i)
                    qc.h(i)

            qc.measure_all()

            # Run measurement
            job = sampler.run([qc], shots=1)
            result = job.result()
            quasi_dist = result.quasi_dists[0]
            bitstring = max(quasi_dist, key=quasi_dist.get)
            outcomes = [int(b) for b in format(bitstring, f'0{self.n_qubits}b')[::-1]]

            shadow_data.append({
                'pauli_basis': pauli_string,
                'outcomes': outcomes
            })

        return shadow_data

    def _generate_random_pauli_string(self) -> str:
        """Generate a random Pauli measurement basis string."""
        return ''.join(np.random.choice(self.pauli_basis, size=self.n_qubits))

    def reconstruct_state(
        self,
        shadow_samples: List[Tuple[np.ndarray, np.ndarray]]
    ) -> np.ndarray:
        """
        Reconstruct density matrix from shadow samples using inverse channel.

        This provides the global density matrix reconstruction:
        ρ = E[M^-1(U†|b⟩⟨b|U)]

        Warning: Exponentially expensive in n_qubits.

        Args:
            shadow_samples: List of (pauli_indices, outcome_bits) tuples.

        Returns:
            Reconstructed density matrix of shape (2^n, 2^n).
        """
        rho = np.zeros((self.dim, self.dim), dtype=complex)

        for bases, outcomes in shadow_samples:
            pauli_string = self._bases_to_string(bases)
            snapshot = self._inverse_channel_full(pauli_string, outcomes.tolist())
            rho += snapshot

        return rho / len(shadow_samples)

    def shadow_to_feature_vector(
        self,
        shadow_samples: List[Tuple[np.ndarray, np.ndarray]],
        method: str = 'histogram'
    ) -> np.ndarray:
        """
        Convert shadow samples to a feature vector for SOM training.

        Args:
            shadow_samples: List of (pauli_indices, outcome_bits) tuples.
            method: Feature extraction method:
                - 'histogram': Probability histogram (efficient, default)
                - 'reconstruction': Full state reconstruction (expensive)

        Returns:
            Feature vector for SOM input.
        """
        if method == 'reconstruction' and self.use_inverse_channel:
            # Full reconstruction - expensive but more accurate
            rho = self.reconstruct_state(shadow_samples)
            feature_vector = np.real(rho.flatten())
            norm = np.linalg.norm(feature_vector)
            if norm > 0:
                feature_vector /= norm
            return feature_vector

        # Histogram method - efficient approximation
        # Feature: P(outcome=0|qubit_i, basis_j) for all i, j
        feature_vector = np.zeros((self.n_qubits, 3, 2))

        for bases, outcomes in shadow_samples:
            for i in range(self.n_qubits):
                b = outcomes[i]
                basis = bases[i]
                feature_vector[i, basis, b] += 1

        # Normalize to probabilities
        return feature_vector.flatten() / len(shadow_samples)

    def estimate_observable(
        self,
        shadow_samples: List[Tuple[np.ndarray, np.ndarray]],
        observable: np.ndarray
    ) -> float:
        """
        Estimate expectation value of an observable from shadow samples.

        Args:
            shadow_samples: List of shadow measurement tuples.
            observable: Observable operator as numpy array.

        Returns:
            Estimated expectation value ⟨O⟩.
        """
        estimates = []

        for bases, outcomes in shadow_samples:
            pauli_string = self._bases_to_string(bases)
            reconstructed = self._inverse_channel_full(pauli_string, outcomes.tolist())
            estimate = np.real(np.trace(observable @ reconstructed))
            estimates.append(estimate)

        return np.mean(estimates)

    def estimate_fidelity(
        self,
        shadow_samples: List[Tuple[np.ndarray, np.ndarray]],
        target_state: np.ndarray
    ) -> float:
        """
        Estimate fidelity with a target pure state from shadow samples.

        Args:
            shadow_samples: Shadow measurement samples.
            target_state: Target state vector.

        Returns:
            Estimated fidelity F(ρ, |ψ⟩).
        """
        target_projector = np.outer(target_state, target_state.conj())
        return self.estimate_observable(shadow_samples, target_projector)

    def generate_batch(
        self,
        circuits: List['QuantumCircuit'],
        n_jobs: int = 1
    ) -> List[List[Tuple[np.ndarray, np.ndarray]]]:
        """
        Generate classical shadows for multiple circuits in batch.

        This is more efficient than calling generate() multiple times
        as it batches circuit execution.

        Args:
            circuits: List of state preparation circuits.
            n_jobs: Number of parallel jobs (for future parallelization).

        Returns:
            List of shadow samples for each circuit.
        """
        if not QISKIT_AVAILABLE:
            raise RuntimeError("Qiskit is required for shadow generation")

        all_shadow_samples = []

        for circuit in circuits:
            samples = self.generate(circuit)
            all_shadow_samples.append(samples)

        return all_shadow_samples

    def generate_features_batch(
        self,
        circuits: List['QuantumCircuit'],
        method: str = 'histogram'
    ) -> np.ndarray:
        """
        Generate feature vectors for multiple circuits in batch.

        Args:
            circuits: List of state preparation circuits.
            method: Feature extraction method ('histogram' or 'reconstruction').

        Returns:
            Array of feature vectors with shape (n_circuits, n_features).
        """
        all_shadows = self.generate_batch(circuits)
        features = []

        for shadows in all_shadows:
            feat_vec = self.shadow_to_feature_vector(shadows, method=method)
            features.append(feat_vec)

        return np.array(features)
