"""
Clifford-Based Classical Shadows.

This module implements classical shadows using random Clifford circuits,
which can provide better estimation for certain observables compared to
Pauli-based shadows.

References:
- Huang et al., "Predicting many properties of a quantum system from very
  few measurements" (2020)
- Elben et al., "The randomized measurement toolbox" (2022)
"""

from typing import List, Tuple, Optional, Any, Dict
import numpy as np

try:
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Clifford, random_clifford, Operator
    from qiskit.circuit.library import HGate, SGate, SdgGate, CXGate
    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False


class CliffordShadow:
    """
    Classical Shadow implementation using random Clifford circuits.

    Clifford shadows measure in random Clifford bases, providing:
    - Better estimation for local observables
    - Efficient classical post-processing
    - Tomographic completeness

    The inverse channel for Clifford shadows is:
    M^-1(ρ) = (2^n + 1) * ρ - I

    For local observables, Clifford shadows often require fewer
    measurements than Pauli shadows.
    """

    def __init__(
        self,
        n_qubits: int,
        backend: Any = None,
        shadow_size: int = 100,
        n_shots: int = 1000,
        use_global_clifford: bool = True
    ):
        """
        Initialize Clifford shadow generator.

        Args:
            n_qubits: Number of qubits.
            backend: Quantum backend for measurements.
            shadow_size: Number of random Clifford circuits.
            n_shots: Shots per circuit (if batching).
            use_global_clifford: Use global n-qubit Cliffords (True) or
                                 tensor product of single-qubit Cliffords (False).
        """
        if not QISKIT_AVAILABLE:
            raise ImportError("Qiskit is required for Clifford shadows")

        self.n_qubits = n_qubits
        self.backend = backend
        self.shadow_size = shadow_size
        self.n_shots = n_shots
        self.use_global_clifford = use_global_clifford
        self.dim = 2 ** n_qubits

        # Cache for random Cliffords
        self._clifford_cache: List[Clifford] = []

    def generate_random_clifford(self) -> Clifford:
        """
        Generate a random Clifford circuit.

        Returns:
            Random Clifford operator.
        """
        if self.use_global_clifford:
            return random_clifford(self.n_qubits)
        else:
            # Tensor product of single-qubit Cliffords
            cliffords = [random_clifford(1) for _ in range(self.n_qubits)]
            return self._tensor_cliffords(cliffords)

    def _tensor_cliffords(self, cliffords: List[Clifford]) -> Clifford:
        """Tensor product of single-qubit Cliffords."""
        result = cliffords[0]
        for cliff in cliffords[1:]:
            result = result.tensor(cliff)
        return result

    def generate(
        self,
        circuit: 'QuantumCircuit'
    ) -> List[Tuple[Clifford, np.ndarray]]:
        """
        Generate classical shadow by measuring in random Clifford bases.

        Args:
            circuit: Quantum circuit preparing the state.

        Returns:
            List of (Clifford, measurement_outcomes) tuples.
        """
        shadow_samples = []

        for _ in range(self.shadow_size):
            # Generate random Clifford
            cliff = self.generate_random_clifford()

            # Create measurement circuit
            meas_circuit = circuit.copy()
            meas_circuit.barrier()

            # Apply inverse Clifford (to measure in Clifford basis)
            cliff_inv = cliff.adjoint()
            meas_circuit.append(cliff_inv.to_instruction(), range(self.n_qubits))
            meas_circuit.measure_all()

            # Run circuit
            if self.backend is not None:
                result = self.backend.run(meas_circuit)
                outcomes = self._extract_outcomes(result)
            else:
                # Simulator fallback
                outcomes = self._simulate_measurement(circuit, cliff)

            shadow_samples.append((cliff, outcomes))

        return shadow_samples

    def _extract_outcomes(self, result: Any) -> np.ndarray:
        """Extract measurement outcomes from backend result."""
        if hasattr(result, 'quasi_dists'):
            dist = result.quasi_dists[0]
            # Sample from distribution
            probs = np.array(list(dist.values()))
            probs = np.maximum(probs, 0)
            probs /= probs.sum()
            keys = list(dist.keys())
            idx = np.random.choice(len(keys), p=probs)
            bitstring = format(keys[idx], f'0{self.n_qubits}b')
            return np.array([int(b) for b in bitstring[::-1]])

        return np.random.randint(0, 2, self.n_qubits)

    def _simulate_measurement(
        self,
        circuit: 'QuantumCircuit',
        cliff: Clifford
    ) -> np.ndarray:
        """Simulate measurement for testing without backend."""
        # Simple simulation using state vector
        try:
            from qiskit.quantum_info import Statevector

            # Get state
            sv = Statevector.from_instruction(circuit)

            # Apply inverse Clifford
            cliff_op = Operator(cliff)
            rotated_sv = sv.evolve(cliff_op.adjoint())

            # Sample from probabilities
            probs = np.abs(rotated_sv.data) ** 2
            idx = np.random.choice(len(probs), p=probs)
            bitstring = format(idx, f'0{self.n_qubits}b')
            return np.array([int(b) for b in bitstring[::-1]])

        except ImportError:
            return np.random.randint(0, 2, self.n_qubits)

    def reconstruct_density_matrix(
        self,
        shadow_samples: List[Tuple[Clifford, np.ndarray]]
    ) -> np.ndarray:
        """
        Reconstruct density matrix from Clifford shadow samples.

        Uses inverse channel: M^-1(|b><b|) = (2^n + 1)|b><b| - I

        Args:
            shadow_samples: List of (Clifford, outcomes) tuples.

        Returns:
            Reconstructed density matrix.
        """
        rho = np.zeros((self.dim, self.dim), dtype=complex)

        for cliff, outcomes in shadow_samples:
            # Convert outcomes to computational basis state
            idx = sum(b * (2 ** i) for i, b in enumerate(outcomes))
            b_state = np.zeros(self.dim, dtype=complex)
            b_state[idx] = 1.0

            # Get Clifford unitary
            cliff_unitary = Operator(cliff).data

            # Rotate back to original basis: |ψ> = U|b>
            psi = cliff_unitary @ b_state

            # Outer product
            snapshot = np.outer(psi, psi.conj())

            # Apply inverse channel
            inverse_snapshot = (self.dim + 1) * snapshot - np.eye(self.dim)

            rho += inverse_snapshot

        # Average
        rho /= len(shadow_samples)

        return rho

    def estimate_observable(
        self,
        shadow_samples: List[Tuple[Clifford, np.ndarray]],
        observable: np.ndarray
    ) -> complex:
        """
        Estimate expectation value of observable from shadow.

        Args:
            shadow_samples: Shadow measurement samples.
            observable: Observable matrix.

        Returns:
            Estimated expectation value.
        """
        total = 0.0

        for cliff, outcomes in shadow_samples:
            # Construct inverse channel output for this sample
            idx = sum(b * (2 ** i) for i, b in enumerate(outcomes))
            b_state = np.zeros(self.dim, dtype=complex)
            b_state[idx] = 1.0

            cliff_unitary = Operator(cliff).data
            psi = cliff_unitary @ b_state
            snapshot = np.outer(psi, psi.conj())
            inverse_snapshot = (self.dim + 1) * snapshot - np.eye(self.dim)

            # Compute Tr(O * inverse_snapshot)
            total += np.trace(observable @ inverse_snapshot)

        return total / len(shadow_samples)

    def shadow_to_feature_vector(
        self,
        shadow_samples: List[Tuple[Clifford, np.ndarray]],
        method: str = 'pauli_expectation'
    ) -> np.ndarray:
        """
        Convert shadow to feature vector for SOM input.

        Args:
            shadow_samples: Shadow measurement samples.
            method: Feature extraction method:
                   - 'pauli_expectation': Local Pauli expectation values
                   - 'histogram': Measurement outcome histogram
                   - 'density': Flattened density matrix elements

        Returns:
            Feature vector.
        """
        if method == 'pauli_expectation':
            return self._pauli_features(shadow_samples)
        elif method == 'histogram':
            return self._histogram_features(shadow_samples)
        elif method == 'density':
            rho = self.reconstruct_density_matrix(shadow_samples)
            # Real and imaginary parts
            features = np.concatenate([rho.real.flatten(), rho.imag.flatten()])
            norm = np.linalg.norm(features)
            return features / norm if norm > 0 else features
        else:
            raise ValueError(f"Unknown method: {method}")

    def _pauli_features(
        self,
        shadow_samples: List[Tuple[Clifford, np.ndarray]]
    ) -> np.ndarray:
        """Extract Pauli expectation value features."""
        features = []

        # Single-qubit Paulis
        pauli_x = np.array([[0, 1], [1, 0]], dtype=complex)
        pauli_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
        pauli_z = np.array([[1, 0], [0, -1]], dtype=complex)
        identity = np.eye(2, dtype=complex)

        for i in range(self.n_qubits):
            for pauli in [pauli_x, pauli_y, pauli_z]:
                # Build full observable
                ops = [identity] * self.n_qubits
                ops[i] = pauli
                observable = self._kron_list(ops)

                exp_val = self.estimate_observable(shadow_samples, observable)
                features.append(np.real(exp_val))

        # Normalize
        features = np.array(features)
        norm = np.linalg.norm(features)
        return features / norm if norm > 0 else features

    def _histogram_features(
        self,
        shadow_samples: List[Tuple[Clifford, np.ndarray]]
    ) -> np.ndarray:
        """Compute histogram features from outcomes."""
        feature_vector = np.zeros(self.dim)

        for _, outcomes in shadow_samples:
            idx = sum(b * (2 ** i) for i, b in enumerate(outcomes))
            feature_vector[idx] += 1

        # Normalize
        feature_vector /= len(shadow_samples)
        return feature_vector

    def _kron_list(self, matrices: List[np.ndarray]) -> np.ndarray:
        """Compute Kronecker product of list of matrices."""
        result = matrices[0]
        for mat in matrices[1:]:
            result = np.kron(result, mat)
        return result


class LocalCliffordShadow(CliffordShadow):
    """
    Classical shadows using local (single-qubit) Clifford gates.

    This is computationally cheaper than global Cliffords and provides
    optimal sample complexity for local observables.
    """

    def __init__(
        self,
        n_qubits: int,
        backend: Any = None,
        shadow_size: int = 100,
        n_shots: int = 1000
    ):
        """
        Initialize local Clifford shadow generator.

        Args:
            n_qubits: Number of qubits.
            backend: Quantum backend.
            shadow_size: Number of measurements.
            n_shots: Shots per circuit.
        """
        super().__init__(
            n_qubits=n_qubits,
            backend=backend,
            shadow_size=shadow_size,
            n_shots=n_shots,
            use_global_clifford=False
        )

        # Pre-generate single-qubit Clifford group
        self._single_qubit_cliffords = self._generate_single_qubit_cliffords()

    def _generate_single_qubit_cliffords(self) -> List['QuantumCircuit']:
        """Generate all 24 single-qubit Clifford gates."""
        cliffords = []

        # Identity
        qc = QuantumCircuit(1)
        cliffords.append(qc)

        # H, S, HS, SH, HSH, SHS
        gates_sequences = [
            ['h'],
            ['s'],
            ['h', 's'],
            ['s', 'h'],
            ['h', 's', 'h'],
            ['s', 'h', 's'],
            ['h', 's', 's'],
            ['s', 's', 'h'],
            ['h', 's', 's', 'h'],
            ['s', 's'],
            ['s', 's', 's'],
            ['h', 's', 's', 's'],
            ['s', 'h', 's', 's'],
            ['h', 's', 'h', 's'],
            ['s', 'h', 's', 'h'],
            ['h', 's', 's', 'h', 's'],
            ['s', 'h', 's', 's', 'h'],
            ['h', 's', 'h', 's', 's'],
            ['s', 's', 'h', 's'],
            ['h', 's', 's', 's', 'h'],
            ['s', 'h', 's', 's', 's'],
            ['h', 's', 'h', 's', 's', 's'],
            ['s', 's', 'h', 's', 's'],
        ]

        for seq in gates_sequences:
            qc = QuantumCircuit(1)
            for gate in seq:
                if gate == 'h':
                    qc.h(0)
                elif gate == 's':
                    qc.s(0)
            cliffords.append(qc)

        return cliffords[:24]  # Ensure exactly 24

    def generate(
        self,
        circuit: 'QuantumCircuit'
    ) -> List[Tuple[List[int], np.ndarray]]:
        """
        Generate local Clifford shadow.

        Returns list of (clifford_indices, outcomes) where clifford_indices
        identifies which single-qubit Clifford was applied to each qubit.
        """
        shadow_samples = []
        n_cliffords = len(self._single_qubit_cliffords)

        for _ in range(self.shadow_size):
            # Random single-qubit Clifford index for each qubit
            cliff_indices = np.random.randint(0, n_cliffords, self.n_qubits)

            # Build measurement circuit
            meas_circuit = circuit.copy()
            meas_circuit.barrier()

            # Apply inverse of random Cliffords
            for q, cliff_idx in enumerate(cliff_indices):
                cliff_circuit = self._single_qubit_cliffords[cliff_idx]
                # Apply inverse (transpose)
                meas_circuit = meas_circuit.compose(
                    cliff_circuit.inverse(), [q]
                )

            meas_circuit.measure_all()

            # Run or simulate
            if self.backend is not None:
                result = self.backend.run(meas_circuit)
                outcomes = self._extract_outcomes(result)
            else:
                outcomes = self._simulate_local_measurement(
                    circuit, cliff_indices
                )

            shadow_samples.append((cliff_indices.tolist(), outcomes))

        return shadow_samples

    def _simulate_local_measurement(
        self,
        circuit: 'QuantumCircuit',
        cliff_indices: np.ndarray
    ) -> np.ndarray:
        """Simulate local Clifford measurement."""
        try:
            from qiskit.quantum_info import Statevector

            # Get state
            sv = Statevector.from_instruction(circuit)

            # Apply inverse local Cliffords
            for q, cliff_idx in enumerate(cliff_indices):
                cliff_circuit = self._single_qubit_cliffords[cliff_idx]
                sv = sv.evolve(cliff_circuit.inverse(), [q])

            # Sample
            probs = np.abs(sv.data) ** 2
            idx = np.random.choice(len(probs), p=probs)
            bitstring = format(idx, f'0{self.n_qubits}b')
            return np.array([int(b) for b in bitstring[::-1]])

        except ImportError:
            return np.random.randint(0, 2, self.n_qubits)

    def reconstruct_density_matrix(
        self,
        shadow_samples: List[Tuple[List[int], np.ndarray]]
    ) -> np.ndarray:
        """
        Reconstruct density matrix from local Clifford shadow.

        For local Cliffords, the inverse channel is tensor product
        of single-qubit inverse channels.
        """
        rho = np.zeros((self.dim, self.dim), dtype=complex)

        for cliff_indices, outcomes in shadow_samples:
            # Build tensor product of single-qubit snapshots
            local_snapshots = []

            for q, cliff_idx in enumerate(cliff_indices):
                # Single-qubit outcome state
                b = outcomes[q]
                b_state = np.array([1 - b, b], dtype=complex)

                # Get Clifford unitary for this qubit
                cliff_circuit = self._single_qubit_cliffords[cliff_idx]
                try:
                    from qiskit.quantum_info import Operator as QOp
                    cliff_unitary = QOp(cliff_circuit).data
                except Exception:
                    cliff_unitary = np.eye(2, dtype=complex)

                # Rotate state
                psi = cliff_unitary @ b_state
                snapshot = np.outer(psi, psi.conj())

                # Single-qubit inverse channel: 3|ψ><ψ| - I
                inv_snapshot = 3 * snapshot - np.eye(2)
                local_snapshots.append(inv_snapshot)

            # Tensor product
            full_snapshot = local_snapshots[0]
            for snap in local_snapshots[1:]:
                full_snapshot = np.kron(full_snapshot, snap)

            rho += full_snapshot

        return rho / len(shadow_samples)


def compare_shadow_methods(
    circuit: 'QuantumCircuit',
    n_qubits: int,
    shadow_size: int = 100,
    backend: Any = None
) -> Dict[str, np.ndarray]:
    """
    Compare Pauli vs Clifford shadow reconstructions.

    Args:
        circuit: Circuit preparing state to characterize.
        n_qubits: Number of qubits.
        shadow_size: Number of shadow samples.
        backend: Quantum backend (optional).

    Returns:
        Dictionary with density matrices from each method.
    """
    from .shadows import ClassicalShadow

    results = {}

    # Pauli shadows
    pauli_shadow = ClassicalShadow(
        n_qubits=n_qubits,
        backend=backend,
        shadow_size=shadow_size
    )
    pauli_samples = pauli_shadow.generate(circuit)
    results['pauli'] = pauli_shadow.reconstruct_state(pauli_samples)

    # Local Clifford shadows
    local_cliff = LocalCliffordShadow(
        n_qubits=n_qubits,
        backend=backend,
        shadow_size=shadow_size
    )
    local_samples = local_cliff.generate(circuit)
    results['local_clifford'] = local_cliff.reconstruct_density_matrix(local_samples)

    # Global Clifford shadows (if small enough)
    if n_qubits <= 4:
        global_cliff = CliffordShadow(
            n_qubits=n_qubits,
            backend=backend,
            shadow_size=shadow_size,
            use_global_clifford=True
        )
        global_samples = global_cliff.generate(circuit)
        results['global_clifford'] = global_cliff.reconstruct_density_matrix(global_samples)

    return results
