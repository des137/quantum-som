"""
Utility Functions for QSOM.

This module provides helper functions for quantum state generation,
data preprocessing, and other common operations.
"""

from typing import List, Optional, Union
import numpy as np

# Qiskit imports with graceful fallback
try:
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Statevector, DensityMatrix
    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False


def generate_quantum_states(
    n_states: int,
    n_qubits: int,
    state_type: str = 'random',
    noise_level: float = 0.0,
    random_seed: Optional[int] = None
) -> List[Union['Statevector', np.ndarray]]:
    """
    Generate sample quantum states for testing and experimentation.

    Args:
        n_states: Number of states to generate.
        n_qubits: Number of qubits per state.
        state_type: Type of states to generate:
            - 'random': Random pure states (Haar random)
            - 'bell': Bell states (for 2 qubits) or GHZ states
            - 'ghz': GHZ states (|00...0⟩ + |11...1⟩)/√2
            - 'w': W states (equal superposition of single excitations)
            - 'product': Random product states
            - 'cluster': 1D cluster states
        noise_level: Amount of depolarizing noise to add (0.0-1.0).
        random_seed: Random seed for reproducibility.

    Returns:
        List of quantum states (Statevector objects if Qiskit available,
        else numpy arrays).

    Examples:
        >>> states = generate_quantum_states(100, 3, state_type='random')
        >>> len(states)
        100
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    dim = 2 ** n_qubits
    states = []

    for _ in range(n_states):
        if QISKIT_AVAILABLE:
            state = _generate_state_qiskit(n_qubits, dim, state_type)
            if noise_level > 0:
                state = _apply_noise(state, noise_level)
            states.append(state)
        else:
            state = _generate_state_numpy(n_qubits, dim, state_type)
            if noise_level > 0:
                state = _apply_noise_numpy(state, noise_level)
            states.append(state)

    return states


def _generate_state_qiskit(n_qubits: int, dim: int, state_type: str) -> 'Statevector':
    """Generate quantum state using Qiskit."""
    if state_type == 'random':
        return Statevector.from_int(0, dim).evolve(
            _random_unitary(dim)
        )

    elif state_type == 'bell':
        if n_qubits == 2:
            # |Φ+⟩ = (|00⟩ + |11⟩)/√2
            coeffs = np.zeros(dim, dtype=complex)
            coeffs[0] = 1 / np.sqrt(2)
            coeffs[dim - 1] = 1 / np.sqrt(2)
            return Statevector(coeffs)
        else:
            # Fall back to GHZ for more qubits
            return _generate_ghz_state(n_qubits, dim)

    elif state_type == 'ghz':
        return _generate_ghz_state(n_qubits, dim)

    elif state_type == 'w':
        return _generate_w_state(n_qubits, dim)

    elif state_type == 'product':
        return _generate_product_state(n_qubits)

    elif state_type == 'cluster':
        return _generate_cluster_state(n_qubits)

    else:
        # Default to random
        return Statevector.from_int(0, dim).evolve(_random_unitary(dim))


def _generate_state_numpy(n_qubits: int, dim: int, state_type: str) -> np.ndarray:
    """Generate quantum state using numpy (fallback)."""
    if state_type == 'random':
        # Haar-random pure state
        real = np.random.randn(dim)
        imag = np.random.randn(dim)
        state = real + 1j * imag
        return state / np.linalg.norm(state)

    elif state_type == 'bell' or state_type == 'ghz':
        state = np.zeros(dim, dtype=complex)
        state[0] = 1 / np.sqrt(2)
        state[dim - 1] = 1 / np.sqrt(2)
        return state

    elif state_type == 'w':
        state = np.zeros(dim, dtype=complex)
        for i in range(n_qubits):
            idx = 2 ** i
            state[idx] = 1 / np.sqrt(n_qubits)
        return state

    elif state_type == 'product':
        # Random product state
        state = np.array([1.0], dtype=complex)
        for _ in range(n_qubits):
            theta = np.random.uniform(0, np.pi)
            phi = np.random.uniform(0, 2 * np.pi)
            qubit = np.array([
                np.cos(theta / 2),
                np.exp(1j * phi) * np.sin(theta / 2)
            ])
            state = np.kron(state, qubit)
        return state

    else:
        # Default to random
        real = np.random.randn(dim)
        imag = np.random.randn(dim)
        state = real + 1j * imag
        return state / np.linalg.norm(state)


def _generate_ghz_state(n_qubits: int, dim: int) -> 'Statevector':
    """Generate GHZ state: (|00...0⟩ + |11...1⟩)/√2."""
    coeffs = np.zeros(dim, dtype=complex)
    coeffs[0] = 1 / np.sqrt(2)
    coeffs[dim - 1] = 1 / np.sqrt(2)
    return Statevector(coeffs)


def _generate_w_state(n_qubits: int, dim: int) -> 'Statevector':
    """Generate W state: equal superposition of single-excitation states."""
    coeffs = np.zeros(dim, dtype=complex)
    for i in range(n_qubits):
        idx = 2 ** i
        coeffs[idx] = 1 / np.sqrt(n_qubits)
    return Statevector(coeffs)


def _generate_product_state(n_qubits: int) -> 'Statevector':
    """Generate random product state."""
    qc = QuantumCircuit(n_qubits)
    for i in range(n_qubits):
        theta = np.random.uniform(0, np.pi)
        phi = np.random.uniform(0, 2 * np.pi)
        qc.ry(theta, i)
        qc.rz(phi, i)
    return Statevector(qc)


def _generate_cluster_state(n_qubits: int) -> 'Statevector':
    """Generate 1D cluster state."""
    qc = QuantumCircuit(n_qubits)
    # Apply Hadamard to all qubits
    for i in range(n_qubits):
        qc.h(i)
    # Apply CZ gates between neighbors
    for i in range(n_qubits - 1):
        qc.cz(i, i + 1)
    return Statevector(qc)


def _random_unitary(dim: int) -> np.ndarray:
    """Generate Haar-random unitary matrix."""
    # Use QR decomposition of random complex matrix
    z = (np.random.randn(dim, dim) + 1j * np.random.randn(dim, dim)) / np.sqrt(2)
    q, r = np.linalg.qr(z)
    # Make it Haar-distributed
    d = np.diagonal(r)
    ph = d / np.abs(d)
    return q * ph


def _apply_noise(state: 'Statevector', noise_level: float) -> 'Statevector':
    """Apply depolarizing noise to a pure state."""
    # Convert to density matrix
    rho = DensityMatrix(state)

    # Apply depolarizing channel
    dim = rho.dim
    identity = np.eye(dim) / dim
    noisy_rho = (1 - noise_level) * rho.data + noise_level * identity

    # Convert back to statevector (approximate for mixed states)
    return Statevector(noisy_rho)


def _apply_noise_numpy(state: np.ndarray, noise_level: float) -> np.ndarray:
    """Apply noise to numpy state representation."""
    # Add random perturbation
    perturbation = np.random.randn(len(state)) + 1j * np.random.randn(len(state))
    noisy = state + noise_level * perturbation * np.linalg.norm(state)
    return noisy / np.linalg.norm(noisy)


def angle_encoding(data: np.ndarray, n_qubits: Optional[int] = None) -> List['Statevector']:
    """
    Encode classical data as quantum states using angle encoding.

    Each feature is encoded as a rotation angle on a qubit.

    Args:
        data: Classical data of shape (n_samples, n_features).
        n_qubits: Number of qubits to use. If None, uses n_features.

    Returns:
        List of Statevector objects.

    Examples:
        >>> from sklearn.datasets import load_iris
        >>> iris = load_iris()
        >>> states = angle_encoding(iris.data[:10])
    """
    if not QISKIT_AVAILABLE:
        raise RuntimeError("Qiskit is required for angle encoding")

    n_samples, n_features = data.shape

    if n_qubits is None:
        n_qubits = n_features

    # Normalize data to [0, π]
    data_normalized = np.pi * (data - data.min(axis=0)) / (data.max(axis=0) - data.min(axis=0) + 1e-10)

    states = []
    for sample in data_normalized:
        qc = QuantumCircuit(n_qubits)

        for i in range(min(n_qubits, n_features)):
            qc.ry(sample[i], i)

        states.append(Statevector(qc))

    return states


def amplitude_encoding(data: np.ndarray) -> List['Statevector']:
    """
    Encode classical data as quantum states using amplitude encoding.

    The data vector is normalized and used directly as state amplitudes.

    Args:
        data: Classical data of shape (n_samples, n_features).
              n_features must be a power of 2.

    Returns:
        List of Statevector objects.
    """
    if not QISKIT_AVAILABLE:
        raise RuntimeError("Qiskit is required for amplitude encoding")

    n_samples, n_features = data.shape

    # Check if n_features is power of 2
    if n_features & (n_features - 1) != 0:
        # Pad to next power of 2
        next_pow2 = 2 ** int(np.ceil(np.log2(n_features)))
        padded_data = np.zeros((n_samples, next_pow2))
        padded_data[:, :n_features] = data
        data = padded_data

    states = []
    for sample in data:
        # Normalize to unit vector
        norm = np.linalg.norm(sample)
        if norm > 0:
            normalized = sample / norm
        else:
            normalized = sample

        states.append(Statevector(normalized.astype(complex)))

    return states


def compute_purity(shadow_features: np.ndarray) -> float:
    """
    Estimate purity of quantum state from shadow features.

    Args:
        shadow_features: Feature vector from classical shadow.

    Returns:
        Estimated purity Tr(ρ²).
    """
    # For histogram features, estimate purity from outcome probabilities
    return np.sum(shadow_features ** 2)


def compute_entropy(shadow_features: np.ndarray) -> float:
    """
    Estimate von Neumann entropy from shadow features.

    Args:
        shadow_features: Feature vector from classical shadow.

    Returns:
        Estimated entropy S(ρ).
    """
    # Filter out zeros to avoid log(0)
    probs = shadow_features[shadow_features > 0]
    return -np.sum(probs * np.log2(probs))
