"""
Sparse Matrix Support for QSOM.

This module provides sparse matrix implementations for efficient handling
of large quantum systems where full density matrices become impractical.

For n qubits, the full density matrix has 2^n x 2^n = 4^n elements.
Sparse representations can significantly reduce memory and computation
for states with structure (e.g., low-rank, localized).
"""

from typing import Optional, Tuple, List, Union
import numpy as np

try:
    from scipy import sparse
    from scipy.sparse import csr_matrix, csc_matrix, lil_matrix
    from scipy.sparse.linalg import eigsh, norm as sparse_norm
    SCIPY_SPARSE_AVAILABLE = True
except ImportError:
    SCIPY_SPARSE_AVAILABLE = False


class SparseClassicalShadow:
    """
    Classical Shadow implementation using sparse matrices.

    This is useful for large qubit systems where full density matrix
    reconstruction would be memory-prohibitive.
    """

    def __init__(
        self,
        n_qubits: int,
        shadow_size: int = 100,
        sparsity_threshold: float = 1e-10
    ):
        """
        Initialize sparse shadow generator.

        Args:
            n_qubits: Number of qubits.
            shadow_size: Number of random measurements.
            sparsity_threshold: Values below this are set to zero.
        """
        if not SCIPY_SPARSE_AVAILABLE:
            raise ImportError("scipy is required for sparse matrix support")

        self.n_qubits = n_qubits
        self.shadow_size = shadow_size
        self.sparsity_threshold = sparsity_threshold
        self.dim = 2 ** n_qubits

        # Sparse Pauli operators
        self._build_sparse_paulis()

    def _build_sparse_paulis(self) -> None:
        """Build sparse representations of Pauli operators."""
        # Single-qubit Paulis as sparse matrices
        self.pauli_X = csr_matrix(np.array([[0, 1], [1, 0]], dtype=complex))
        self.pauli_Y = csr_matrix(np.array([[0, -1j], [1j, 0]], dtype=complex))
        self.pauli_Z = csr_matrix(np.array([[1, 0], [0, -1]], dtype=complex))
        self.pauli_I = sparse.eye(2, format='csr', dtype=complex)

        self.paulis = {
            'I': self.pauli_I,
            'X': self.pauli_X,
            'Y': self.pauli_Y,
            'Z': self.pauli_Z
        }

    def _sparse_kron(self, matrices: List[csr_matrix]) -> csr_matrix:
        """Compute sparse Kronecker product of multiple matrices."""
        result = matrices[0]
        for mat in matrices[1:]:
            result = sparse.kron(result, mat, format='csr')
        return result

    def _sparse_projector(self, pauli: str, outcome: int) -> csr_matrix:
        """
        Create sparse projector for single-qubit Pauli measurement.

        Args:
            pauli: Pauli basis ('X', 'Y', 'Z').
            outcome: Measurement outcome (0 or 1).

        Returns:
            Sparse 2x2 projector matrix.
        """
        if pauli == 'X':
            if outcome == 0:
                vec = np.array([1, 1]) / np.sqrt(2)
            else:
                vec = np.array([1, -1]) / np.sqrt(2)
        elif pauli == 'Y':
            if outcome == 0:
                vec = np.array([1, 1j]) / np.sqrt(2)
            else:
                vec = np.array([1, -1j]) / np.sqrt(2)
        else:  # Z
            if outcome == 0:
                vec = np.array([1, 0])
            else:
                vec = np.array([0, 1])

        proj = np.outer(vec, vec.conj())
        return csr_matrix(proj)

    def reconstruct_state_sparse(
        self,
        shadow_samples: List[Tuple[np.ndarray, np.ndarray]]
    ) -> csr_matrix:
        """
        Reconstruct density matrix as sparse matrix from shadow samples.

        Uses the inverse channel: M^-1(|b><b|) = 3|b><b| - I

        Args:
            shadow_samples: List of (pauli_indices, outcome_bits) tuples.

        Returns:
            Sparse density matrix reconstruction.
        """
        # Use LIL format for efficient construction
        rho = lil_matrix((self.dim, self.dim), dtype=complex)

        for bases, outcomes in shadow_samples:
            # Build tensor product of single-qubit inverse channels
            local_ops = []
            for i, (basis_idx, outcome) in enumerate(zip(bases, outcomes)):
                pauli = ['X', 'Y', 'Z'][basis_idx]
                proj = self._sparse_projector(pauli, outcome)
                # Inverse channel: 3 * proj - I
                inv_channel = 3 * proj - self.pauli_I
                local_ops.append(inv_channel)

            # Tensor product
            snapshot = self._sparse_kron(local_ops)
            rho += snapshot

        # Convert to CSR and normalize
        rho = rho.tocsr() / len(shadow_samples)

        # Apply sparsity threshold
        rho.data[np.abs(rho.data) < self.sparsity_threshold] = 0
        rho.eliminate_zeros()

        return rho

    def shadow_to_sparse_features(
        self,
        shadow_samples: List[Tuple[np.ndarray, np.ndarray]],
        n_components: Optional[int] = None
    ) -> np.ndarray:
        """
        Convert shadow to feature vector using sparse reconstruction.

        For large systems, uses truncated eigendecomposition for dimensionality
        reduction instead of full flattening.

        Args:
            shadow_samples: Shadow measurement samples.
            n_components: Number of principal components to keep.
                         If None, uses min(100, dim).

        Returns:
            Feature vector.
        """
        rho_sparse = self.reconstruct_state_sparse(shadow_samples)

        if n_components is None:
            n_components = min(100, self.dim - 2)

        # Use sparse eigendecomposition for feature extraction
        try:
            # Get largest eigenvalues/vectors
            eigenvalues, eigenvectors = eigsh(
                rho_sparse,
                k=n_components,
                which='LM'
            )
            # Feature vector from eigenspectrum
            features = np.concatenate([
                np.real(eigenvalues),
                np.abs(eigenvectors[:, 0])[:min(n_components, self.dim)]
            ])
        except Exception:
            # Fallback to histogram features
            features = self._histogram_features(shadow_samples)

        # Normalize
        norm = np.linalg.norm(features)
        if norm > 0:
            features /= norm

        return features

    def _histogram_features(
        self,
        shadow_samples: List[Tuple[np.ndarray, np.ndarray]]
    ) -> np.ndarray:
        """Compute histogram features (fallback for sparse case)."""
        feature_vector = np.zeros((self.n_qubits, 3, 2))

        for bases, outcomes in shadow_samples:
            for i in range(self.n_qubits):
                b = outcomes[i]
                basis = bases[i]
                feature_vector[i, basis, b] += 1

        return feature_vector.flatten() / len(shadow_samples)


class SparseSOM:
    """
    Self-Organizing Map optimized for sparse input data.

    Uses sparse matrix operations where beneficial for memory efficiency.
    """

    def __init__(
        self,
        grid_size: Tuple[int, int],
        input_dim: int,
        learning_rate: float = 0.5,
        sigma: float = 1.0,
        use_sparse_weights: bool = False
    ):
        """
        Initialize sparse-aware SOM.

        Args:
            grid_size: (height, width) of the map.
            input_dim: Dimension of input vectors.
            learning_rate: Initial learning rate.
            sigma: Initial neighborhood radius.
            use_sparse_weights: Store weights as sparse matrices.
        """
        if not SCIPY_SPARSE_AVAILABLE:
            raise ImportError("scipy is required for sparse SOM")

        self.height, self.width = grid_size
        self.input_dim = input_dim
        self.learning_rate = learning_rate
        self.sigma = sigma
        self.use_sparse_weights = use_sparse_weights

        # Initialize weights
        if use_sparse_weights:
            # Sparse weight initialization (random sparse)
            density = min(0.1, 100 / input_dim)  # Adaptive density
            self.weights = []
            for i in range(self.height):
                row = []
                for j in range(self.width):
                    w = sparse.random(1, input_dim, density=density, format='csr')
                    w = w / (sparse_norm(w) + 1e-10)
                    row.append(w)
                self.weights.append(row)
        else:
            self.weights = np.random.randn(self.height, self.width, input_dim)
            # Normalize
            norms = np.linalg.norm(self.weights, axis=2, keepdims=True)
            self.weights /= (norms + 1e-10)

    def _sparse_distance(self, x: csr_matrix, y: csr_matrix) -> float:
        """Compute distance between sparse vectors."""
        diff = x - y
        return sparse_norm(diff)

    def find_bmu_sparse(self, x: Union[np.ndarray, csr_matrix]) -> Tuple[int, int]:
        """
        Find BMU for sparse or dense input.

        Args:
            x: Input vector (sparse or dense).

        Returns:
            Grid coordinates of BMU.
        """
        if isinstance(x, np.ndarray):
            x_sparse = csr_matrix(x.reshape(1, -1))
        else:
            x_sparse = x

        min_dist = float('inf')
        bmu = (0, 0)

        for i in range(self.height):
            for j in range(self.width):
                if self.use_sparse_weights:
                    dist = self._sparse_distance(x_sparse, self.weights[i][j])
                else:
                    w = csr_matrix(self.weights[i, j].reshape(1, -1))
                    dist = self._sparse_distance(x_sparse, w)

                if dist < min_dist:
                    min_dist = dist
                    bmu = (i, j)

        return bmu

    def train_sparse(
        self,
        data: Union[np.ndarray, List[csr_matrix]],
        n_iterations: int = 1000,
        verbose: bool = True
    ) -> None:
        """
        Train SOM on sparse data.

        Args:
            data: Training data (dense array or list of sparse vectors).
            n_iterations: Number of training iterations.
            verbose: Print progress.
        """
        # Convert to list of sparse if dense
        if isinstance(data, np.ndarray):
            data_sparse = [csr_matrix(x.reshape(1, -1)) for x in data]
        else:
            data_sparse = data

        n_samples = len(data_sparse)

        for t in range(n_iterations):
            # Decay parameters
            progress = t / n_iterations
            current_lr = self.learning_rate * np.exp(-4 * progress)
            current_sigma = max(self.sigma * np.exp(-4 * progress), 0.1)

            # Random sample
            idx = np.random.randint(0, n_samples)
            x = data_sparse[idx]

            # Find BMU
            bmu = self.find_bmu_sparse(x)

            # Update weights
            self._update_weights_sparse(x, bmu, current_lr, current_sigma)

            if verbose and t % 100 == 0:
                print(f"Iteration {t}: LR={current_lr:.4f}, σ={current_sigma:.4f}")

    def _update_weights_sparse(
        self,
        x: csr_matrix,
        bmu: Tuple[int, int],
        lr: float,
        sigma: float
    ) -> None:
        """Update weights using neighborhood function."""
        for i in range(self.height):
            for j in range(self.width):
                # Grid distance
                dist_sq = (i - bmu[0])**2 + (j - bmu[1])**2
                influence = np.exp(-dist_sq / (2 * sigma**2)) * lr

                if influence < 1e-6:
                    continue

                if self.use_sparse_weights:
                    # Sparse update
                    diff = x - self.weights[i][j]
                    self.weights[i][j] = self.weights[i][j] + influence * diff
                else:
                    # Dense update
                    x_dense = x.toarray().flatten()
                    self.weights[i, j] += influence * (x_dense - self.weights[i, j])


def estimate_sparsity(n_qubits: int) -> dict:
    """
    Estimate memory requirements and recommend sparse vs dense.

    Args:
        n_qubits: Number of qubits.

    Returns:
        Dictionary with recommendations.
    """
    dim = 2 ** n_qubits
    dense_elements = dim * dim
    dense_memory_mb = dense_elements * 16 / (1024 * 1024)  # complex128

    # Typical sparsity for physical states
    typical_sparsity = min(0.01, 1000 / dense_elements)
    sparse_elements = int(dense_elements * typical_sparsity)
    sparse_memory_mb = sparse_elements * 24 / (1024 * 1024)  # CSR overhead

    return {
        'n_qubits': n_qubits,
        'dimension': dim,
        'dense_elements': dense_elements,
        'dense_memory_mb': dense_memory_mb,
        'sparse_memory_mb': sparse_memory_mb,
        'recommended': 'sparse' if n_qubits > 10 else 'dense',
        'memory_savings': f"{(1 - sparse_memory_mb/dense_memory_mb)*100:.1f}%"
    }
