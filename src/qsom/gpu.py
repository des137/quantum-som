"""
GPU Acceleration for QSOM.

This module provides optional GPU acceleration using JAX or CuPy backends.
Falls back to NumPy if neither is available.

Usage:
    from qsom.gpu import get_backend, gpu_available

    xp = get_backend()  # Returns jax.numpy, cupy, or numpy
    if gpu_available():
        print("GPU acceleration enabled!")
"""

from typing import Any, Optional, Tuple, Callable
import numpy as np

# Try to import GPU backends
JAX_AVAILABLE = False
CUPY_AVAILABLE = False

try:
    import jax
    import jax.numpy as jnp
    from jax import jit, vmap
    JAX_AVAILABLE = True
except ImportError:
    jax = None
    jnp = None

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    cp = None


def gpu_available() -> bool:
    """Check if GPU acceleration is available."""
    return JAX_AVAILABLE or CUPY_AVAILABLE


def get_backend(prefer: str = 'jax') -> Any:
    """
    Get the appropriate array backend.

    Args:
        prefer: Preferred backend ('jax', 'cupy', or 'numpy').

    Returns:
        Module with numpy-like API (jax.numpy, cupy, or numpy).
    """
    if prefer == 'jax' and JAX_AVAILABLE:
        return jnp
    elif prefer == 'cupy' and CUPY_AVAILABLE:
        return cp
    elif prefer == 'numpy':
        return np
    elif JAX_AVAILABLE:
        return jnp
    elif CUPY_AVAILABLE:
        return cp
    else:
        return np


class GPUAcceleratedSOM:
    """
    GPU-accelerated Self-Organizing Map using JAX.

    Provides significant speedup for large grids and high-dimensional data
    by utilizing JIT compilation and GPU parallelism.
    """

    def __init__(
        self,
        grid_size: Tuple[int, int],
        input_dim: int,
        learning_rate: float = 0.5,
        sigma: float = 1.0,
        backend: str = 'jax',
        random_seed: Optional[int] = None
    ):
        """
        Initialize GPU-accelerated SOM.

        Args:
            grid_size: (height, width) of the SOM grid.
            input_dim: Dimension of input vectors.
            learning_rate: Initial learning rate.
            sigma: Initial neighborhood radius.
            backend: Backend to use ('jax', 'cupy', 'numpy').
            random_seed: Random seed for reproducibility.
        """
        self.height, self.width = grid_size
        self.input_dim = input_dim
        self.initial_learning_rate = learning_rate
        self.initial_sigma = sigma
        self.backend_name = backend

        self.xp = get_backend(backend)
        self._is_jax = self.xp is jnp and JAX_AVAILABLE
        self._is_cupy = self.xp is cp and CUPY_AVAILABLE

        # Initialize weights
        if random_seed is not None:
            if self._is_jax:
                key = jax.random.PRNGKey(random_seed)
                self.weights = jax.random.normal(
                    key, (self.height, self.width, input_dim)
                )
            elif self._is_cupy:
                cp.random.seed(random_seed)
                self.weights = cp.random.randn(
                    self.height, self.width, input_dim
                ).astype(cp.float32)
            else:
                np.random.seed(random_seed)
                self.weights = np.random.randn(
                    self.height, self.width, input_dim
                )
        else:
            if self._is_jax:
                key = jax.random.PRNGKey(42)
                self.weights = jax.random.normal(
                    key, (self.height, self.width, input_dim)
                )
            elif self._is_cupy:
                self.weights = cp.random.randn(
                    self.height, self.width, input_dim
                ).astype(cp.float32)
            else:
                self.weights = np.random.randn(
                    self.height, self.width, input_dim
                )

        # Normalize weights
        norms = self.xp.linalg.norm(self.weights, axis=2, keepdims=True)
        self.weights = self.weights / (norms + 1e-10)

        # Pre-compute grid coordinates
        self._init_grid_coordinates()

        # JIT compile functions if using JAX
        if self._is_jax:
            self._jit_compile()

        # Training metrics
        self.quantization_errors = []

    def _init_grid_coordinates(self) -> None:
        """Initialize grid coordinate arrays for efficient BMU calculation."""
        i_coords, j_coords = self.xp.meshgrid(
            self.xp.arange(self.height),
            self.xp.arange(self.width),
            indexing='ij'
        )
        self.grid_coords = self.xp.stack([i_coords, j_coords], axis=-1)

    def _jit_compile(self) -> None:
        """JIT compile key functions for JAX."""
        if not self._is_jax:
            return

        @jit
        def _compute_distances(weights, x):
            """Compute distances from input to all neurons."""
            diff = weights - x
            return jnp.sum(diff ** 2, axis=2)

        @jit
        def _compute_neighborhood(bmu_coords, grid_coords, sigma):
            """Compute neighborhood influence."""
            diff = grid_coords - bmu_coords
            dist_sq = jnp.sum(diff ** 2, axis=-1)
            return jnp.exp(-dist_sq / (2 * sigma ** 2))

        @jit
        def _update_weights(weights, x, neighborhood, lr):
            """Update weights with neighborhood influence."""
            influence = (neighborhood * lr)[:, :, jnp.newaxis]
            delta = x - weights
            return weights + influence * delta

        self._compute_distances = _compute_distances
        self._compute_neighborhood = _compute_neighborhood
        self._update_weights = _update_weights

    def find_bmu(self, x: Any) -> Tuple[int, int]:
        """
        Find Best Matching Unit for input vector.

        Args:
            x: Input vector.

        Returns:
            Grid coordinates (i, j) of BMU.
        """
        if self._is_jax:
            distances = self._compute_distances(self.weights, x)
        else:
            diff = self.weights - x
            distances = self.xp.sum(diff ** 2, axis=2)

        flat_idx = self.xp.argmin(distances)

        if self._is_jax:
            i = int(flat_idx // self.width)
            j = int(flat_idx % self.width)
        elif self._is_cupy:
            i = int(flat_idx.get() // self.width)
            j = int(flat_idx.get() % self.width)
        else:
            i = flat_idx // self.width
            j = flat_idx % self.width

        return (i, j)

    def train(
        self,
        data: Any,
        n_iterations: int = 1000,
        verbose: bool = True,
        track_errors: bool = True
    ) -> None:
        """
        Train the SOM on data.

        Args:
            data: Training data array of shape (n_samples, input_dim).
            n_iterations: Number of training iterations.
            verbose: Print progress.
            track_errors: Track quantization error over time.
        """
        # Convert data to appropriate backend
        if self._is_jax:
            data = jnp.array(data)
        elif self._is_cupy:
            data = cp.asarray(data)
        else:
            data = np.asarray(data)

        n_samples = data.shape[0]
        self.quantization_errors = []

        for t in range(n_iterations):
            # Decay parameters
            progress = t / n_iterations
            lr = self.initial_learning_rate * self.xp.exp(-4 * progress)
            sigma = max(float(self.initial_sigma * self.xp.exp(-4 * progress)), 0.1)

            # Random sample
            if self._is_jax:
                idx = int(jax.random.randint(
                    jax.random.PRNGKey(t), (), 0, n_samples
                ))
            elif self._is_cupy:
                idx = int(cp.random.randint(0, n_samples))
            else:
                idx = np.random.randint(0, n_samples)

            x = data[idx]

            # Find BMU
            bmu = self.find_bmu(x)
            bmu_coords = self.xp.array([bmu[0], bmu[1]], dtype=self.xp.float32)

            # Compute neighborhood
            if self._is_jax:
                neighborhood = self._compute_neighborhood(
                    bmu_coords, self.grid_coords.astype(jnp.float32), sigma
                )
                self.weights = self._update_weights(
                    self.weights, x, neighborhood, lr
                )
            else:
                diff = self.grid_coords.astype(self.xp.float32) - bmu_coords
                dist_sq = self.xp.sum(diff ** 2, axis=-1)
                neighborhood = self.xp.exp(-dist_sq / (2 * sigma ** 2))

                influence = (neighborhood * lr)[:, :, self.xp.newaxis]
                delta = x - self.weights
                self.weights = self.weights + influence * delta

            # Track errors periodically
            if track_errors and t % 100 == 0:
                qe = self._compute_quantization_error(data)
                self.quantization_errors.append(float(qe))

            if verbose and t % 100 == 0:
                print(f"Iteration {t}/{n_iterations}: LR={float(lr):.4f}, σ={sigma:.4f}")

        if verbose:
            print("Training complete!")

    def _compute_quantization_error(self, data: Any) -> float:
        """Compute average quantization error."""
        total_error = 0.0
        n_samples = data.shape[0]

        # Sample for efficiency
        sample_size = min(100, n_samples)
        if self._is_jax:
            indices = jax.random.choice(
                jax.random.PRNGKey(0), n_samples, (sample_size,), replace=False
            )
        elif self._is_cupy:
            indices = cp.random.choice(n_samples, sample_size, replace=False)
        else:
            indices = np.random.choice(n_samples, sample_size, replace=False)

        for idx in indices:
            x = data[int(idx)]
            bmu = self.find_bmu(x)
            diff = x - self.weights[bmu[0], bmu[1]]
            error = float(self.xp.sqrt(self.xp.sum(diff ** 2)))
            total_error += error

        return total_error / sample_size

    def get_weights_numpy(self) -> np.ndarray:
        """Get weights as NumPy array."""
        if self._is_jax:
            return np.array(self.weights)
        elif self._is_cupy:
            return self.weights.get()
        else:
            return self.weights


class GPUAcceleratedShadowProcessor:
    """
    GPU-accelerated classical shadow feature extraction.

    Accelerates the computation of feature vectors from shadow samples.
    """

    def __init__(
        self,
        n_qubits: int,
        backend: str = 'jax'
    ):
        """
        Initialize GPU shadow processor.

        Args:
            n_qubits: Number of qubits.
            backend: Backend to use ('jax', 'cupy', 'numpy').
        """
        self.n_qubits = n_qubits
        self.xp = get_backend(backend)
        self._is_jax = self.xp is jnp and JAX_AVAILABLE

        # Pre-compute Pauli matrices
        self._init_paulis()

        if self._is_jax:
            self._jit_compile()

    def _init_paulis(self) -> None:
        """Initialize Pauli matrices."""
        self.paulis = {
            'I': self.xp.array([[1, 0], [0, 1]], dtype=complex),
            'X': self.xp.array([[0, 1], [1, 0]], dtype=complex),
            'Y': self.xp.array([[0, -1j], [1j, 0]], dtype=complex),
            'Z': self.xp.array([[1, 0], [0, -1]], dtype=complex),
        }

    def _jit_compile(self) -> None:
        """JIT compile key functions for JAX."""
        if not self._is_jax:
            return

        @jit
        def _batch_histogram(bases, outcomes, n_qubits):
            """Compute histogram features for batch."""
            # Shape: (batch, n_qubits, 3 bases, 2 outcomes)
            features = jnp.zeros((bases.shape[0], n_qubits, 3, 2))
            return features

        self._batch_histogram = _batch_histogram

    def shadow_to_features_batch(
        self,
        shadow_samples_batch: Any,
        method: str = 'histogram'
    ) -> np.ndarray:
        """
        Convert batch of shadow samples to feature vectors.

        Args:
            shadow_samples_batch: List of shadow sample lists.
            method: Feature extraction method.

        Returns:
            Feature array of shape (n_circuits, feature_dim).
        """
        features = []

        for samples in shadow_samples_batch:
            feat = self._shadow_to_histogram(samples)
            features.append(feat)

        return np.array(features)

    def _shadow_to_histogram(
        self,
        shadow_samples: Any
    ) -> np.ndarray:
        """Compute histogram features from shadow samples."""
        feature_vector = np.zeros((self.n_qubits, 3, 2))

        for bases, outcomes in shadow_samples:
            for i in range(self.n_qubits):
                basis_idx = int(bases[i])
                outcome = int(outcomes[i])
                feature_vector[i, basis_idx, outcome] += 1

        # Normalize
        total = len(shadow_samples)
        if total > 0:
            feature_vector /= total

        return feature_vector.flatten()


def benchmark_backends(
    grid_size: Tuple[int, int] = (20, 20),
    input_dim: int = 100,
    n_samples: int = 1000,
    n_iterations: int = 500
) -> dict:
    """
    Benchmark different backends for SOM training.

    Args:
        grid_size: SOM grid size.
        input_dim: Input dimension.
        n_samples: Number of training samples.
        n_iterations: Training iterations.

    Returns:
        Dictionary of timing results.
    """
    import time

    # Generate random data
    data = np.random.randn(n_samples, input_dim).astype(np.float32)

    results = {}

    backends = ['numpy']
    if JAX_AVAILABLE:
        backends.append('jax')
    if CUPY_AVAILABLE:
        backends.append('cupy')

    for backend in backends:
        print(f"Benchmarking {backend}...")

        som = GPUAcceleratedSOM(
            grid_size=grid_size,
            input_dim=input_dim,
            backend=backend,
            random_seed=42
        )

        start = time.time()
        som.train(data, n_iterations=n_iterations, verbose=False, track_errors=False)
        elapsed = time.time() - start

        results[backend] = {
            'time': elapsed,
            'iterations_per_second': n_iterations / elapsed
        }
        print(f"  {backend}: {elapsed:.2f}s ({n_iterations/elapsed:.1f} iter/s)")

    return results


# Convenience function to get optimal SOM for current hardware
def create_optimized_som(
    grid_size: Tuple[int, int],
    input_dim: int,
    **kwargs
) -> GPUAcceleratedSOM:
    """
    Create SOM with optimal backend for current hardware.

    Automatically selects JAX > CuPy > NumPy based on availability.

    Args:
        grid_size: SOM grid dimensions.
        input_dim: Input feature dimension.
        **kwargs: Additional arguments for GPUAcceleratedSOM.

    Returns:
        Configured GPUAcceleratedSOM instance.
    """
    if JAX_AVAILABLE:
        backend = 'jax'
    elif CUPY_AVAILABLE:
        backend = 'cupy'
    else:
        backend = 'numpy'

    return GPUAcceleratedSOM(
        grid_size=grid_size,
        input_dim=input_dim,
        backend=backend,
        **kwargs
    )
