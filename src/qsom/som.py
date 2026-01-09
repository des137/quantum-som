"""
Quantum Self-Organizing Map (QSOM) Core Logic.

This module implements the SOM algorithm adapted for quantum data features,
with support for quantum-specific distance metrics and visualization.

Features:
- Multiple distance metrics (Euclidean, fidelity, Bures, trace)
- Adaptive learning rate scheduling
- U-matrix visualization
- Training history tracking
- Data projection and clustering
"""

from typing import Tuple, List, Optional, Dict, Any, Union
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import sqrtm


class QuantumSOM:
    """
    Self-Organizing Map for Quantum Feature Vectors.

    Implements the Kohonen SOM algorithm with extensions for
    quantum state space exploration using classical shadows.

    Examples:
        >>> from qsom import QuantumSOM
        >>> som = QuantumSOM(grid_size=(10, 10), input_dim=36)
        >>> som.train(shadow_features, n_iterations=1000)
        >>> som.visualize(shadow_features, labels=labels)
    """

    def __init__(
        self,
        grid_size: Tuple[int, int] = (10, 10),
        input_dim: Optional[int] = None,
        learning_rate: float = 0.5,
        sigma: float = 1.0,
        n_iterations: int = 1000,
        distance_metric: str = 'euclidean',
        random_seed: Optional[int] = None
    ):
        """
        Initialize the Quantum SOM.

        Args:
            grid_size: Tuple (height, width) of the 2D map.
            input_dim: Dimension of input feature vectors. If None, inferred from data.
            learning_rate: Initial learning rate (α₀).
            sigma: Initial neighborhood radius (σ₀).
            n_iterations: Default number of training iterations.
            distance_metric: Distance metric to use:
                - 'euclidean': Standard Euclidean distance
                - 'fidelity': Quantum fidelity-based distance
                - 'quantum': Combined metric (70% fidelity + 30% Euclidean)
                - 'bures': Bures distance for density matrices
                - 'trace': Trace distance for density matrices
            random_seed: Random seed for reproducibility.
        """
        self.height, self.width = grid_size
        self.grid_size = grid_size
        self.input_dim = input_dim
        self.learning_rate = learning_rate
        self.sigma = sigma
        self.n_iterations = n_iterations
        self.distance_metric = distance_metric

        if random_seed is not None:
            np.random.seed(random_seed)

        # Initialize weights if input_dim is known
        if input_dim is not None:
            self.weights = np.random.randn(self.height, self.width, input_dim)
        else:
            self.weights = None

        # Precompute coordinate grid for efficiency
        self._grid_coords = np.zeros((self.height, self.width, 2))
        for i in range(self.height):
            for j in range(self.width):
                self._grid_coords[i, j] = [i, j]

        # Training history
        self.training_history: List[Dict[str, Any]] = []
        self.quantization_errors: List[float] = []
        self.topographic_errors: List[float] = []

    def _quantum_fidelity_distance(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        Calculate quantum fidelity-based distance.

        Uses cosine similarity as a proxy for fidelity between shadow vectors.

        Args:
            x: First feature vector.
            y: Second feature vector.

        Returns:
            Distance in range [0, 2].
        """
        norm_x = np.linalg.norm(x)
        norm_y = np.linalg.norm(y)

        if norm_x == 0 or norm_y == 0:
            return 2.0  # Maximum distance

        similarity = np.dot(x, y) / (norm_x * norm_y)
        # Convert similarity [-1, 1] to distance [0, 2]
        return 1.0 - similarity

    def _bures_distance(self, rho1: np.ndarray, rho2: np.ndarray) -> float:
        """
        Calculate Bures distance between density matrices.

        Bures distance = sqrt(2 - 2 * sqrt(F(ρ₁, ρ₂)))
        where F is the Uhlmann fidelity.

        Args:
            rho1: First density matrix.
            rho2: Second density matrix.

        Returns:
            Bures distance.
        """
        try:
            sqrt_rho1 = sqrtm(rho1)
            product = sqrt_rho1 @ rho2 @ sqrt_rho1
            fidelity = np.real(np.trace(sqrtm(product)))
            fidelity = np.clip(fidelity, 0, 1)
            return np.sqrt(2 - 2 * np.sqrt(fidelity))
        except Exception:
            return np.linalg.norm(rho1 - rho2)

    def _trace_distance(self, rho1: np.ndarray, rho2: np.ndarray) -> float:
        """
        Calculate trace distance between density matrices.

        Trace distance = 0.5 * Tr|ρ₁ - ρ₂|

        Args:
            rho1: First density matrix.
            rho2: Second density matrix.

        Returns:
            Trace distance.
        """
        try:
            diff = rho1 - rho2
            eigenvalues = np.linalg.eigvalsh(diff)
            return 0.5 * np.sum(np.abs(eigenvalues))
        except Exception:
            return np.linalg.norm(rho1 - rho2)

    def _distance(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        Calculate distance between two vectors using configured metric.

        Args:
            x: First vector.
            y: Second vector.

        Returns:
            Distance value.
        """
        if self.distance_metric == 'fidelity':
            return self._quantum_fidelity_distance(x, y)

        elif self.distance_metric == 'quantum':
            # Combined metric
            fidelity_dist = self._quantum_fidelity_distance(x, y)
            euclidean_dist = np.linalg.norm(x - y)
            return 0.7 * fidelity_dist + 0.3 * euclidean_dist

        elif self.distance_metric == 'bures':
            dim = int(np.sqrt(len(x)))
            if dim * dim == len(x):
                rho1 = x.reshape(dim, dim)
                rho2 = y.reshape(dim, dim)
                return self._bures_distance(rho1, rho2)
            return np.linalg.norm(x - y)

        elif self.distance_metric == 'trace':
            dim = int(np.sqrt(len(x)))
            if dim * dim == len(x):
                rho1 = x.reshape(dim, dim)
                rho2 = y.reshape(dim, dim)
                return self._trace_distance(rho1, rho2)
            return np.linalg.norm(x - y)

        else:  # euclidean (default)
            return np.linalg.norm(x - y)

    def _find_bmu(self, x: np.ndarray) -> Tuple[int, int]:
        """
        Find the Best Matching Unit (BMU) for input vector x.

        Args:
            x: Input feature vector.

        Returns:
            Grid coordinates (i, j) of the BMU.
        """
        if self.distance_metric == 'euclidean':
            # Optimized for Euclidean distance
            delta = self.weights - x
            dist_sq = np.sum(delta ** 2, axis=2)
            bmu_idx = np.unravel_index(np.argmin(dist_sq), (self.height, self.width))
        else:
            # General case for other metrics
            min_dist = float('inf')
            bmu_idx = (0, 0)
            for i in range(self.height):
                for j in range(self.width):
                    dist = self._distance(x, self.weights[i, j])
                    if dist < min_dist:
                        min_dist = dist
                        bmu_idx = (i, j)

        return bmu_idx

    def _neighborhood_function(self, distance: float, sigma: float) -> float:
        """
        Gaussian neighborhood function.

        Args:
            distance: Grid distance from BMU.
            sigma: Current neighborhood radius.

        Returns:
            Neighborhood influence value in [0, 1].
        """
        return np.exp(-(distance ** 2) / (2 * sigma ** 2))

    def _update_weights(
        self,
        x: np.ndarray,
        bmu_idx: Tuple[int, int],
        learning_rate: float,
        sigma: float
    ) -> None:
        """
        Update weights of all neurons based on BMU and input.

        Args:
            x: Input vector.
            bmu_idx: Grid position of BMU.
            learning_rate: Current learning rate.
            sigma: Current neighborhood radius.
        """
        # Calculate grid distances from BMU
        grid_dist_sq = np.sum((self._grid_coords - bmu_idx) ** 2, axis=2)
        neighborhood = np.exp(-grid_dist_sq / (2 * sigma ** 2))

        # Update weights
        influence = neighborhood[:, :, np.newaxis] * learning_rate
        self.weights += influence * (x - self.weights)

    def _adaptive_learning_rate(
        self,
        iteration: int,
        n_iterations: int,
        schedule: str = 'exponential'
    ) -> float:
        """
        Calculate adaptive learning rate.

        Args:
            iteration: Current iteration.
            n_iterations: Total iterations.
            schedule: Learning rate schedule ('linear', 'exponential', 'plateau').

        Returns:
            Current learning rate.
        """
        progress = iteration / n_iterations

        if schedule == 'linear':
            return self.learning_rate * (1 - progress)

        elif schedule == 'exponential':
            return self.learning_rate * np.exp(-4 * progress)

        elif schedule == 'plateau':
            # Plateau for first 10%, then exponential decay
            if progress < 0.1:
                return self.learning_rate
            else:
                adjusted_progress = (progress - 0.1) / 0.9
                return self.learning_rate * (0.95 ** (adjusted_progress * 100))

        return self.learning_rate * (1 - progress)

    def _adaptive_sigma(self, iteration: int, n_iterations: int) -> float:
        """
        Calculate adaptive neighborhood radius.

        Args:
            iteration: Current iteration.
            n_iterations: Total iterations.

        Returns:
            Current sigma value (minimum 0.1).
        """
        progress = iteration / n_iterations
        current_sigma = self.sigma * np.exp(-4 * progress)
        return max(current_sigma, 0.1)

    def initialize_weights(
        self,
        data: np.ndarray,
        method: str = 'random_sample'
    ) -> None:
        """
        Initialize weight vectors.

        Args:
            data: Training data of shape (n_samples, n_features).
            method: Initialization method:
                - 'random': Random normal distribution
                - 'random_sample': Sample from training data
                - 'pca': PCA-based initialization
        """
        if self.input_dim is None:
            self.input_dim = data.shape[1]

        n_neurons = self.height * self.width

        if method == 'random_sample':
            indices = np.random.choice(len(data), size=n_neurons, replace=True)
            samples = data[indices]
            self.weights = samples.reshape(self.height, self.width, self.input_dim)

        elif method == 'pca':
            # PCA-based initialization
            mean = np.mean(data, axis=0)
            centered = data - mean
            cov = np.cov(centered.T)
            eigenvalues, eigenvectors = np.linalg.eigh(cov)

            # Use top 2 principal components
            idx = np.argsort(eigenvalues)[::-1]
            pc1 = eigenvectors[:, idx[0]]
            pc2 = eigenvectors[:, idx[1]]

            # Create grid
            self.weights = np.zeros((self.height, self.width, self.input_dim))
            for i in range(self.height):
                for j in range(self.width):
                    self.weights[i, j] = mean + \
                        (i / self.height - 0.5) * 2 * np.sqrt(eigenvalues[idx[0]]) * pc1 + \
                        (j / self.width - 0.5) * 2 * np.sqrt(eigenvalues[idx[1]]) * pc2

        else:  # random
            self.weights = np.random.randn(self.height, self.width, self.input_dim)

    def train(
        self,
        data: np.ndarray,
        n_iterations: Optional[int] = None,
        learning_schedule: str = 'exponential',
        verbose: bool = True,
        track_errors: bool = True
    ) -> None:
        """
        Train the SOM on the provided data.

        Args:
            data: Training data of shape (n_samples, n_features).
            n_iterations: Number of training iterations (default: self.n_iterations).
            learning_schedule: Learning rate decay schedule.
            verbose: Whether to print training progress.
            track_errors: Whether to track quantization errors during training.
        """
        if n_iterations is None:
            n_iterations = self.n_iterations

        if self.weights is None:
            self.initialize_weights(data)

        n_samples = len(data)
        self.training_history = []
        self.quantization_errors = []

        for t in range(n_iterations):
            # Adaptive parameters
            current_lr = self._adaptive_learning_rate(t, n_iterations, learning_schedule)
            current_sigma = self._adaptive_sigma(t, n_iterations)

            # Random sample selection
            idx = np.random.randint(0, n_samples)
            x = data[idx]

            # Find BMU and update weights
            bmu = self._find_bmu(x)
            self._update_weights(x, bmu, current_lr, current_sigma)

            # Track progress
            if track_errors and t % 100 == 0:
                qe = self._quantization_error(data)
                self.quantization_errors.append(qe)
                self.training_history.append({
                    'iteration': t,
                    'quantization_error': qe,
                    'learning_rate': current_lr,
                    'sigma': current_sigma
                })

                if verbose:
                    print(f"Iteration {t:5d}: QE={qe:.4f}, LR={current_lr:.4f}, σ={current_sigma:.4f}")

    def _quantization_error(self, data: np.ndarray) -> float:
        """
        Calculate average quantization error.

        Args:
            data: Data to evaluate.

        Returns:
            Mean distance from samples to their BMUs.
        """
        total_error = 0.0
        for x in data:
            bmu = self._find_bmu(x)
            total_error += self._distance(x, self.weights[bmu[0], bmu[1]])
        return total_error / len(data)

    def _topographic_error(self, data: np.ndarray) -> float:
        """
        Calculate topographic error.

        Measures the proportion of samples for which the first and second BMU
        are not adjacent on the grid.

        Args:
            data: Data to evaluate.

        Returns:
            Topographic error in [0, 1].
        """
        errors = 0
        for x in data:
            # Find two best matching units
            distances = np.zeros((self.height, self.width))
            for i in range(self.height):
                for j in range(self.width):
                    distances[i, j] = self._distance(x, self.weights[i, j])

            flat_idx = np.argsort(distances.flatten())[:2]
            bmu1 = np.unravel_index(flat_idx[0], (self.height, self.width))
            bmu2 = np.unravel_index(flat_idx[1], (self.height, self.width))

            # Check if adjacent
            if abs(bmu1[0] - bmu2[0]) + abs(bmu1[1] - bmu2[1]) > 1:
                errors += 1

        return errors / len(data)

    def predict(self, x: np.ndarray) -> Tuple[int, int]:
        """
        Find the BMU for a single sample.

        Args:
            x: Input feature vector.

        Returns:
            Grid coordinates of the BMU.
        """
        return self._find_bmu(x)

    def predict_batch(self, data: np.ndarray) -> np.ndarray:
        """
        Find BMUs for multiple samples.

        Args:
            data: Array of shape (n_samples, n_features).

        Returns:
            Array of shape (n_samples, 2) with grid coordinates.
        """
        predictions = []
        for x in data:
            predictions.append(self._find_bmu(x))
        return np.array(predictions)

    def get_umatrix(self) -> np.ndarray:
        """
        Calculate the Unified Distance Matrix (U-Matrix).

        The U-matrix shows the average distance between each neuron
        and its neighbors, revealing cluster boundaries.

        Returns:
            U-matrix of shape (height, width).
        """
        umatrix = np.zeros((self.height, self.width))

        for i in range(self.height):
            for j in range(self.width):
                distances = []

                # 8-connectivity (including diagonals)
                for di in [-1, 0, 1]:
                    for dj in [-1, 0, 1]:
                        if di == 0 and dj == 0:
                            continue
                        ni, nj = i + di, j + dj
                        if 0 <= ni < self.height and 0 <= nj < self.width:
                            dist = self._distance(
                                self.weights[i, j],
                                self.weights[ni, nj]
                            )
                            distances.append(dist)

                umatrix[i, j] = np.mean(distances) if distances else 0

        return umatrix

    def get_hit_map(self, data: np.ndarray) -> np.ndarray:
        """
        Calculate hit map (frequency of BMU activations).

        Args:
            data: Input data.

        Returns:
            Hit map of shape (height, width).
        """
        hit_map = np.zeros((self.height, self.width))
        for x in data:
            bmu = self._find_bmu(x)
            hit_map[bmu[0], bmu[1]] += 1
        return hit_map

    def get_component_planes(self) -> np.ndarray:
        """
        Get component planes (weight visualization per dimension).

        Returns:
            Array of shape (input_dim, height, width).
        """
        return self.weights.transpose(2, 0, 1)

    def visualize(
        self,
        data: Optional[np.ndarray] = None,
        labels: Optional[np.ndarray] = None,
        title: str = "Quantum SOM Visualization",
        save_path: Optional[str] = None,
        show: bool = True
    ) -> plt.Figure:
        """
        Visualize the trained SOM.

        Creates a figure with U-matrix and hit map or data projection.

        Args:
            data: Optional data to project onto the map.
            labels: Optional labels for coloring data points.
            title: Figure title.
            save_path: Path to save the figure.
            show: Whether to display the figure.

        Returns:
            Matplotlib Figure object.
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # U-matrix
        umatrix = self.get_umatrix()
        im1 = axes[0].imshow(umatrix, cmap='viridis', origin='lower')
        axes[0].set_title('U-Matrix (Distance Map)')
        axes[0].set_xlabel('Grid X')
        axes[0].set_ylabel('Grid Y')
        plt.colorbar(im1, ax=axes[0], label='Avg Distance')

        # Hit map or data projection
        if data is not None:
            projections = self.predict_batch(data)
            hit_map = self.get_hit_map(data)

            im2 = axes[1].imshow(hit_map, cmap='hot', origin='lower', alpha=0.7)

            if labels is not None:
                unique_labels = np.unique(labels)
                colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))

                for label, color in zip(unique_labels, colors):
                    mask = labels == label
                    proj_label = projections[mask]
                    # Add jitter for visualization
                    jitter = np.random.uniform(-0.3, 0.3, proj_label.shape)
                    axes[1].scatter(
                        proj_label[:, 1] + jitter[:, 1],
                        proj_label[:, 0] + jitter[:, 0],
                        c=[color], label=f'Class {label}',
                        alpha=0.7, s=40, edgecolors='white', linewidth=0.5
                    )
                axes[1].legend(loc='upper right', fontsize=8)

            axes[1].set_title('Data Hit Map')
            plt.colorbar(im2, ax=axes[1], label='Hit Count')
        else:
            # Component plane for first dimension
            component = self.weights[:, :, 0]
            im2 = axes[1].imshow(component, cmap='coolwarm', origin='lower')
            axes[1].set_title('Component Plane (Dim 0)')
            plt.colorbar(im2, ax=axes[1])

        axes[1].set_xlabel('Grid X')
        axes[1].set_ylabel('Grid Y')

        plt.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')

        if show:
            plt.show()

        return fig

    def visualize_training(
        self,
        save_path: Optional[str] = None,
        show: bool = True
    ) -> plt.Figure:
        """
        Visualize training history.

        Args:
            save_path: Path to save the figure.
            show: Whether to display the figure.

        Returns:
            Matplotlib Figure object.
        """
        if not self.training_history:
            print("No training history available. Run train() first.")
            return None

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        iterations = [h['iteration'] for h in self.training_history]
        qe = [h['quantization_error'] for h in self.training_history]
        lr = [h['learning_rate'] for h in self.training_history]
        sigma = [h['sigma'] for h in self.training_history]

        axes[0].plot(iterations, qe, 'b-', linewidth=2)
        axes[0].set_xlabel('Iteration')
        axes[0].set_ylabel('Quantization Error')
        axes[0].set_title('Training Convergence')
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(iterations, lr, 'g-', linewidth=2)
        axes[1].set_xlabel('Iteration')
        axes[1].set_ylabel('Learning Rate')
        axes[1].set_title('Learning Rate Decay')
        axes[1].grid(True, alpha=0.3)

        axes[2].plot(iterations, sigma, 'r-', linewidth=2)
        axes[2].set_xlabel('Iteration')
        axes[2].set_ylabel('Sigma (σ)')
        axes[2].set_title('Neighborhood Radius Decay')
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')

        if show:
            plt.show()

        return fig

    # ==================== Checkpointing Methods ====================

    def save_checkpoint(self, filepath: Union[str, Path]) -> None:
        """
        Save model checkpoint to file.

        Args:
            filepath: Path to save the checkpoint.
        """
        filepath = Path(filepath)

        checkpoint = {
            'weights': self.weights.tolist() if self.weights is not None else None,
            'grid_size': self.grid_size,
            'input_dim': self.input_dim,
            'learning_rate': self.learning_rate,
            'sigma': self.sigma,
            'n_iterations': self.n_iterations,
            'distance_metric': self.distance_metric,
            'training_history': self.training_history,
            'quantization_errors': self.quantization_errors,
        }

        # Save weights as numpy file, metadata as JSON
        np.save(filepath.with_suffix('.npy'), self.weights)

        with open(filepath.with_suffix('.json'), 'w') as f:
            # Remove weights from JSON (stored separately)
            checkpoint_meta = {k: v for k, v in checkpoint.items() if k != 'weights'}
            json.dump(checkpoint_meta, f, indent=2)

    @classmethod
    def load_checkpoint(cls, filepath: Union[str, Path]) -> 'QuantumSOM':
        """
        Load model from checkpoint.

        Args:
            filepath: Path to the checkpoint file.

        Returns:
            Loaded QuantumSOM instance.
        """
        filepath = Path(filepath)

        # Load metadata
        with open(filepath.with_suffix('.json'), 'r') as f:
            checkpoint = json.load(f)

        # Load weights
        weights = np.load(filepath.with_suffix('.npy'))

        # Create instance
        som = cls(
            grid_size=tuple(checkpoint['grid_size']),
            input_dim=checkpoint['input_dim'],
            learning_rate=checkpoint['learning_rate'],
            sigma=checkpoint['sigma'],
            n_iterations=checkpoint['n_iterations'],
            distance_metric=checkpoint['distance_metric'],
        )

        som.weights = weights
        som.training_history = checkpoint.get('training_history', [])
        som.quantization_errors = checkpoint.get('quantization_errors', [])

        return som

    # ==================== Mini-Batch Training ====================

    def train_minibatch(
        self,
        data: np.ndarray,
        n_epochs: int = 10,
        batch_size: int = 32,
        learning_schedule: str = 'exponential',
        verbose: bool = True,
        track_errors: bool = True,
        checkpoint_dir: Optional[Union[str, Path]] = None,
        checkpoint_freq: int = 5
    ) -> None:
        """
        Train the SOM using mini-batch gradient descent.

        This is more efficient for large datasets as it processes
        samples in batches rather than one at a time.

        Args:
            data: Training data of shape (n_samples, n_features).
            n_epochs: Number of training epochs.
            batch_size: Number of samples per mini-batch.
            learning_schedule: Learning rate decay schedule.
            verbose: Whether to print training progress.
            track_errors: Whether to track quantization errors.
            checkpoint_dir: Directory to save checkpoints.
            checkpoint_freq: Save checkpoint every N epochs.
        """
        if self.weights is None:
            self.initialize_weights(data)

        n_samples = len(data)
        n_batches = max(1, n_samples // batch_size)
        total_iterations = n_epochs * n_batches

        if checkpoint_dir:
            checkpoint_dir = Path(checkpoint_dir)
            checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.training_history = []
        self.quantization_errors = []
        iteration = 0

        for epoch in range(n_epochs):
            # Shuffle data at start of each epoch
            indices = np.random.permutation(n_samples)

            epoch_loss = 0.0

            for batch_idx in range(n_batches):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, n_samples)
                batch_indices = indices[start_idx:end_idx]
                batch = data[batch_indices]

                # Compute adaptive parameters
                progress = iteration / total_iterations
                current_lr = self._adaptive_learning_rate(
                    iteration, total_iterations, learning_schedule
                )
                current_sigma = self._adaptive_sigma(iteration, total_iterations)

                # Process batch
                batch_loss = 0.0
                for x in batch:
                    bmu = self._find_bmu(x)
                    self._update_weights(x, bmu, current_lr, current_sigma)
                    batch_loss += self._distance(x, self.weights[bmu[0], bmu[1]])

                epoch_loss += batch_loss
                iteration += 1

            # Track progress at end of epoch
            if track_errors:
                qe = epoch_loss / n_samples
                self.quantization_errors.append(qe)
                self.training_history.append({
                    'epoch': epoch,
                    'iteration': iteration,
                    'quantization_error': qe,
                    'learning_rate': current_lr,
                    'sigma': current_sigma
                })

                if verbose:
                    print(f"Epoch {epoch + 1:3d}/{n_epochs}: "
                          f"QE={qe:.4f}, LR={current_lr:.4f}, σ={current_sigma:.4f}")

            # Save checkpoint
            if checkpoint_dir and (epoch + 1) % checkpoint_freq == 0:
                checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch + 1}"
                self.save_checkpoint(checkpoint_path)
                if verbose:
                    print(f"  Saved checkpoint: {checkpoint_path}")

    # ==================== Additional Distance Metrics ====================

    def _hilbert_schmidt_distance(self, rho1: np.ndarray, rho2: np.ndarray) -> float:
        """
        Calculate Hilbert-Schmidt distance between density matrices.

        HS distance = sqrt(Tr[(ρ₁ - ρ₂)²])

        Args:
            rho1: First density matrix.
            rho2: Second density matrix.

        Returns:
            Hilbert-Schmidt distance.
        """
        diff = rho1 - rho2
        return np.sqrt(np.real(np.trace(diff @ diff)))

    def _cosine_distance(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        Calculate cosine distance.

        Args:
            x: First vector.
            y: Second vector.

        Returns:
            Cosine distance in [0, 2].
        """
        norm_x = np.linalg.norm(x)
        norm_y = np.linalg.norm(y)

        if norm_x == 0 or norm_y == 0:
            return 1.0

        return 1.0 - np.dot(x, y) / (norm_x * norm_y)

    def _manhattan_distance(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        Calculate Manhattan (L1) distance.

        Args:
            x: First vector.
            y: Second vector.

        Returns:
            Manhattan distance.
        """
        return np.sum(np.abs(x - y))
