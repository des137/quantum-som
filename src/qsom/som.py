"""
Quantum Self-Organizing Map (QSOM) Core Logic.

This module implements the SOM algorithm adapted for quantum data features.
"""

from typing import Tuple, List, Optional, Callable
import numpy as np
import matplotlib.pyplot as plt

class QuantumSOM:
    """
    Self-Organizing Map for Quantum Feature Vectors.
    """

    def __init__(
        self,
        grid_size: Tuple[int, int],
        input_dim: int,
        learning_rate: float = 0.5,
        sigma: float = 1.0,
        n_iterations: int = 1000
    ):
        """
        Initialize the SOM.

        Args:
            grid_size: Tuple (height, width) of the map.
            input_dim: Dimension of the input feature vectors.
            learning_rate: Initial learning rate.
            sigma: Initial neighborhood radius.
            n_iterations: Total training iterations.
        """
        self.height, self.width = grid_size
        self.input_dim = input_dim
        self.learning_rate = learning_rate
        self.sigma = sigma
        self.n_iterations = n_iterations
        
        # Initialize weights randomly
        self.weights = np.random.rand(self.height, self.width, input_dim)
        
        # Precompute coordinate grid for efficiency
        self._grid_coords = np.zeros((self.height, self.width, 2))
        for i in range(self.height):
            for j in range(self.width):
                self._grid_coords[i, j] = [i, j]

    def _find_bmu(self, x: np.ndarray) -> Tuple[int, int]:
        """Find the Best Matching Unit for input vector x."""
        # Calculate distances to all neurons
        # Efficient broadcasting
        # weights shape: (H, W, D), x shape: (D,)
        # delta shape: (H, W, D)
        delta = self.weights - x
        dist_sq = np.sum(delta**2, axis=2)
        
        bmu_idx = np.unravel_index(np.argmin(dist_sq), (self.height, self.width))
        return bmu_idx

    def _update_weights(self, x: np.ndarray, bmu_idx: Tuple[int, int], iteration: int):
        """Update weights based on iteration and BMU."""
        # Decay parameters
        progress = iteration / self.n_iterations
        curr_lr = self.learning_rate * np.exp(-4 * progress) # Exponential decay
        curr_sigma = self.sigma * np.exp(-4 * progress)
        
        # Calculate neighborhood function
        # Distances on grid from BMU
        # grid_coords: (H, W, 2)
        # bmu_coords = (2,)
        
        grid_dist_sq = np.sum((self._grid_coords - bmu_idx)**2, axis=2)
        neighborhood = np.exp(-grid_dist_sq / (2 * curr_sigma**2))
        
        # Update weights
        # x-w: (H, W, D)
        # neighborhood: (H, W) -> unsqueeze to (H, W, 1)
        influence = neighborhood[:, :, np.newaxis] * curr_lr
        self.weights += influence * (x - self.weights)

    def train(self, data: np.ndarray):
        """
        Train the SOM on the provided data.

        Args:
            data: Numpy array of shape (N, input_dim).
        """
        n_samples = data.shape[0]
        for t in range(self.n_iterations):
            # Pick random sample
            idx = np.random.randint(0, n_samples)
            x = data[idx]
            
            bmu = self._find_bmu(x)
            self._update_weights(x, bmu, t)

    def get_u_matrix(self) -> np.ndarray:
        """
        Calculate the Unified Distance Matrix (U-Matrix).
        
        Returns:
            np.ndarray: (H, W) matrix representing average distance to neighbors.
        """
        u_matrix = np.zeros((self.height, self.width))
        for i in range(self.height):
            for j in range(self.width):
                # Calculate avg distance to neighbors
                dists = []
                # 4-neighbors
                if i > 0: dists.append(np.linalg.norm(self.weights[i,j] - self.weights[i-1,j]))
                if i < self.height - 1: dists.append(np.linalg.norm(self.weights[i,j] - self.weights[i+1,j]))
                if j > 0: dists.append(np.linalg.norm(self.weights[i,j] - self.weights[i,j-1]))
                if j < self.width - 1: dists.append(np.linalg.norm(self.weights[i,j] - self.weights[i,j+1]))
                
                u_matrix[i, j] = np.mean(dists) if dists else 0.0
        return u_matrix

    def visualize(self, save_path: str = "som_umatrix.png"):
        """Save visualization of the U-Matrix."""
        u_mat = self.get_u_matrix()
        plt.figure(figsize=(8, 6))
        plt.imshow(u_mat, cmap='viridis', interpolation='nearest')
        plt.colorbar(label='Average Distance')
        plt.title('SOM U-Matrix')
        plt.savefig(save_path)
        plt.close()
