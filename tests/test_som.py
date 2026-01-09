"""Tests for the QuantumSOM module."""

import sys
from pathlib import Path

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest
import numpy as np
from qsom.som import QuantumSOM


class TestQuantumSOM:
    """Tests for QuantumSOM class."""

    def test_som_init_with_input_dim(self):
        """Test SOM initialization with input dimension."""
        som = QuantumSOM(grid_size=(5, 5), input_dim=10)
        assert som.weights.shape == (5, 5, 10)
        assert som.height == 5
        assert som.width == 5

    def test_som_init_without_input_dim(self):
        """Test SOM initialization without input dimension."""
        som = QuantumSOM(grid_size=(5, 5))
        assert som.weights is None

    def test_som_init_with_seed(self):
        """Test reproducibility with random seed."""
        som1 = QuantumSOM(grid_size=(3, 3), input_dim=5, random_seed=42)
        som2 = QuantumSOM(grid_size=(3, 3), input_dim=5, random_seed=42)
        assert np.array_equal(som1.weights, som2.weights)

    def test_find_bmu(self):
        """Test best matching unit finding."""
        som = QuantumSOM(grid_size=(5, 5), input_dim=3)
        som.weights.fill(0.0)
        som.weights[2, 2] = np.array([1.0, 1.0, 1.0])

        x = np.array([1.0, 1.0, 1.0])
        bmu = som._find_bmu(x)
        assert bmu == (2, 2)

    def test_training_updates_weights(self):
        """Test that training updates weights."""
        som = QuantumSOM(grid_size=(3, 3), input_dim=2, n_iterations=10)
        data = np.random.rand(5, 2)

        initial_weights = som.weights.copy()
        som.train(data, verbose=False)

        assert not np.array_equal(initial_weights, som.weights)

    def test_training_with_auto_init(self):
        """Test training with automatic weight initialization."""
        som = QuantumSOM(grid_size=(3, 3), n_iterations=10)
        data = np.random.rand(10, 4)

        som.train(data, verbose=False)

        assert som.weights is not None
        assert som.weights.shape == (3, 3, 4)

    def test_predict(self):
        """Test prediction returns valid grid coordinates."""
        som = QuantumSOM(grid_size=(5, 5), input_dim=3)
        x = np.random.rand(3)

        bmu = som.predict(x)

        assert 0 <= bmu[0] < 5
        assert 0 <= bmu[1] < 5

    def test_predict_batch(self):
        """Test batch prediction."""
        som = QuantumSOM(grid_size=(5, 5), input_dim=3)
        data = np.random.rand(10, 3)

        predictions = som.predict_batch(data)

        assert predictions.shape == (10, 2)
        assert np.all(predictions >= 0)
        assert np.all(predictions < 5)

    def test_get_umatrix(self):
        """Test U-matrix calculation."""
        som = QuantumSOM(grid_size=(4, 4), input_dim=3)
        umatrix = som.get_umatrix()

        assert umatrix.shape == (4, 4)
        assert np.all(umatrix >= 0)

    def test_get_hit_map(self):
        """Test hit map calculation."""
        som = QuantumSOM(grid_size=(4, 4), input_dim=3)
        data = np.random.rand(20, 3)

        hit_map = som.get_hit_map(data)

        assert hit_map.shape == (4, 4)
        assert hit_map.sum() == 20  # Total hits equals number of samples

    def test_distance_metrics(self):
        """Test different distance metrics."""
        x = np.array([1.0, 0.0, 0.0])
        y = np.array([0.0, 1.0, 0.0])

        # Euclidean
        som_euc = QuantumSOM(grid_size=(3, 3), input_dim=3, distance_metric='euclidean')
        dist_euc = som_euc._distance(x, y)
        assert dist_euc == pytest.approx(np.sqrt(2), rel=1e-5)

        # Fidelity-based
        som_fid = QuantumSOM(grid_size=(3, 3), input_dim=3, distance_metric='fidelity')
        dist_fid = som_fid._distance(x, y)
        assert 0 <= dist_fid <= 2

        # Quantum (combined)
        som_q = QuantumSOM(grid_size=(3, 3), input_dim=3, distance_metric='quantum')
        dist_q = som_q._distance(x, y)
        assert dist_q > 0

    def test_adaptive_learning_rate(self):
        """Test adaptive learning rate decreases over iterations."""
        som = QuantumSOM(grid_size=(3, 3), input_dim=3, learning_rate=0.5)

        lr_start = som._adaptive_learning_rate(0, 100)
        lr_mid = som._adaptive_learning_rate(50, 100)
        lr_end = som._adaptive_learning_rate(99, 100)

        assert lr_start > lr_mid > lr_end
        assert lr_start == pytest.approx(0.5, rel=1e-5)

    def test_training_history(self):
        """Test that training history is recorded."""
        som = QuantumSOM(grid_size=(3, 3), input_dim=2, n_iterations=200)
        data = np.random.rand(10, 2)

        som.train(data, verbose=False, track_errors=True)

        assert len(som.training_history) > 0
        assert 'iteration' in som.training_history[0]
        assert 'quantization_error' in som.training_history[0]

    def test_initialize_weights_pca(self):
        """Test PCA-based weight initialization."""
        som = QuantumSOM(grid_size=(4, 4))
        data = np.random.rand(50, 5)

        som.initialize_weights(data, method='pca')

        assert som.weights is not None
        assert som.weights.shape == (4, 4, 5)

    def test_initialize_weights_random_sample(self):
        """Test random sample weight initialization."""
        som = QuantumSOM(grid_size=(4, 4))
        data = np.random.rand(50, 5)

        som.initialize_weights(data, method='random_sample')

        assert som.weights is not None
        assert som.weights.shape == (4, 4, 5)
