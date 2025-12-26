import numpy as np
from qsom.som import QuantumSOM

def test_som_init():
    som = QuantumSOM(grid_size=(5, 5), input_dim=10)
    assert som.weights.shape == (5, 5, 10)

def test_find_bmu():
    som = QuantumSOM(grid_size=(5, 5), input_dim=3)
    # Mock weights to predictable values
    som.weights.fill(0.0)
    som.weights[2, 2] = np.array([1.0, 1.0, 1.0])
    
    x = np.array([1.0, 1.0, 1.0])
    bmu = som._find_bmu(x)
    assert bmu == (2, 2)

def test_training_update():
    som = QuantumSOM(grid_size=(3, 3), input_dim=2, n_iterations=1)
    data = np.random.rand(5, 2)
    
    initial_weights = som.weights.copy()
    som.train(data)
    
    # Weights should have changed
    assert not np.array_equal(initial_weights, som.weights)
