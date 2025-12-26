"""
Example: QSOM on Iris Dataset.

This script demonstrates how to:
1. Load classical data (Iris).
2. Encode it into quantum states (Angle Encoding).
3. Generate classical shadows from these states.
4. Train a Quantum SOM on the shadow features.
"""

from typing import List
import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit

from qsom.backend import QuantumBackend
from qsom.shadows import ClassicalShadow
from qsom.som import QuantumSOM

def encode_data(data: np.ndarray) -> List[QuantumCircuit]:
    """
    Encode Angle Encoding: normalize to [0, pi], apply Ry(theta).
    One qubit per feature.
    """
    circuits = []
    n_features = data.shape[1]
    
    for sample in data:
        qc = QuantumCircuit(n_features)
        for i, val in enumerate(sample):
            qc.ry(val, i)
        circuits.append(qc)
    return circuits

def main():
    print("Loading Iris dataset...")
    iris = load_iris()
    X = iris.data
    y = iris.target
    
    # Normalize to [0, pi] for angle encoding
    # Use subset for faster demo
    # Shuffle and select subset for faster demo
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    indices = indices[:30]
    X = X[indices]
    y = y[indices]
    scaler = MinMaxScaler(feature_range=(0, np.pi))
    X_scaled = scaler.fit_transform(X)
    
    print("Encoding data into quantum circuits...")
    circuits = encode_data(X_scaled)
    n_qubits = X.shape[1]
    
    print("Initializing Quantum Backend (Simulator)...")
    # Using simulator for example, but capable of real hardware
    backend = QuantumBackend(use_simulator=True, shots=1) 
    
    print("Generating Classical Shadows...")
    # Generate shadow for each data point
    # Shadow size = 50 snapshots per data point
    shadow_gen = ClassicalShadow(n_qubits=n_qubits, backend=backend, shadow_size=20)
    
    shadow_features = []
    for i, qc in enumerate(circuits):
        if i % 10 == 0:
            print(f"Processing sample {i}/{len(circuits)}")
        
        shadow_samples = shadow_gen.generate(qc)
        feat_vec = shadow_gen.shadow_to_feature_vector(shadow_samples)
        shadow_features.append(feat_vec)
        
    shadow_features = np.array(shadow_features)
    print(f"Feature vector shape: {shadow_features.shape}")
    
    print("Training Quantum SOM...")
    som = QuantumSOM(grid_size=(10, 10), input_dim=shadow_features.shape[1], n_iterations=1000)
    som.train(shadow_features)
    
    print("Visualizing results...")
    som.visualize("iris_som_umatrix.png")
    
    # Optional: Plot winning neurons with labels
    plt.figure(figsize=(10, 10))
    # U-matrix background
    u_mat = som.get_u_matrix()
    plt.imshow(u_mat, cmap='viridis', alpha=0.5)
    
    # Scatter plot
    colors = ['r', 'g', 'b']
    for idx, x in enumerate(shadow_features):
        bmu = som._find_bmu(x)
        plt.scatter(bmu[1], bmu[0], c=colors[y[idx]], s=20)
        
    plt.title("QSOM on Iris (Shadow Features)")
    plt.savefig("iris_som_clusters.png")
    print("Done! Saved visualizations.")

if __name__ == "__main__":
    from typing import List
    import sys
    # Ensure src is in path if running from root
    sys.path.append("src")
    main()
