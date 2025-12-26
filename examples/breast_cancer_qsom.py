"""
Example: QSOM on Breast Cancer Dataset (10 Qubits).

This script:
1. Loads Breast Cancer dataset.
2. Selects top 10 features using SelectKBest.
3. Encodes into 10-qubit states.
4. Generates classical shadows.
5. Trains QSOM.
"""

from typing import List
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import MinMaxScaler
from qiskit import QuantumCircuit

from qsom.backend import QuantumBackend
from qsom.shadows import ClassicalShadow
from qsom.som import QuantumSOM

def encode_data(data: np.ndarray) -> List[QuantumCircuit]:
    """
    Encode data into quantum circuits using Angle Encoding.
    Each feature maps to one qubit rotation Ry.
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
    print("Loading Breast Cancer dataset...")
    data_obj = load_breast_cancer()
    X = data_obj.data
    y = data_obj.target
    feature_names = data_obj.feature_names
    
    print(f"Original shape: {X.shape}")
    
    # Select top 10 features
    print("Selecting top 10 features...")
    selector = SelectKBest(f_classif, k=10)
    X_selected = selector.fit_transform(X, y)
    selected_indices = selector.get_support(indices=True)
    print(f"Selected features: {feature_names[selected_indices]}")
    
    # Shuffle and subset for demo speed
    n_samples = 30 # Demo subset size
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    indices = indices[:n_samples]
    
    X_subset = X_selected[indices]
    y_subset = y[indices]
    
    # Normalize to [0, pi]
    scaler = MinMaxScaler(feature_range=(0, np.pi))
    X_scaled = scaler.fit_transform(X_subset)
    
    n_qubits = 10
    print(f"Encoding {n_samples} samples into {n_qubits}-qubit circuits...")
    circuits = encode_data(X_scaled)
    
    # Initialize Backend
    backend = QuantumBackend(use_simulator=True, shots=1)
    
    print("Generating Classical Shadows (this may take a moment)...")
    shadow_gen = ClassicalShadow(n_qubits=n_qubits, backend=backend, shadow_size=20)
    
    shadow_features = []
    for i, qc in enumerate(circuits):
        if i % 5 == 0:
            print(f"  Processed {i}/{len(circuits)}")
        
        shadow_samples = shadow_gen.generate(qc)
        feat_vec = shadow_gen.shadow_to_feature_vector(shadow_samples)
        shadow_features.append(feat_vec)
        
    shadow_features = np.array(shadow_features)
    print(f"Shadow feature vector shape: {shadow_features.shape}")
    
    print("Training Quantum SOM...")
    som = QuantumSOM(grid_size=(10, 10), input_dim=shadow_features.shape[1], n_iterations=1000)
    som.train(shadow_features)
    
    print("Visualizing results...")
    # Plot U-matrix and clusters
    plt.figure(figsize=(12, 5))
    
    # Subplot 1: U-Matrix
    plt.subplot(1, 2, 1)
    u_mat = som.get_u_matrix()
    plt.imshow(u_mat, cmap='viridis', interpolation='nearest')
    plt.colorbar(label='Avg Distance')
    plt.title("U-Matrix")
    
    # Subplot 2: Clusters
    plt.subplot(1, 2, 2)
    plt.imshow(u_mat, cmap='gray_r', alpha=0.3)
    
    # Scatter plot
    # 0 = Malignant, 1 = Benign
    colors = ['r', 'b'] # Red for Malignant, Blue for Benign
    labels = ['Malignant', 'Benign']
    
    for cls in [0, 1]:
        mask = (y_subset == cls)
        subset_feats = shadow_features[mask]
        
        # Find BMUs
        rows, cols = [], []
        for feat in subset_feats:
            bmu = som._find_bmu(feat)
            rows.append(bmu[0])
            cols.append(bmu[1])
            
        plt.scatter(cols, rows, c=colors[cls], label=labels[cls], s=50, alpha=0.8)
        
    plt.legend()
    plt.title("Breast Cancer Classification Clusters")
    
    plt.tight_layout()
    plt.savefig("breast_cancer_som.png")
    print("Saved visualization to 'breast_cancer_som.png'")

if __name__ == "__main__":
    import sys
    sys.path.append("src")
    main()
