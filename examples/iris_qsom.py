"""
Example: QSOM on Iris Dataset.

This script demonstrates how to:
1. Load classical data (Iris dataset)
2. Encode it into quantum states using angle encoding
3. Generate classical shadows from these states
4. Train a Quantum SOM on the shadow features
5. Visualize the results with cluster separation
"""

import sys
from pathlib import Path

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler
from qiskit import QuantumCircuit

from qsom import QuantumBackend, ClassicalShadow, QuantumSOM


def encode_data_angle(data: np.ndarray) -> list:
    """
    Encode classical data using angle encoding.

    Each feature is encoded as a Ry rotation angle on a separate qubit.

    Args:
        data: Normalized data in range [0, π].

    Returns:
        List of QuantumCircuit objects.
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
    print("=" * 60)
    print("QSOM Example: Iris Dataset Classification")
    print("=" * 60)

    # Load and prepare data
    print("\n1. Loading Iris dataset...")
    iris = load_iris()
    X = iris.data
    y = iris.target
    target_names = iris.target_names

    # Use subset for faster demo
    np.random.seed(42)
    indices = np.random.choice(len(X), size=50, replace=False)
    X = X[indices]
    y = y[indices]

    print(f"   Samples: {len(X)}, Features: {X.shape[1]}")

    # Normalize to [0, π] for angle encoding
    scaler = MinMaxScaler(feature_range=(0, np.pi))
    X_scaled = scaler.fit_transform(X)

    # Encode as quantum circuits
    print("\n2. Encoding data into quantum circuits...")
    circuits = encode_data_angle(X_scaled)
    n_qubits = X.shape[1]
    print(f"   Created {len(circuits)} circuits with {n_qubits} qubits each")

    # Initialize backend
    print("\n3. Initializing quantum backend (simulator)...")
    backend = QuantumBackend(use_simulator=True, shots=1)

    # Generate classical shadows
    print("\n4. Generating classical shadows...")
    shadow_gen = ClassicalShadow(
        n_qubits=n_qubits,
        backend=backend,
        shadow_size=30,  # Number of random Pauli measurements
        use_inverse_channel=True
    )

    shadow_features = []
    for i, qc in enumerate(circuits):
        if (i + 1) % 10 == 0 or i == 0:
            print(f"   Processing sample {i + 1}/{len(circuits)}")

        shadow_samples = shadow_gen.generate(qc)
        feat_vec = shadow_gen.shadow_to_feature_vector(shadow_samples)
        shadow_features.append(feat_vec)

    shadow_features = np.array(shadow_features)
    print(f"   Feature vector dimension: {shadow_features.shape[1]}")

    # Train SOM
    print("\n5. Training Quantum SOM...")
    som = QuantumSOM(
        grid_size=(8, 8),
        input_dim=shadow_features.shape[1],
        learning_rate=0.5,
        sigma=2.0,
        n_iterations=500,
        distance_metric='quantum',
        random_seed=42
    )
    som.train(shadow_features, verbose=True)

    # Visualize results
    print("\n6. Generating visualizations...")

    # Main visualization
    fig = som.visualize(
        data=shadow_features,
        labels=y,
        title="Quantum SOM: Iris Dataset Classification",
        save_path="iris_som_visualization.png",
        show=False
    )
    plt.close(fig)

    # U-matrix with labeled clusters
    plt.figure(figsize=(10, 8))
    u_mat = som.get_umatrix()
    plt.imshow(u_mat, cmap='viridis', alpha=0.7, origin='lower')
    plt.colorbar(label='Average Distance')

    # Plot samples with class colors
    colors = ['red', 'green', 'blue']
    markers = ['o', 's', '^']

    for idx, x in enumerate(shadow_features):
        bmu = som.predict(x)
        jitter = np.random.uniform(-0.2, 0.2, 2)
        plt.scatter(
            bmu[1] + jitter[1],
            bmu[0] + jitter[0],
            c=colors[y[idx]],
            marker=markers[y[idx]],
            s=60,
            edgecolors='white',
            linewidth=0.5,
            alpha=0.8
        )

    # Legend
    for i, name in enumerate(target_names):
        plt.scatter([], [], c=colors[i], marker=markers[i], s=60, label=name)
    plt.legend(loc='upper right')

    plt.title("QSOM Clustering of Iris Dataset", fontsize=14, fontweight='bold')
    plt.xlabel('Grid X')
    plt.ylabel('Grid Y')
    plt.savefig("iris_som_clusters.png", dpi=150, bbox_inches='tight')
    plt.close()

    # Training history
    fig = som.visualize_training(save_path="iris_training_history.png", show=False)
    if fig:
        plt.close(fig)

    # Summary statistics
    print("\n7. Results Summary:")
    print(f"   Final quantization error: {som.quantization_errors[-1]:.4f}")
    print(f"   U-matrix range: [{u_mat.min():.4f}, {u_mat.max():.4f}]")

    print("\n" + "=" * 60)
    print("Visualizations saved:")
    print("  - iris_som_visualization.png")
    print("  - iris_som_clusters.png")
    print("  - iris_training_history.png")
    print("=" * 60)


if __name__ == "__main__":
    main()
