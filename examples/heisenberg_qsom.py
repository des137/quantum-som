"""
Example: QSOM on 2D Heisenberg Model (4x4 Lattice).

This script demonstrates clustering of quantum phases in a 2D Heisenberg model.
System: 4x4 Square Lattice (16 Qubits).
Hamiltonian: H = J * sum(<ij>) (X_i X_j + Y_i Y_j + Z_i Z_j) + B * sum(Z_i)

We use Sparse Diagonalization (Lanczos) to find the ground state.
"""

import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp, Statevector
from scipy.sparse.linalg import eigsh

from qsom.backend import QuantumBackend
from qsom.shadows import ClassicalShadow
from qsom.som import QuantumSOM

def create_heisenberg_hamiltonian(J: float, B: float, grid_size: tuple = (4, 4)) -> SparsePauliOp:
    """
    Construct 2D Heisenberg Hamiltonian on a grid.
    H = J sum (XiXj + YiYj + ZiZj) + B sum Zi
    """
    rows, cols = grid_size
    n_qubits = rows * cols
    
    pauli_list = []
    
    # Interaction terms
    # Horizontal bonds
    for r in range(rows):
        for c in range(cols - 1):
            i = r * cols + c
            j = r * cols + c + 1
            for p_char in ["X", "Y", "Z"]:
                p_str = ["I"] * n_qubits
                p_str[n_qubits - 1 - i] = p_char
                p_str[n_qubits - 1 - j] = p_char
                pauli_list.append(("".join(p_str), J))
            
    # Vertical bonds
    for r in range(rows - 1):
        for c in range(cols):
            i = r * cols + c
            j = (r + 1) * cols + c
            for p_char in ["X", "Y", "Z"]:
                p_str = ["I"] * n_qubits
                p_str[n_qubits - 1 - i] = p_char
                p_str[n_qubits - 1 - j] = p_char
                pauli_list.append(("".join(p_str), J))
            
    # Magnetic field terms
    for i in range(n_qubits):
        p_str = ["I"] * n_qubits
        p_str[n_qubits - 1 - i] = "Z"
        pauli_list.append(("".join(p_str), B))
        
    return SparsePauliOp.from_list(pauli_list)

def get_ground_state(J: float, B: float) -> QuantumCircuit:
    """Compute ground state using Sparse Eigensolver."""
    # 4x4 = 16 qubits. Dimension 65536. Sparse matrix is feasible.
    ham = create_heisenberg_hamiltonian(J, B, grid_size=(4, 4))
    
    # Convert to scipy sparse matrix
    matrix = ham.to_matrix(sparse=True)
    
    # Find lowest eigenstate (k=1, which='SA' for Smallest Algebraic)
    evals, evecs = eigsh(matrix, k=1, which='SA')
    ground_state_vec = evecs[:, 0]
    
    # Create circuit
    qc = QuantumCircuit(16)
    qc.initialize(ground_state_vec, range(16))
    return qc

def main():
    print("Generating Heisenberg Ground States (4x4 Lattice)...")
    
    J = 1.0 
    
    # Scan B field from 0 to 6
    B_values = np.linspace(0, 6, 50)
    
    print(f"J = {J}")
    print(f"B values: {B_values}")
    
    circuits = []
    labels = [] 
    
    # Heuristic transition point (approximate)
    transition_B = 4.0 
    
    for i, B in enumerate(B_values):
        print(f"  Calculating Ground State {i+1}/{len(B_values)} for B={B:.2f}...")
        qc = get_ground_state(J, B)
        circuits.append(qc)
        labels.append(0 if B < transition_B else 1)
        
    n_qubits = 16
    
    print("Initializing Quantum Backend...")
    backend = QuantumBackend(use_simulator=True, shots=1)
    
    print("Generating Classical Shadows...")
    shadow_gen = ClassicalShadow(n_qubits=n_qubits, backend=backend, shadow_size=50)
    
    shadow_features = []
    for i, qc in enumerate(circuits):
        print(f"  Making shadow for sample {i+1}...")
        shadow_samples = shadow_gen.generate(qc)
        feat_vec = shadow_gen.shadow_to_feature_vector(shadow_samples)
        shadow_features.append(feat_vec)
        
    shadow_features = np.array(shadow_features)
    
    print("Training Quantum SOM...")
    som = QuantumSOM(grid_size=(8, 8), input_dim=shadow_features.shape[1], n_iterations=2000)
    som.train(shadow_features)
    
    print("Visualizing...")
    plt.figure(figsize=(10, 5))
    
    # U-Matrix
    plt.subplot(1, 2, 1)
    u_mat = som.get_u_matrix()
    plt.imshow(u_mat, cmap='viridis', interpolation='nearest')
    plt.title("U-Matrix")
    
    # Clusters
    plt.subplot(1, 2, 2)
    plt.imshow(u_mat, cmap='gray_r', alpha=0.3)
    
    colors = ['cyan', 'magenta']
    phase_names = ['AFM Phase', 'Polarized Phase']
    
    for phase in [0, 1]:
        mask = (np.array(labels) == phase)
        subset = shadow_features[mask]
        
        if len(subset) == 0: continue
        
        rows, cols = [], []
        for feat in subset:
            bmu = som._find_bmu(feat)
            rows.append(bmu[0])
            cols.append(bmu[1])
            
        plt.scatter(cols, rows, c=colors[phase], label=phase_names[phase], s=50)
        
    plt.legend()
    plt.title("Heisenberg (4x4) Phase Clustering")
    
    plt.tight_layout()
    plt.savefig("heisenberg_som_4x4.png")
    print("Done. Saved 'heisenberg_som_4x4.png'.")

if __name__ == "__main__":
    import sys
    sys.path.append("src")
    main()
