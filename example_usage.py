"""
Example usage of Quantum Self-Organizing Maps

This script demonstrates how to use the QSOM implementation
to explore quantum state spaces.
"""

import numpy as np
import matplotlib.pyplot as plt
from qsom_implementation import (
    QuantumSOM, 
    ClassicalShadow, 
    generate_quantum_states
)
from qsom_enhanced import EnhancedClassicalShadow, EnhancedQuantumSOM

# Qiskit 2.0+ imports
try:
    from qiskit.quantum_info import Statevector
    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False
    print("Warning: Qiskit not available. Some features may be limited.")


def example_basic_usage():
    """Basic example of QSOM usage."""
    print("\n" + "="*60)
    print("Example 1: Basic QSOM Usage")
    print("="*60)
    
    # Parameters
    n_qubits = 3
    n_states = 50
    n_shots = 500
    grid_size = (8, 8)
    
    # Generate quantum states using Qiskit 2.0+
    print(f"Generating {n_states} quantum states using Qiskit 2.0+...")
    quantum_states = generate_quantum_states(n_states, n_qubits, state_type='random', use_qiskit=True)
    
    # Generate classical shadows with Qiskit
    print(f"Generating classical shadows using Qiskit (n_shots={n_shots})...")
    shadow_generator = ClassicalShadow(n_qubits, n_shots, use_qiskit=True)
    classical_shadows = np.array([
        shadow_generator.generate_shadow(state) for state in quantum_states
    ])
    
    # Train SOM
    print(f"Training SOM (grid size: {grid_size})...")
    qsom = QuantumSOM(
        grid_size=grid_size,
        input_dim=classical_shadows.shape[1],
        learning_rate=0.5,
        sigma=3.0,
        distance_metric='quantum',
        use_fidelity=True
    )
    
    qsom.train(classical_shadows, n_iterations=500, verbose=False)
    
    # Visualize
    print("Creating visualization...")
    fig = qsom.visualize(
        data=classical_shadows,
        title="Basic QSOM: Quantum State Space Exploration"
    )
    plt.savefig('example_basic_qsom.png', dpi=150, bbox_inches='tight')
    print("Saved to 'example_basic_qsom.png'")
    plt.close()
    
    return qsom, classical_shadows


def example_enhanced_usage():
    """Example using enhanced implementation."""
    print("\n" + "="*60)
    print("Example 2: Enhanced QSOM with Improved Shadows")
    print("="*60)
    
    n_qubits = 2  # Smaller for faster computation
    n_states = 30
    n_shots = 300
    grid_size = (6, 6)
    
    # Generate quantum states using Qiskit 2.0+
    print(f"Generating {n_states} quantum states using Qiskit 2.0+...")
    quantum_states = generate_quantum_states(n_states, n_qubits, state_type='random', use_qiskit=True)
    
    # Use enhanced shadow generator with Qiskit
    print(f"Generating enhanced classical shadows using Qiskit 2.0+...")
    enhanced_shadow_gen = EnhancedClassicalShadow(n_qubits, n_shots, use_qiskit=True)
    enhanced_shadows = np.array([
        enhanced_shadow_gen.generate_shadow(state) for state in quantum_states
    ])
    
    # Train enhanced SOM
    print(f"Training enhanced SOM...")
    enhanced_qsom = EnhancedQuantumSOM(
        grid_size=grid_size,
        input_dim=enhanced_shadows.shape[1],
        learning_rate=0.5,
        sigma=2.0,
        distance_metric='quantum',
        use_fidelity=True
    )
    
    enhanced_qsom.train(enhanced_shadows, n_iterations=500, verbose=False)
    
    # Visualize
    print("Creating visualization...")
    fig = enhanced_qsom.visualize(
        data=enhanced_shadows,
        title="Enhanced QSOM with Improved Classical Shadows"
    )
    plt.savefig('example_enhanced_qsom.png', dpi=150, bbox_inches='tight')
    print("Saved to 'example_enhanced_qsom.png'")
    plt.close()
    
    return enhanced_qsom, enhanced_shadows


def example_comparison():
    """Compare different distance metrics."""
    print("\n" + "="*60)
    print("Example 3: Comparing Distance Metrics")
    print("="*60)
    
    n_qubits = 2
    n_states = 20
    n_shots = 200
    grid_size = (5, 5)
    
    # Generate data using Qiskit 2.0+
    quantum_states = generate_quantum_states(n_states, n_qubits, use_qiskit=True)
    shadow_gen = ClassicalShadow(n_qubits, n_shots, use_qiskit=True)
    shadows = np.array([shadow_gen.generate_shadow(s) for s in quantum_states])
    
    metrics = ['euclidean', 'quantum']
    results = {}
    
    for metric in metrics:
        print(f"\nTraining with {metric} distance metric...")
        qsom = QuantumSOM(
            grid_size=grid_size,
            input_dim=shadows.shape[1],
            distance_metric=metric,
            use_fidelity=(metric == 'quantum')
        )
        qsom.train(shadows, n_iterations=300, verbose=False)
        
        qe = qsom._quantization_error(shadows)
        results[metric] = {
            'quantization_error': qe,
            'som': qsom
        }
        print(f"  Quantization error: {qe:.4f}")
    
    # Compare results
    print("\nComparison:")
    for metric, result in results.items():
        print(f"  {metric}: QE = {result['quantization_error']:.4f}")
    
    return results


def example_state_clustering():
    """Example of clustering different types of quantum states."""
    print("\n" + "="*60)
    print("Example 4: Clustering Different Quantum State Types")
    print("="*60)
    
    n_qubits = 2
    n_per_type = 15
    n_shots = 300
    
    # Generate different types of states using Qiskit 2.0+
    print("Generating different state types using Qiskit 2.0+...")
    random_states = generate_quantum_states(n_per_type, n_qubits, 'random', use_qiskit=True)
    bell_states = generate_quantum_states(n_per_type, n_qubits, 'bell', use_qiskit=True)
    
    all_states = random_states + bell_states
    labels = np.array([0]*n_per_type + [1]*n_per_type)
    
    # Generate shadows with Qiskit
    shadow_gen = ClassicalShadow(n_qubits, n_shots, use_qiskit=True)
    shadows = np.array([shadow_gen.generate_shadow(s) for s in all_states])
    
    # Train SOM
    print("Training SOM for clustering...")
    qsom = QuantumSOM(
        grid_size=(6, 6),
        input_dim=shadows.shape[1],
        distance_metric='quantum',
        use_fidelity=True
    )
    qsom.train(shadows, n_iterations=500, verbose=False)
    
    # Visualize with labels
    print("Creating labeled visualization...")
    fig = qsom.visualize(
        data=shadows,
        labels=labels,
        title="Quantum State Clustering: Random vs Bell States"
    )
    plt.savefig('example_clustering.png', dpi=150, bbox_inches='tight')
    print("Saved to 'example_clustering.png'")
    plt.close()
    
    return qsom, shadows, labels


if __name__ == "__main__":
    print("Quantum Self-Organizing Maps - Example Usage")
    print("="*60)
    
    try:
        # Run examples
        example_basic_usage()
        example_enhanced_usage()
        example_comparison()
        example_state_clustering()
        
        print("\n" + "="*60)
        print("All examples completed successfully!")
        print("="*60)
        print("\nGenerated files:")
        print("  - example_basic_qsom.png")
        print("  - example_enhanced_qsom.png")
        print("  - example_clustering.png")
        
    except Exception as e:
        print(f"\nError running examples: {e}")
        import traceback
        traceback.print_exc()

