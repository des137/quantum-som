"""
Quantum Self-Organizing Maps (QSOM) Implementation

This module implements a self-organizing map algorithm for exploring
the state space of quantum states using classical shadows formalism.

Key Features:
- Classical shadows generation for efficient quantum state representation
- Self-organizing map implementation with quantum-specific distance metrics
- Visualization tools for exploring quantum state space
- Qiskit 2.0+ integration for real quantum measurements
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional, Callable, Dict
from scipy.linalg import expm
from scipy.spatial.distance import cdist
import warnings
warnings.filterwarnings('ignore')

# Qiskit 2.0+ imports
try:
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Statevector, DensityMatrix, SparsePauliOp, Pauli
    from qiskit.primitives import Sampler, StatevectorSampler
    from qiskit_aer import AerSimulator
    from qiskit_aer.primitives import Sampler as AerSampler
    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False
    print("Warning: Qiskit not available. Install with: pip install qiskit>=2.0 qiskit-aer")


class ClassicalShadow:
    """
    Classical Shadow representation of a quantum state using Qiskit 2.0+.
    
    Classical shadows provide an efficient way to represent quantum states
    using randomized measurements, enabling classical processing of quantum data.
    """
    
    def __init__(self, n_qubits: int, n_shots: int = 1000, use_qiskit: bool = True):
        """
        Initialize classical shadow generator.
        
        Args:
            n_qubits: Number of qubits in the quantum system
            n_shots: Number of measurement shots for shadow generation
            use_qiskit: Whether to use Qiskit for actual quantum measurements
        """
        self.n_qubits = n_qubits
        self.n_shots = n_shots
        self.use_qiskit = use_qiskit and QISKIT_AVAILABLE
        self.pauli_basis = ['X', 'Y', 'Z']
        
        # Initialize Qiskit sampler if available
        if self.use_qiskit:
            try:
                self.sampler = AerSampler()
            except:
                self.use_qiskit = False
                print("Warning: Could not initialize Qiskit sampler, using classical simulation")
        
    def _pauli_to_rotation_gates(self, pauli_string: str) -> QuantumCircuit:
        """
        Convert Pauli string to rotation gates for measurement.
        
        Args:
            pauli_string: String of Pauli operators (e.g., 'XYZ')
            
        Returns:
            QuantumCircuit with rotation gates
        """
        qc = QuantumCircuit(self.n_qubits)
        
        for i, pauli in enumerate(pauli_string):
            if pauli == 'X':
                qc.h(i)  # Hadamard rotates Z basis to X basis
            elif pauli == 'Y':
                qc.sdg(i)  # S-dagger
                qc.h(i)    # Rotates Z basis to Y basis
            # Z basis is default, no rotation needed
        
        return qc
    
    def generate_random_pauli_measurement(self) -> str:
        """Generate a random Pauli measurement basis for each qubit."""
        return ''.join(np.random.choice(self.pauli_basis, size=self.n_qubits))
    
    def measure_with_qiskit(self, statevector: Statevector, pauli_string: str) -> str:
        """
        Perform measurement in Pauli basis using Qiskit 2.0+.
        
        Args:
            statevector: Qiskit Statevector object
            pauli_string: Pauli measurement basis (e.g., 'XYZ')
            
        Returns:
            Bitstring measurement outcome
        """
        # Create circuit with state preparation
        qc = QuantumCircuit(self.n_qubits)
        qc.prepare_state(statevector, range(self.n_qubits))
        
        # Add rotation gates for Pauli measurement
        rotation_circuit = self._pauli_to_rotation_gates(pauli_string)
        qc.compose(rotation_circuit, inplace=True)
        
        # Measure all qubits
        qc.measure_all()
        
        # Run sampler
        job = self.sampler.run([qc], shots=1)
        result = job.result()
        
        # Get measurement outcome
        quasi_dist = result.quasi_dists[0]
        # Get the most likely outcome
        bitstring = max(quasi_dist, key=quasi_dist.get)
        
        return bitstring
    
    def generate_shadow_from_statevector(self, statevector: Statevector) -> List[Dict]:
        """
        Generate classical shadow from Qiskit Statevector.
        
        Args:
            statevector: Qiskit Statevector object
            
        Returns:
            List of shadow measurement records
        """
        shadow_data = []
        
        for _ in range(self.n_shots):
            # Generate random Pauli measurement
            pauli_string = self.generate_random_pauli_measurement()
            
            if self.use_qiskit:
                # Use Qiskit for actual measurement
                bitstring = self.measure_with_qiskit(statevector, pauli_string)
                outcomes = [int(bit) for bit in bitstring]
            else:
                # Fallback to classical simulation
                outcomes = self._classical_measurement(statevector, pauli_string)
            
            shadow_data.append({
                'pauli_basis': pauli_string,
                'outcomes': outcomes
            })
        
        return shadow_data
    
    def _classical_measurement(self, statevector: Statevector, pauli_string: str) -> List[int]:
        """Classical simulation of Pauli measurement."""
        # Get probabilities in computational basis after rotation
        qc = QuantumCircuit(self.n_qubits)
        qc.prepare_state(statevector, range(self.n_qubits))
        rotation_circuit = self._pauli_to_rotation_gates(pauli_string)
        qc.compose(rotation_circuit, inplace=True)
        
        # Get probabilities
        state_after_rotation = Statevector(qc)
        probs = state_after_rotation.probabilities()
        
        # Sample from distribution
        outcome_idx = np.random.choice(len(probs), p=probs)
        bitstring = format(outcome_idx, f'0{self.n_qubits}b')
        
        return [int(bit) for bit in bitstring]
    
    def generate_shadow(self, state) -> np.ndarray:
        """
        Generate classical shadow of a quantum state.
        
        Args:
            state: Quantum state (Statevector, DensityMatrix, or numpy array)
            
        Returns:
            Classical shadow representation as a feature vector
        """
        # Convert input to Qiskit Statevector if needed
        if isinstance(state, np.ndarray):
            if state.ndim == 1:
                statevector = Statevector(state)
            elif state.ndim == 2:
                # Density matrix - convert to statevector (assuming pure state)
                # For mixed states, we'd need a different approach
                statevector = Statevector(state)
            else:
                raise ValueError("Invalid state format")
        elif isinstance(state, Statevector):
            statevector = state
        elif isinstance(state, DensityMatrix):
            # For density matrices, sample a pure state decomposition
            # This is a simplification - full implementation would handle mixed states
            statevector = Statevector(state)
        else:
            raise ValueError(f"Unsupported state type: {type(state)}")
        
        # Generate shadow measurements
        shadow_data = self.generate_shadow_from_statevector(statevector)
        
        # Convert to feature vector
        feature_vector = self._shadow_to_feature_vector(shadow_data)
        
        return feature_vector
    
    def _shadow_to_feature_vector(self, shadow_data: List[dict]) -> np.ndarray:
        """
        Convert shadow data to a feature vector.
        
        This is a simplified encoding. A more sophisticated approach
        would use the inverse channel reconstruction.
        """
        # Encode Pauli measurements and outcomes
        feature_dim = self.n_qubits * len(self.pauli_basis) * 2
        feature_vector = np.zeros(feature_dim)
        
        for record in shadow_data:
            for i, (pauli, outcome) in enumerate(zip(record['pauli_basis'], 
                                                      record['outcomes'])):
                pauli_idx = self.pauli_basis.index(pauli)
                base_idx = i * len(self.pauli_basis) * 2 + pauli_idx * 2
                feature_vector[base_idx + outcome] += 1
        
        # Normalize
        feature_vector = feature_vector / len(shadow_data)
        
        return feature_vector


class QuantumSOM:
    """
    Self-Organizing Map for Quantum State Space Exploration.
    
    This implementation uses classical shadows to represent quantum states
    and applies SOM to visualize and explore the quantum state space.
    """
    
    def __init__(
        self,
        grid_size: Tuple[int, int] = (10, 10),
        input_dim: int = None,
        learning_rate: float = 0.5,
        sigma: float = 1.0,
        distance_metric: str = 'euclidean',
        use_fidelity: bool = False
    ):
        """
        Initialize Quantum SOM.
        
        Args:
            grid_size: Size of the SOM grid (height, width)
            input_dim: Dimension of input vectors (classical shadows)
            learning_rate: Initial learning rate
            sigma: Initial neighborhood radius
            distance_metric: Distance metric ('euclidean', 'fidelity', 'quantum')
            use_fidelity: Whether to use quantum fidelity as distance metric
        """
        self.grid_size = grid_size
        self.input_dim = input_dim
        self.learning_rate = learning_rate
        self.sigma = sigma
        self.distance_metric = distance_metric
        self.use_fidelity = use_fidelity
        
        # Initialize weight vectors (neurons)
        if input_dim is not None:
            self.weights = np.random.randn(grid_size[0], grid_size[1], input_dim)
        else:
            self.weights = None
            
        # For tracking training
        self.quantization_error = []
        self.topographic_error = []
        
    def _quantum_fidelity(self, state1: np.ndarray, state2: np.ndarray) -> float:
        """
        Calculate quantum fidelity between two states.
        
        For classical shadows, we use a simplified fidelity measure.
        In practice, this would involve proper quantum state reconstruction.
        """
        # Simplified fidelity: cosine similarity for shadow vectors
        # Real implementation would reconstruct density matrices first
        dot_product = np.dot(state1, state2)
        norm1 = np.linalg.norm(state1)
        norm2 = np.linalg.norm(state2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        similarity = dot_product / (norm1 * norm2)
        # Convert similarity to distance (1 - similarity)
        return 1.0 - similarity
    
    def _distance(self, x: np.ndarray, y: np.ndarray) -> float:
        """Calculate distance between two vectors."""
        if self.use_fidelity or self.distance_metric == 'fidelity':
            return self._quantum_fidelity(x, y)
        elif self.distance_metric == 'quantum':
            # Quantum distance: combination of fidelity and other metrics
            fidelity_dist = self._quantum_fidelity(x, y)
            euclidean_dist = np.linalg.norm(x - y)
            return 0.7 * fidelity_dist + 0.3 * euclidean_dist
        else:
            return np.linalg.norm(x - y)
    
    def _find_best_matching_unit(self, x: np.ndarray) -> Tuple[int, int]:
        """Find the best matching unit (BMU) for input vector x."""
        min_dist = float('inf')
        bmu = (0, 0)
        
        for i in range(self.grid_size[0]):
            for j in range(self.grid_size[1]):
                dist = self._distance(x, self.weights[i, j])
                if dist < min_dist:
                    min_dist = dist
                    bmu = (i, j)
        
        return bmu
    
    def _neighborhood_function(self, distance: float, sigma: float) -> float:
        """Gaussian neighborhood function."""
        return np.exp(-(distance**2) / (2 * sigma**2))
    
    def _get_neighborhood(self, center: Tuple[int, int], sigma: float) -> List[Tuple[int, int]]:
        """Get neurons in the neighborhood of the center."""
        neighbors = []
        center_i, center_j = center
        
        # Calculate neighborhood radius
        radius = int(np.ceil(3 * sigma))  # 3-sigma rule
        
        for i in range(max(0, center_i - radius), 
                      min(self.grid_size[0], center_i + radius + 1)):
            for j in range(max(0, center_j - radius),
                          min(self.grid_size[1], center_j + radius + 1)):
                neighbors.append((i, j))
        
        return neighbors
    
    def initialize_weights(self, data: np.ndarray):
        """Initialize weights using PCA or random sampling from data."""
        if self.input_dim is None:
            self.input_dim = data.shape[1]
            self.weights = np.random.randn(
                self.grid_size[0], self.grid_size[1], self.input_dim
            )
        
        # Initialize with random samples from data
        n_samples = self.grid_size[0] * self.grid_size[1]
        indices = np.random.choice(len(data), size=n_samples, replace=True)
        samples = data[indices]
        
        self.weights = samples.reshape(
            self.grid_size[0], self.grid_size[1], self.input_dim
        )
    
    def train(
        self,
        data: np.ndarray,
        n_iterations: int = 1000,
        verbose: bool = True
    ):
        """
        Train the SOM on quantum state data.
        
        Args:
            data: Array of classical shadows (n_samples, n_features)
            n_iterations: Number of training iterations
            verbose: Whether to print training progress
        """
        if self.weights is None:
            self.initialize_weights(data)
        
        n_samples = len(data)
        
        for iteration in range(n_iterations):
            # Update learning parameters
            current_lr = self.learning_rate * (1 - iteration / n_iterations)
            current_sigma = self.sigma * (1 - iteration / n_iterations)
            current_sigma = max(current_sigma, 0.1)  # Minimum sigma
            
            # Randomly select a sample
            idx = np.random.randint(n_samples)
            x = data[idx]
            
            # Find BMU
            bmu = self._find_best_matching_unit(x)
            
            # Update weights in neighborhood
            neighbors = self._get_neighborhood(bmu, current_sigma)
            
            for i, j in neighbors:
                # Calculate distance from BMU
                dist_to_bmu = np.sqrt((i - bmu[0])**2 + (j - bmu[1])**2)
                
                # Calculate neighborhood influence
                influence = self._neighborhood_function(dist_to_bmu, current_sigma)
                
                # Update weight
                self.weights[i, j] += current_lr * influence * (x - self.weights[i, j])
            
            # Track errors periodically
            if iteration % 100 == 0 and verbose:
                qe = self._quantization_error(data)
                self.quantization_error.append(qe)
                if verbose:
                    print(f"Iteration {iteration}: Quantization Error = {qe:.4f}")
    
    def _quantization_error(self, data: np.ndarray) -> float:
        """Calculate quantization error (average distance to BMU)."""
        total_error = 0.0
        for x in data:
            bmu = self._find_best_matching_unit(x)
            dist = self._distance(x, self.weights[bmu[0], bmu[1]])
            total_error += dist
        return total_error / len(data)
    
    def predict(self, x: np.ndarray) -> Tuple[int, int]:
        """Predict the grid position for a quantum state."""
        return self._find_best_matching_unit(x)
    
    def get_umatrix(self) -> np.ndarray:
        """
        Calculate U-matrix (unified distance matrix) for visualization.
        
        The U-matrix shows distances between neighboring neurons,
        helping identify clusters in the quantum state space.
        """
        umatrix = np.zeros(self.grid_size)
        
        for i in range(self.grid_size[0]):
            for j in range(self.grid_size[1]):
                distances = []
                
                # Check neighbors
                for di in [-1, 0, 1]:
                    for dj in [-1, 0, 1]:
                        if di == 0 and dj == 0:
                            continue
                        ni, nj = i + di, j + dj
                        if 0 <= ni < self.grid_size[0] and 0 <= nj < self.grid_size[1]:
                            dist = self._distance(
                                self.weights[i, j],
                                self.weights[ni, nj]
                            )
                            distances.append(dist)
                
                umatrix[i, j] = np.mean(distances) if distances else 0
        
        return umatrix
    
    def visualize(
        self,
        data: np.ndarray = None,
        labels: np.ndarray = None,
        title: str = "Quantum SOM Visualization"
    ):
        """
        Visualize the trained SOM.
        
        Args:
            data: Optional data to plot on the map
            labels: Optional labels for the data points
            title: Plot title
        """
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 1: U-matrix
        umatrix = self.get_umatrix()
        im1 = axes[0].imshow(umatrix, cmap='viridis', origin='lower')
        axes[0].set_title('U-Matrix (Distance Map)')
        axes[0].set_xlabel('Grid X')
        axes[0].set_ylabel('Grid Y')
        plt.colorbar(im1, ax=axes[0])
        
        # Plot 2: Component planes or data projection
        if data is not None:
            # Project data onto the map
            projections = []
            for x in data:
                bmu = self.predict(x)
                projections.append(bmu)
            
            projections = np.array(projections)
            
            # Create hit histogram
            hit_map = np.zeros(self.grid_size)
            for proj in projections:
                hit_map[proj[0], proj[1]] += 1
            
            im2 = axes[1].imshow(hit_map, cmap='hot', origin='lower')
            axes[1].set_title('Data Hit Map')
            axes[1].set_xlabel('Grid X')
            axes[1].set_ylabel('Grid Y')
            plt.colorbar(im2, ax=axes[1])
            
            # Overlay labels if provided
            if labels is not None:
                unique_labels = np.unique(labels)
                colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
                for label, color in zip(unique_labels, colors):
                    mask = labels == label
                    proj_label = projections[mask]
                    axes[1].scatter(
                        proj_label[:, 1], proj_label[:, 0],
                        c=[color], label=f'Class {label}', alpha=0.6, s=50
                    )
                axes[1].legend()
        else:
            # Show component planes (first few dimensions)
            n_components = min(4, self.input_dim)
            for idx in range(n_components):
                component = self.weights[:, :, idx]
                im2 = axes[1].imshow(component, cmap='coolwarm', origin='lower')
                axes[1].set_title(f'Component Plane (Dimension {idx})')
                plt.colorbar(im2, ax=axes[1])
        
        plt.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        return fig


def generate_quantum_states(
    n_states: int,
    n_qubits: int,
    state_type: str = 'random',
    use_qiskit: bool = True
) -> List:
    """
    Generate sample quantum states for testing using Qiskit 2.0+.
    
    Args:
        n_states: Number of states to generate
        n_qubits: Number of qubits
        state_type: Type of states ('random', 'bell', 'ghz', 'w')
        use_qiskit: Whether to return Qiskit Statevector objects
        
    Returns:
        List of quantum states (Statevector objects if use_qiskit=True, else numpy arrays)
    """
    states = []
    
    if use_qiskit and QISKIT_AVAILABLE:
        for _ in range(n_states):
            if state_type == 'random':
                # Random pure state using Qiskit
                statevector = Statevector.random(2**n_qubits)
                states.append(statevector)
            elif state_type == 'bell':
                # Bell state (for 2 qubits)
                if n_qubits == 2:
                    bell_state = Statevector([1, 0, 0, 1] / np.sqrt(2))
                    states.append(bell_state)
                else:
                    # GHZ state for more qubits
                    ghz_state = Statevector([1] + [0] * (2**n_qubits - 2) + [1]) / np.sqrt(2)
                    states.append(ghz_state)
            elif state_type == 'ghz':
                # GHZ state
                ghz = np.zeros(2**n_qubits, dtype=complex)
                ghz[0] = 1 / np.sqrt(2)
                ghz[-1] = 1 / np.sqrt(2)
                states.append(Statevector(ghz))
            elif state_type == 'w':
                # W state (equal superposition of all single-excitation states)
                w = np.zeros(2**n_qubits, dtype=complex)
                for i in range(n_qubits):
                    idx = 2**i
                    w[idx] = 1 / np.sqrt(n_qubits)
                states.append(Statevector(w))
            else:
                # Default to random
                statevector = Statevector.random(2**n_qubits)
                states.append(statevector)
    else:
        # Fallback to numpy arrays
        dim = 2 ** n_qubits
        for _ in range(n_states):
            if state_type == 'random':
                state_vector = np.random.randn(dim) + 1j * np.random.randn(dim)
                state_vector = state_vector / np.linalg.norm(state_vector)
                rho = np.outer(state_vector, state_vector.conj())
                states.append(rho)
            elif state_type == 'bell':
                if n_qubits == 2:
                    bell = np.array([1, 0, 0, 1]) / np.sqrt(2)
                    rho = np.outer(bell, bell.conj())
                    states.append(rho)
                else:
                    state_vector = np.random.randn(dim) + 1j * np.random.randn(dim)
                    state_vector = state_vector / np.linalg.norm(state_vector)
                    rho = np.outer(state_vector, state_vector.conj())
                    states.append(rho)
            else:
                state_vector = np.random.randn(dim) + 1j * np.random.randn(dim)
                state_vector = state_vector / np.linalg.norm(state_vector)
                rho = np.outer(state_vector, state_vector.conj())
                states.append(rho)
    
    return states


# Example usage and testing
if __name__ == "__main__":
    print("Quantum Self-Organizing Maps (QSOM) Implementation")
    print("=" * 60)
    
    # Parameters
    n_qubits = 3
    n_states = 100
    n_shots = 500
    grid_size = (10, 10)
    
    print(f"\n1. Generating {n_states} quantum states ({n_qubits} qubits)...")
    quantum_states = generate_quantum_states(n_states, n_qubits, state_type='random', use_qiskit=True)
    
    print(f"2. Generating classical shadows using Qiskit 2.0+ (n_shots={n_shots})...")
    shadow_generator = ClassicalShadow(n_qubits, n_shots, use_qiskit=True)
    classical_shadows = []
    
    for i, state in enumerate(quantum_states):
        shadow = shadow_generator.generate_shadow(state)
        classical_shadows.append(shadow)
        if (i + 1) % 10 == 0:
            print(f"   Processed {i + 1}/{n_states} states...")
    
    classical_shadows = np.array(classical_shadows)
    print(f"   Shadow dimension: {classical_shadows.shape[1]}")
    
    print(f"3. Training Quantum SOM (grid size: {grid_size})...")
    qsom = QuantumSOM(
        grid_size=grid_size,
        input_dim=classical_shadows.shape[1],
        learning_rate=0.5,
        sigma=3.0,
        distance_metric='quantum',
        use_fidelity=True
    )
    
    qsom.train(classical_shadows, n_iterations=1000, verbose=True)
    
    print(f"4. Visualizing results...")
    fig = qsom.visualize(
        data=classical_shadows,
        title="Quantum State Space Exploration using SOM"
    )
    plt.savefig('qsom_visualization.png', dpi=150, bbox_inches='tight')
    print("   Saved visualization to 'qsom_visualization.png'")
    
    print("\n5. Analysis:")
    print(f"   Final quantization error: {qsom._quantization_error(classical_shadows):.4f}")
    print(f"   U-matrix range: [{qsom.get_umatrix().min():.4f}, {qsom.get_umatrix().max():.4f}]")
    
    print("\n" + "=" * 60)
    print("Implementation complete!")

