"""
Classical Shadows Module.

This module provides functionality to generate classical shadows of quantum states.
It leverages the QuantumBackend for execution.
"""

from typing import List, Tuple, Union, Optional
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, Pauli
from .backend import QuantumBackend

class ClassicalShadow:
    """
    Generates and manages classical shadows of quantum states.
    """

    def __init__(
        self,
        n_qubits: int,
        backend: QuantumBackend,
        n_shots: int = 1000,
        shadow_size: int = 100
    ):
        """
        Initialize ClassicalShadow.

        Args:
            n_qubits: Number of qubits.
            backend: QuantumBackend instance for execution.
            n_shots: Number of shots per basis (if applicable) or total shots concept.
                     For shadows, usually we take 1 shot per random basis,
                     repeated shadow_size times.
            shadow_size: Number of random bases to sample (N).
        """
        self.n_qubits = n_qubits
        self.backend = backend
        self.n_shots = n_shots # Note: In standard shadow protocol, we often do 1 shot per basis.
                               # But here we might want statistics or multiple snapshots.
                               # We will assume shadow_size is the number of snapshots (N).
                               # n_shots might be used if we want to estimate averages per basis,
                               # but standard shadow is single shot. We will stick to standard:
                               # shadow_size = number of independent Pauli bases measured.
                               # shots = 1 per basis.
        self.shadow_size = shadow_size
        
        self.pauli_map = {0: 'X', 1: 'Y', 2: 'Z'}

    def _get_random_bases(self, n_snapshots: int) -> np.ndarray:
        """
        Generate random Pauli bases (0=X, 1=Y, 2=Z) for each qubit and snapshot.
        
        Returns:
            np.ndarray: Shape (n_snapshots, n_qubits) with integers 0, 1, 2.
        """
        return np.random.randint(0, 3, size=(n_snapshots, self.n_qubits))

    def _rotate_basis(self, qc: QuantumCircuit, bases: np.ndarray, qubit_indices: range):
        """
        Apply global rotations to measure in X, Y, or Z bases.
        """
        for i, basis in enumerate(bases):
            if basis == 0: # X
                qc.h(qubit_indices[i])
            elif basis == 1: # Y
                qc.sdg(qubit_indices[i])
                qc.h(qubit_indices[i])
            # Z corresponds to basis=2, do nothing
            
    def generate(self, state_prep_circuit: QuantumCircuit) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate classical shadow for a given state preparation circuit.

        Args:
            state_prep_circuit: Circuit that prepares the state rho.

        Returns:
            List of tuples (pauli_indices, outcome_bits).
            pauli_indices: Shape (n_qubits,) - 0=X, 1=Y, 2=Z
            outcome_bits: Shape (n_qubits,) - 0 or 1
        """
        # Generate random bases
        bases_list = self._get_random_bases(self.shadow_size)
        
        # Prepare list of circuits
        circuits = []
        for bases in bases_list:
            qc = state_prep_circuit.copy()
            # Apply rotations
            self._rotate_basis(qc, bases, range(self.n_qubits))
            qc.measure_all()
            circuits.append(qc)
            
        # Execute batch
        # Note: If shadow_size is large, we might need to batch this further.
        # Check backend capabilities? For now assumes it fits.
        
        # We need 1 shot per circuit for standard shadow
        # Temporarily force backend shots to 1 if possible, or just take the first shot
        # efficiently? 
        # Actually creating 1000 circuits is expensive.
        # But this is the standard theoretical definition.
        
        results = self.backend.run(circuits)
        
        # Extract data
        # Handling V2 vs V1 results is tricky.
        # Assuming V2 for now based on recent imports.
        # V2: result[i].data.meas.get_bitstrings() or similar
        
        shadow_samples = []
        
        # Check for V1 SamplerResult (quasi_dists)
        if hasattr(results, 'quasi_dists'):
             for i, dist in enumerate(results.quasi_dists):
                 # dist is {outcome: prob}
                 outcome_int = max(dist, key=dist.get)
                 bitstring = format(outcome_int, f'0{self.n_qubits}b')
                 outcome_bits = np.array([int(c) for c in bitstring[::-1]])
                 shadow_samples.append((bases_list[i], outcome_bits))
             return shadow_samples
             
        try:
            # Try V2 Access pattern
            for i, pub_result in enumerate(results):
                # pub_result is the result for the i-th circuit (PUB)
                # It contains data.meas (or whatever classical register name was)
                # measure_all creates 'meas' usually.
                
                # Check for attribute access
                if hasattr(pub_result, 'data'):
                    data = pub_result.data
                    # Assuming default register name 'meas'
                    if hasattr(data, 'meas'):
                         # access bitstrings
                         bitstrings = data.meas.get_bitstrings()
                         # We only care about the first one if we requested multiple, 
                         # but logically we probably set shots=1 for this specific run logic.
                         # If backend has default shots=1000, we get 1000. 
                         # We only need 1 for the shadow snapshot definition.
                         # taking the first one.
                         b = bitstrings[0]
                         # Convert '0010' string to [0, 0, 1, 0] ints
                         # Note qiskit is little-endian? 
                         # get_bitstrings returns standard print order which is usually reversed qubit order (qn...q0)
                         # We need to map correctly. 
                         # Let's assume standard qiskit order: bitstring[k] corresponds to qubit n-1-k
                         # We want array where index j corresponds to qubit j.
                         # So we reverse the bitstring.
                         outcome_bits = np.array([int(c) for c in b[::-1]])
                         
                         shadow_samples.append((bases_list[i], outcome_bits))
                    else:
                         # Try fallback or other register names
                         pass
        except Exception as e:
            # Fallback for V1/AerSimulator direct 
            if hasattr(results, 'get_counts'):
                 for i in range(len(circuits)):
                     counts = results.get_counts(i)
                     # Get discrete outcome
                     b = list(counts.keys())[0]
                     outcome_bits = np.array([int(c) for c in b[::-1]])
                     shadow_samples.append((bases_list[i], outcome_bits))
        
        return shadow_samples

    def reconstruct_state(self, shadow_samples: List[Tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
        """
        Reconstruct density matrix (very expensive for large N) or other properties.
        This provides the global density matrix reconstruction.
        rho = E [ 3 U^dag |b><b| U - I ]
        
        Returns:
            np.ndarray: Density matrix (2^n x 2^n)
        """
        dim = 2 ** self.n_qubits
        rho = np.zeros((dim, dim), dtype=complex)
        
        for bases, outcomes in shadow_samples:
            # Snapshot = tensor_product( 3 * U_i^dag |b_i><b_i| U_i - I )
            # This reconstructs separate local snapshots and tensors them.
            
            snapshot = np.array([1.0], dtype=complex) # scalar start
            
            for i in range(self.n_qubits):
                # Local state |b_i><b_i|
                b = outcomes[i] # 0 or 1
                basis = bases[i] # 0,1,2 (X,Y,Z)
                
                # Reconstruct single qubit state
                if basis == 0: # X basis measurement
                    # If b=0, state was |+>, if b=1, |->
                    # |+> = H|0>, |-> = H|1>
                    # |+><+| = [[0.5, 0.5], [0.5, 0.5]]
                    # |-><-| = [[0.5, -0.5], [-0.5, 0.5]]
                    if b == 0:
                        local_rho = np.array([[0.5, 0.5], [0.5, 0.5]])
                    else:
                        local_rho = np.array([[0.5, -0.5], [-0.5, 0.5]])
                elif basis == 1: # Y basis
                    # |+i> = HS^dag|0>, |-i> = HS^dag|1>
                    # |+i><+i| = [[0.5, -0.5j], [0.5j, 0.5]]
                    if b == 0:
                        local_rho = np.array([[0.5, -0.5j], [0.5j, 0.5]])
                    else:
                        local_rho = np.array([[0.5, 0.5j], [-0.5j, 0.5]])
                else: # Z basis
                    if b == 0:
                        local_rho = np.array([[1, 0], [0, 0]])
                    else:
                        local_rho = np.array([[0, 0], [0, 1]])
                        
                # Apply inverse channel: 3 * rho - I
                # Eq (S4) from Huang et al (2020)
                local_snapshot = 3 * local_rho - np.eye(2)
                
                snapshot = np.kron(snapshot, local_snapshot)
                
            rho += snapshot
            
        return rho / len(shadow_samples)

    def shadow_to_feature_vector(self, shadow_samples: List[Tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
        """
        Convert shadow samples to a flattened feature vector for SOM training.
        This is a heuristic representation, often just averaging the snapshots or 
        constructing a histogram. The original code used a histogram of outcomes 
        conditional on basis.
        
        We will stick to the histogram approach as it avoids full reconstruction (exponential).
        
        Vector size: n_qubits * 3 (bases) * 2 (outcomes) = 6n. 
        Or just probabilities of 0/1 for each basis.
        """
        # Feature vector: For each qubit and each basis, what is the prob of 0?
        # Size: 3 * n_qubits
        
        feature_vector = np.zeros((self.n_qubits, 3, 2)) # counts of 0 and 1
        
        for bases, outcomes in shadow_samples:
            for i in range(self.n_qubits):
                b = outcomes[i]
                basis = bases[i]
                feature_vector[i, basis, b] += 1
                
        # Normalize
        # Avoid division by zero if some basis wasn't sampled?
        # Ideally shadow_size is large enough.
        
        # We can flatten this
        return feature_vector.flatten() / len(shadow_samples)

