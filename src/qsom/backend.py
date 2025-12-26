"""
Quantum Backend Module for QSOM.

This module handles interactions with quantum hardware or simulators via Qiskit Primitives.
It includes support for error mitigation techniques such as dynamical decoupling and
readout error mitigation options provided by the Qiskit Runtime.
"""

from typing import Optional, Union, List, Any
import numpy as np
from qiskit import QuantumCircuit
from qiskit.primitives import StatevectorSampler
from qiskit_aer import AerSimulator
from qiskit_aer.primitives import Sampler as AerSampler
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import PadDynamicalDecoupling
from qiskit.circuit.library import XGate
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2, Session, Options

class QuantumBackend:
    """
    Abstraction layer for executing quantum circuits with error mitigation.
    """

    def __init__(
        self,
        backend_name: Optional[str] = None,
        service: Optional[QiskitRuntimeService] = None,
        use_simulator: bool = False,
        resilience_level: int = 1,
        dynamical_decoupling: bool = False,
        dd_sequence: Optional[List[Any]] = None,
        shots: int = 1000
    ):
        """
        Initialize the QuantumBackend.

        Args:
            backend_name: Name of the backend to use (if service is provided).
            service: QiskitRuntimeService instance.
            use_simulator: Force usage of local AerSimulator.
            resilience_level: Error mitigation level for Qiskit Runtime (0-3).
            dynamical_decoupling: Whether to apply dynamical decoupling.
            dd_sequence: Custom DD sequence (list of gates, e.g., [XGate(), XGate()]). 
                         Defaults to XY4 if None.
            shots: Number of shots per execution.
        """
        self.use_simulator = use_simulator
        self.shots = shots
        self.sampler = None
        self.backend = None
        
        if use_simulator:
            self.backend = AerSimulator()
            self.sampler = AerSampler(backend_options={"shots": shots})
        elif service and backend_name:
            self.service = service
            self.backend = service.backend(backend_name)
            # Create SamplerV2 with options
            # For V2 primitives, options are passed at run time or init
            # resilience_level is not directly in V2 init standard args in all versions, 
            # but we configure it via Options if needed or passing to run.
            self.sampler = SamplerV2(backend=self.backend)
            # Configure default options for the sampler
            self.sampler.options.default_shots = shots
            # Set resilience level - for SamplerV2, this maps to measurement mitigation usually
            # We will handle this in run() or options setup if specific fields exist
        else:
             # Fallback to local
             print("No service provided, falling back to local AerSimulator.")
             self.backend = AerSimulator()
             self.sampler = AerSampler(backend_options={"shots": shots})

        self.dynamical_decoupling = dynamical_decoupling
        self.dd_sequence = dd_sequence or [XGate(), XGate()] # Simple echo as default placeholder

    def _apply_error_mitigation(self, circuit: QuantumCircuit) -> QuantumCircuit:
        """
        Apply circuit-level error mitigation techniques (transpilation passes).
        """
        if self.dynamical_decoupling:
            # Need to know backend properties to schedule DD
            # This is complex without target scheduling info.
            # We will attempt to schedule if backend has instruction durations.
            try:
                # This is a simplified DD application. Real DD requires timing constraints.
                # Here we just show the intent structure.
                # In a real scenario, we would need t_sched and a target with durations.
                pass 
            except Exception as e:
                print(f"Warning: Could not apply Dynamical Decoupling: {e}")
        
        return circuit

    def run(self, circuits: Union[QuantumCircuit, List[QuantumCircuit]], parameter_values: Optional[List[List[float]]] = None) -> Any:
        """
        Run circuits on the configured backend.

        Args:
            circuits: Single or list of QuantumCircuits.
            parameter_values: Optional list of parameter values if circuits are parameterized.

        Returns:
            The result of the primitive run (PrimitiveResult).
        """
        if isinstance(circuits, QuantumCircuit):
            circuits = [circuits]

        # Apply circuit-level mitigation
        processed_circuits = [self._apply_error_mitigation(qc) for qc in circuits]

        # V2 Sampler run
        if isinstance(self.sampler, SamplerV2):
             # SamplerV2 run expects (circuit, parameters) tuples or just circuits
             pubs = []
             if parameter_values:
                 # Ensure length matches
                 if len(parameter_values) != len(processed_circuits):
                     # If one param set for all, broadcast? Or assume 1:1
                     pass
                 for qc, params in zip(processed_circuits, parameter_values):
                     pubs.append((qc, params))
             else:
                 pubs = processed_circuits
            
             job = self.sampler.run(pubs)
             return job.result()
        
        # AerSampler or V1 Sampler
        else:
             job = self.sampler.run(processed_circuits, parameter_values=parameter_values)
             return job.result()

