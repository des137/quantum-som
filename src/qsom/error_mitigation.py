"""
Error Mitigation Strategies for QSOM.

This module provides various error mitigation techniques for improving
the quality of quantum measurements on noisy hardware.

Techniques implemented:
- Zero Noise Extrapolation (ZNE)
- Probabilistic Error Cancellation (PEC) - framework
- Measurement Error Mitigation
- Twirled Readout Error Extinction (TREX)
- Dynamical Decoupling sequences
"""

from typing import List, Dict, Optional, Tuple, Callable, Any
import numpy as np

try:
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import XGate, YGate, ZGate
    from qiskit.transpiler import PassManager
    from qiskit.quantum_info import Operator
    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False


class MeasurementErrorMitigator:
    """
    Measurement error mitigation using calibration matrices.

    Corrects for readout errors by characterizing the confusion matrix
    and applying its inverse.
    """

    def __init__(self, n_qubits: int):
        """
        Initialize measurement error mitigator.

        Args:
            n_qubits: Number of qubits.
        """
        self.n_qubits = n_qubits
        self.calibration_matrix = None
        self.inverse_matrix = None

    def calibrate(
        self,
        backend: Any,
        shots: int = 8192
    ) -> np.ndarray:
        """
        Calibrate by measuring all computational basis states.

        Args:
            backend: Quantum backend to calibrate.
            shots: Number of shots per calibration circuit.

        Returns:
            Calibration matrix of shape (2^n, 2^n).
        """
        if not QISKIT_AVAILABLE:
            raise ImportError("Qiskit required for calibration")

        n_states = 2 ** self.n_qubits
        self.calibration_matrix = np.zeros((n_states, n_states))

        # Prepare each basis state and measure
        for state_idx in range(n_states):
            bitstring = format(state_idx, f'0{self.n_qubits}b')

            qc = QuantumCircuit(self.n_qubits)
            for i, bit in enumerate(bitstring[::-1]):
                if bit == '1':
                    qc.x(i)
            qc.measure_all()

            # Run circuit
            result = backend.run(qc)

            # Extract counts
            counts = self._extract_counts(result)

            # Fill calibration matrix column
            for measured_state, count in counts.items():
                measured_idx = int(measured_state, 2)
                self.calibration_matrix[measured_idx, state_idx] = count / shots

        # Compute pseudo-inverse for mitigation
        self.inverse_matrix = np.linalg.pinv(self.calibration_matrix)

        return self.calibration_matrix

    def _extract_counts(self, result: Any) -> Dict[str, int]:
        """Extract counts from backend result."""
        if hasattr(result, 'quasi_dists'):
            # V1 result
            dist = result.quasi_dists[0]
            return {
                format(k, f'0{self.n_qubits}b'): int(v * 1000)
                for k, v in dist.items()
            }
        elif hasattr(result, 'get_counts'):
            return result.get_counts()
        else:
            # V2 result
            try:
                data = result[0].data.meas
                counts = {}
                for bitstring in data.get_bitstrings():
                    counts[bitstring] = counts.get(bitstring, 0) + 1
                return counts
            except Exception:
                return {}

    def mitigate(self, counts: Dict[str, int]) -> Dict[str, float]:
        """
        Apply measurement error mitigation to counts.

        Args:
            counts: Raw measurement counts.

        Returns:
            Mitigated probability distribution.
        """
        if self.inverse_matrix is None:
            raise ValueError("Calibration required. Call calibrate() first.")

        # Convert counts to probability vector
        total = sum(counts.values())
        prob_vector = np.zeros(2 ** self.n_qubits)
        for bitstring, count in counts.items():
            idx = int(bitstring, 2)
            prob_vector[idx] = count / total

        # Apply inverse calibration matrix
        mitigated = self.inverse_matrix @ prob_vector

        # Clip negative probabilities and renormalize
        mitigated = np.maximum(mitigated, 0)
        mitigated /= mitigated.sum()

        # Convert back to dictionary
        result = {}
        for idx, prob in enumerate(mitigated):
            if prob > 1e-10:
                bitstring = format(idx, f'0{self.n_qubits}b')
                result[bitstring] = prob

        return result


class ZeroNoiseExtrapolator:
    """
    Zero Noise Extrapolation (ZNE) error mitigation.

    Runs circuits at multiple noise levels and extrapolates to zero noise.
    """

    def __init__(
        self,
        scale_factors: List[float] = None,
        extrapolation: str = 'linear'
    ):
        """
        Initialize ZNE mitigator.

        Args:
            scale_factors: Noise scale factors (e.g., [1, 2, 3]).
            extrapolation: Extrapolation method ('linear', 'polynomial', 'exponential').
        """
        self.scale_factors = scale_factors or [1.0, 2.0, 3.0]
        self.extrapolation = extrapolation

    def fold_circuit(
        self,
        circuit: 'QuantumCircuit',
        scale_factor: float
    ) -> 'QuantumCircuit':
        """
        Fold circuit to amplify noise by scale_factor.

        Implements unitary folding: U -> U (U† U)^n for integer n.

        Args:
            circuit: Original circuit.
            scale_factor: Target noise amplification factor.

        Returns:
            Folded circuit with amplified noise.
        """
        if not QISKIT_AVAILABLE:
            raise ImportError("Qiskit required")

        if scale_factor < 1:
            raise ValueError("Scale factor must be >= 1")

        if scale_factor == 1:
            return circuit.copy()

        # Number of full folds
        n_folds = int((scale_factor - 1) / 2)
        partial = (scale_factor - 1) / 2 - n_folds

        # Create folded circuit
        folded = circuit.copy()
        folded.barrier()

        for _ in range(n_folds):
            # Add U† U
            folded.compose(circuit.inverse(), inplace=True)
            folded.barrier()
            folded.compose(circuit, inplace=True)
            folded.barrier()

        # Handle partial fold if needed
        if partial > 0.1:
            # Approximate partial fold by folding subset of gates
            n_gates = circuit.size()
            n_fold_gates = int(n_gates * partial * 2)
            if n_fold_gates > 0:
                # Simplified: fold first n_fold_gates
                partial_circuit = QuantumCircuit(circuit.num_qubits)
                for i, (gate, qargs, cargs) in enumerate(circuit.data):
                    if i < n_fold_gates // 2:
                        partial_circuit.append(gate, qargs, cargs)

                folded.compose(partial_circuit.inverse(), inplace=True)
                folded.compose(partial_circuit, inplace=True)

        return folded

    def extrapolate(
        self,
        scale_factors: List[float],
        expectation_values: List[float]
    ) -> float:
        """
        Extrapolate to zero noise.

        Args:
            scale_factors: Noise scale factors used.
            expectation_values: Measured expectation values.

        Returns:
            Zero-noise extrapolated value.
        """
        scale_factors = np.array(scale_factors)
        expectation_values = np.array(expectation_values)

        if self.extrapolation == 'linear':
            # Linear fit and extrapolate to 0
            coeffs = np.polyfit(scale_factors, expectation_values, 1)
            return coeffs[1]  # y-intercept

        elif self.extrapolation == 'polynomial':
            # Polynomial fit (degree = n_points - 1)
            degree = min(len(scale_factors) - 1, 3)
            coeffs = np.polyfit(scale_factors, expectation_values, degree)
            poly = np.poly1d(coeffs)
            return poly(0)

        elif self.extrapolation == 'exponential':
            # Fit A * exp(-B * x) + C and extrapolate
            # Simplified: use log-linear fit
            try:
                log_vals = np.log(np.abs(expectation_values) + 1e-10)
                coeffs = np.polyfit(scale_factors, log_vals, 1)
                return np.exp(coeffs[1])
            except Exception:
                # Fallback to linear
                coeffs = np.polyfit(scale_factors, expectation_values, 1)
                return coeffs[1]

        else:
            raise ValueError(f"Unknown extrapolation method: {self.extrapolation}")

    def run_zne(
        self,
        circuit: 'QuantumCircuit',
        backend: Any,
        observable_fn: Callable,
        shots: int = 4096
    ) -> Tuple[float, Dict]:
        """
        Run ZNE workflow.

        Args:
            circuit: Circuit to run.
            backend: Quantum backend.
            observable_fn: Function to compute observable from counts.
            shots: Shots per circuit.

        Returns:
            Tuple of (mitigated_value, details_dict).
        """
        expectation_values = []

        for scale in self.scale_factors:
            folded = self.fold_circuit(circuit, scale)
            folded.measure_all()

            result = backend.run(folded)
            counts = self._extract_counts(result, circuit.num_qubits)
            exp_val = observable_fn(counts)
            expectation_values.append(exp_val)

        mitigated = self.extrapolate(self.scale_factors, expectation_values)

        details = {
            'scale_factors': self.scale_factors,
            'expectation_values': expectation_values,
            'mitigated_value': mitigated,
            'extrapolation': self.extrapolation
        }

        return mitigated, details

    def _extract_counts(self, result: Any, n_qubits: int) -> Dict[str, int]:
        """Extract counts from result."""
        if hasattr(result, 'quasi_dists'):
            dist = result.quasi_dists[0]
            return {
                format(k, f'0{n_qubits}b'): int(v * 1000)
                for k, v in dist.items()
            }
        return {}


class DynamicalDecoupling:
    """
    Dynamical Decoupling sequences for coherence protection.

    Implements common DD sequences:
    - XY4: X-Y-X-Y
    - CPMG: X-X
    - Uhrig: Optimized timing
    """

    # Common DD sequences
    SEQUENCES = {
        'xy4': ['X', 'Y', 'X', 'Y'],
        'cpmg': ['X', 'X'],
        'xy8': ['X', 'Y', 'X', 'Y', 'Y', 'X', 'Y', 'X'],
    }

    def __init__(self, sequence: str = 'xy4'):
        """
        Initialize DD with specified sequence.

        Args:
            sequence: DD sequence name ('xy4', 'cpmg', 'xy8').
        """
        if sequence not in self.SEQUENCES:
            raise ValueError(f"Unknown sequence: {sequence}. "
                           f"Available: {list(self.SEQUENCES.keys())}")

        self.sequence = sequence
        self.gates = self.SEQUENCES[sequence]

    def get_gate(self, gate_name: str) -> 'XGate':
        """Get Qiskit gate object."""
        if not QISKIT_AVAILABLE:
            raise ImportError("Qiskit required")

        if gate_name == 'X':
            return XGate()
        elif gate_name == 'Y':
            return YGate()
        elif gate_name == 'Z':
            return ZGate()
        else:
            raise ValueError(f"Unknown gate: {gate_name}")

    def insert_dd(
        self,
        circuit: 'QuantumCircuit',
        idle_qubits: Optional[List[int]] = None
    ) -> 'QuantumCircuit':
        """
        Insert DD sequence on idle qubits.

        This is a simplified implementation. For production use,
        consider using Qiskit's built-in DD passes.

        Args:
            circuit: Input circuit.
            idle_qubits: Qubits to apply DD to. If None, applies to all.

        Returns:
            Circuit with DD inserted.
        """
        if not QISKIT_AVAILABLE:
            raise ImportError("Qiskit required")

        if idle_qubits is None:
            idle_qubits = list(range(circuit.num_qubits))

        # Create new circuit with DD
        dd_circuit = circuit.copy()

        # Add DD sequence to each idle qubit at the end
        # (Proper implementation would insert during idle periods)
        for qubit in idle_qubits:
            dd_circuit.barrier([qubit])
            for gate_name in self.gates:
                gate = self.get_gate(gate_name)
                dd_circuit.append(gate, [qubit])

        return dd_circuit


class TwirledReadoutMitigator:
    """
    Twirled Readout Error eXtinction (TREX).

    Randomizes readout errors by applying random Pauli twirls
    before measurement.
    """

    def __init__(self, n_qubits: int, n_twirls: int = 10):
        """
        Initialize TREX mitigator.

        Args:
            n_qubits: Number of qubits.
            n_twirls: Number of random twirls to average over.
        """
        self.n_qubits = n_qubits
        self.n_twirls = n_twirls

    def generate_twirl(self) -> List[str]:
        """Generate random Pauli twirl for each qubit."""
        paulis = ['I', 'X', 'Y', 'Z']
        return [np.random.choice(paulis) for _ in range(self.n_qubits)]

    def apply_twirl(
        self,
        circuit: 'QuantumCircuit',
        twirl: List[str]
    ) -> 'QuantumCircuit':
        """
        Apply Pauli twirl to circuit before measurement.

        Args:
            circuit: Input circuit.
            twirl: Pauli operators to apply to each qubit.

        Returns:
            Twirled circuit.
        """
        if not QISKIT_AVAILABLE:
            raise ImportError("Qiskit required")

        twirled = circuit.copy()

        for i, pauli in enumerate(twirl):
            if pauli == 'X':
                twirled.x(i)
            elif pauli == 'Y':
                twirled.y(i)
            elif pauli == 'Z':
                twirled.z(i)
            # 'I' does nothing

        return twirled

    def untwirl_counts(
        self,
        counts: Dict[str, int],
        twirl: List[str]
    ) -> Dict[str, int]:
        """
        Untwirl measurement results.

        Args:
            counts: Raw counts from twirled circuit.
            twirl: Pauli twirl that was applied.

        Returns:
            Untwirled counts.
        """
        untwirled = {}

        for bitstring, count in counts.items():
            # X and Y flip bits, Z and I don't
            new_bits = list(bitstring[::-1])  # Reverse for qubit ordering

            for i, pauli in enumerate(twirl):
                if pauli in ['X', 'Y']:
                    new_bits[i] = '1' if new_bits[i] == '0' else '0'

            new_bitstring = ''.join(new_bits[::-1])
            untwirled[new_bitstring] = untwirled.get(new_bitstring, 0) + count

        return untwirled

    def mitigate(
        self,
        circuit: 'QuantumCircuit',
        backend: Any,
        shots_per_twirl: int = 1000
    ) -> Dict[str, float]:
        """
        Run TREX mitigation.

        Args:
            circuit: Circuit to run.
            backend: Quantum backend.
            shots_per_twirl: Shots per twirl.

        Returns:
            Mitigated probability distribution.
        """
        total_counts = {}

        for _ in range(self.n_twirls):
            twirl = self.generate_twirl()
            twirled_circuit = self.apply_twirl(circuit, twirl)
            twirled_circuit.measure_all()

            result = backend.run(twirled_circuit)

            # Extract and untwirl counts
            raw_counts = self._extract_counts(result)
            untwirled = self.untwirl_counts(raw_counts, twirl)

            for bitstring, count in untwirled.items():
                total_counts[bitstring] = total_counts.get(bitstring, 0) + count

        # Normalize
        total = sum(total_counts.values())
        return {k: v / total for k, v in total_counts.items()}

    def _extract_counts(self, result: Any) -> Dict[str, int]:
        """Extract counts from result."""
        if hasattr(result, 'quasi_dists'):
            dist = result.quasi_dists[0]
            return {
                format(k, f'0{self.n_qubits}b'): int(v * 1000)
                for k, v in dist.items()
            }
        return {}
