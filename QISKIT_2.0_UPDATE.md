# Qiskit 2.0+ Integration Update

This document describes the updates made to integrate Qiskit 2.0+ into the Quantum Self-Organizing Maps implementation.

## Key Changes

### 1. Updated Imports

The code now uses Qiskit 2.0+ APIs:

```python
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, DensityMatrix, SparsePauliOp, Pauli
from qiskit.primitives import Sampler, StatevectorSampler
from qiskit_aer import AerSimulator
from qiskit_aer.primitives import Sampler as AerSampler
```

### 2. Classical Shadows with Qiskit

The `ClassicalShadow` class now:
- Uses Qiskit's `Statevector` objects for quantum states
- Performs actual quantum measurements using `AerSampler` (Qiskit 2.0+ primitives)
- Supports both Qiskit-based and classical fallback modes

**Key Methods:**
- `measure_with_qiskit()`: Uses Qiskit 2.0+ Sampler for measurements
- `_pauli_to_rotation_gates()`: Converts Pauli measurements to rotation gates
- `generate_shadow_from_statevector()`: Generates shadows from Qiskit Statevectors

### 3. Enhanced Shadow Generation

The `EnhancedClassicalShadow` class:
- Uses Qiskit for inverse channel reconstruction
- Implements proper quantum state preparation and measurement
- Supports both Pauli and Clifford shadow types (framework ready)

### 4. Quantum State Generation

The `generate_quantum_states()` function now:
- Returns Qiskit `Statevector` objects by default
- Supports multiple state types: random, Bell, GHZ, W states
- Uses `Statevector.random()` for random state generation
- Falls back to numpy arrays if Qiskit is not available

### 5. Primitives-Based Architecture

The implementation uses Qiskit 2.0+'s primitives system:
- **Sampler**: For quantum measurements
- **StatevectorSampler**: For statevector-based operations
- **AerSimulator**: For simulation backend

## Usage Example

```python
from qiskit.quantum_info import Statevector
from qsom_implementation import ClassicalShadow, QuantumSOM

# Generate quantum state using Qiskit
state = Statevector.random(2**3)  # 3-qubit random state

# Generate classical shadow with Qiskit
shadow_gen = ClassicalShadow(n_qubits=3, n_shots=1000, use_qiskit=True)
shadow = shadow_gen.generate_shadow(state)

# Train SOM
qsom = QuantumSOM(grid_size=(10, 10), input_dim=shadow.shape[0])
qsom.train(np.array([shadow]), n_iterations=1000)
```

## Qiskit 2.0+ Features Used

1. **New Primitives API**: Uses `Sampler` instead of deprecated `execute()`
2. **Statevector API**: Uses `Statevector` objects with `prepare_state()` method
3. **QuantumCircuit**: Updated circuit construction with new API
4. **AerSimulator**: Modern simulator interface

## Backward Compatibility

The code maintains backward compatibility:
- Falls back to classical simulation if Qiskit is not available
- Supports numpy arrays as input (converts to Statevector)
- Graceful error handling for missing Qiskit installation

## Installation

```bash
pip install qiskit>=2.0.0 qiskit-aer>=0.15.0
```

## Benefits of Qiskit 2.0+ Integration

1. **Real Quantum Measurements**: Actual quantum circuit execution
2. **Better Performance**: Optimized primitives system
3. **Modern API**: Cleaner, more intuitive interface
4. **Extensibility**: Easy to extend to hardware backends
5. **Standardization**: Follows Qiskit ecosystem best practices

## Migration Notes

If migrating from the previous version:

1. **State Objects**: Convert numpy arrays to `Statevector` objects
2. **Measurement**: Use `Sampler` instead of `execute()`
3. **Circuit Construction**: Use `prepare_state()` for state preparation
4. **Results**: Access results via `quasi_dists` instead of `get_counts()`

## Future Enhancements

Potential improvements using Qiskit 2.0+ features:

1. **Hardware Backends**: Easy integration with real quantum hardware
2. **Estimator Primitive**: For observable estimation from shadows
3. **Runtime Services**: Cloud-based quantum computing
4. **Optimization**: Use Qiskit's optimization modules for SOM training

