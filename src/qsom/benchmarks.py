"""
Benchmarking Utilities for QSOM.

This module provides tools for benchmarking QSOM performance
across different backends, configurations, and datasets.
"""

import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import json
from pathlib import Path
import numpy as np

try:
    from qiskit import QuantumCircuit
    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False


@dataclass
class BenchmarkResult:
    """Container for benchmark results."""
    name: str
    backend_type: str
    n_qubits: int
    n_samples: int
    shadow_size: int
    grid_size: Tuple[int, int]

    # Timing results (seconds)
    shadow_generation_time: float = 0.0
    som_training_time: float = 0.0
    total_time: float = 0.0

    # Quality metrics
    final_quantization_error: float = 0.0
    convergence_iterations: int = 0

    # Resource usage
    peak_memory_mb: float = 0.0
    n_circuits_executed: int = 0

    # Metadata
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    extra_info: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'name': self.name,
            'backend_type': self.backend_type,
            'n_qubits': self.n_qubits,
            'n_samples': self.n_samples,
            'shadow_size': self.shadow_size,
            'grid_size': self.grid_size,
            'shadow_generation_time': self.shadow_generation_time,
            'som_training_time': self.som_training_time,
            'total_time': self.total_time,
            'final_quantization_error': self.final_quantization_error,
            'convergence_iterations': self.convergence_iterations,
            'peak_memory_mb': self.peak_memory_mb,
            'n_circuits_executed': self.n_circuits_executed,
            'timestamp': self.timestamp,
            'extra_info': self.extra_info
        }


class QSOMBenchmark:
    """
    Benchmark suite for QSOM.

    Provides standardized benchmarks for comparing:
    - Simulator vs hardware performance
    - Different shadow sizes
    - Different SOM configurations
    - Scaling behavior
    """

    def __init__(self, results_dir: Optional[str] = None):
        """
        Initialize benchmark suite.

        Args:
            results_dir: Directory to save results.
        """
        self.results: List[BenchmarkResult] = []
        self.results_dir = Path(results_dir) if results_dir else None

        if self.results_dir:
            self.results_dir.mkdir(parents=True, exist_ok=True)

    def benchmark_shadow_generation(
        self,
        backend: Any,
        n_qubits: int,
        n_samples: int,
        shadow_size: int,
        name: str = "shadow_generation"
    ) -> BenchmarkResult:
        """
        Benchmark shadow generation performance.

        Args:
            backend: QuantumBackend to use.
            n_qubits: Number of qubits.
            n_samples: Number of quantum states.
            shadow_size: Shadow size per state.
            name: Benchmark name.

        Returns:
            BenchmarkResult with timing and quality metrics.
        """
        from .shadows import ClassicalShadow
        from .utils import generate_quantum_states

        result = BenchmarkResult(
            name=name,
            backend_type='simulator' if hasattr(backend, 'use_simulator') and backend.use_simulator else 'hardware',
            n_qubits=n_qubits,
            n_samples=n_samples,
            shadow_size=shadow_size,
            grid_size=(0, 0)
        )

        # Generate states
        states = generate_quantum_states(n_samples, n_qubits, state_type='random')

        # Create circuits
        circuits = []
        for state in states:
            qc = QuantumCircuit(n_qubits)
            qc.prepare_state(state, range(n_qubits))
            circuits.append(qc)

        # Initialize shadow generator
        shadow_gen = ClassicalShadow(
            n_qubits=n_qubits,
            backend=backend,
            shadow_size=shadow_size
        )

        # Time shadow generation
        start_time = time.time()

        features = []
        for qc in circuits:
            samples = shadow_gen.generate(qc)
            feat = shadow_gen.shadow_to_feature_vector(samples)
            features.append(feat)

        result.shadow_generation_time = time.time() - start_time
        result.total_time = result.shadow_generation_time
        result.n_circuits_executed = n_samples * shadow_size

        # Memory estimate
        features_array = np.array(features)
        result.peak_memory_mb = features_array.nbytes / (1024 * 1024)

        result.extra_info['feature_dim'] = features_array.shape[1]

        self.results.append(result)
        return result

    def benchmark_full_pipeline(
        self,
        backend: Any,
        n_qubits: int,
        n_samples: int,
        shadow_size: int,
        grid_size: Tuple[int, int],
        n_iterations: int,
        name: str = "full_pipeline"
    ) -> BenchmarkResult:
        """
        Benchmark complete QSOM pipeline.

        Args:
            backend: QuantumBackend to use.
            n_qubits: Number of qubits.
            n_samples: Number of quantum states.
            shadow_size: Shadow size per state.
            grid_size: SOM grid size.
            n_iterations: SOM training iterations.
            name: Benchmark name.

        Returns:
            BenchmarkResult with comprehensive metrics.
        """
        from .shadows import ClassicalShadow
        from .som import QuantumSOM
        from .utils import generate_quantum_states

        result = BenchmarkResult(
            name=name,
            backend_type='simulator' if hasattr(backend, 'use_simulator') and backend.use_simulator else 'hardware',
            n_qubits=n_qubits,
            n_samples=n_samples,
            shadow_size=shadow_size,
            grid_size=grid_size
        )

        total_start = time.time()

        # Generate states
        states = generate_quantum_states(n_samples, n_qubits, state_type='random')

        circuits = []
        for state in states:
            qc = QuantumCircuit(n_qubits)
            qc.prepare_state(state, range(n_qubits))
            circuits.append(qc)

        # Shadow generation
        shadow_gen = ClassicalShadow(
            n_qubits=n_qubits,
            backend=backend,
            shadow_size=shadow_size
        )

        shadow_start = time.time()
        features = shadow_gen.generate_features_batch(circuits)
        result.shadow_generation_time = time.time() - shadow_start

        # SOM training
        som = QuantumSOM(
            grid_size=grid_size,
            input_dim=features.shape[1],
            n_iterations=n_iterations,
            distance_metric='euclidean'
        )

        train_start = time.time()
        som.train(features, verbose=False, track_errors=True)
        result.som_training_time = time.time() - train_start

        result.total_time = time.time() - total_start
        result.n_circuits_executed = n_samples * shadow_size

        # Quality metrics
        if som.quantization_errors:
            result.final_quantization_error = som.quantization_errors[-1]
            result.convergence_iterations = len(som.quantization_errors) * 100

        # Memory
        result.peak_memory_mb = (features.nbytes + som.weights.nbytes) / (1024 * 1024)

        result.extra_info = {
            'feature_dim': features.shape[1],
            'n_neurons': grid_size[0] * grid_size[1],
            'iterations_per_second': n_iterations / result.som_training_time
        }

        self.results.append(result)
        return result

    def benchmark_scaling(
        self,
        backend: Any,
        qubit_range: List[int],
        n_samples: int = 20,
        shadow_size: int = 30,
        name: str = "scaling"
    ) -> List[BenchmarkResult]:
        """
        Benchmark scaling behavior with number of qubits.

        Args:
            backend: QuantumBackend to use.
            qubit_range: List of qubit counts to test.
            n_samples: Samples per configuration.
            shadow_size: Shadow size.
            name: Benchmark name prefix.

        Returns:
            List of BenchmarkResults.
        """
        results = []

        for n_qubits in qubit_range:
            print(f"Benchmarking {n_qubits} qubits...")
            result = self.benchmark_shadow_generation(
                backend=backend,
                n_qubits=n_qubits,
                n_samples=n_samples,
                shadow_size=shadow_size,
                name=f"{name}_{n_qubits}q"
            )
            results.append(result)

        return results

    def compare_backends(
        self,
        backends: Dict[str, Any],
        n_qubits: int = 3,
        n_samples: int = 20,
        shadow_size: int = 30,
        grid_size: Tuple[int, int] = (5, 5),
        n_iterations: int = 200
    ) -> Dict[str, BenchmarkResult]:
        """
        Compare performance across different backends.

        Args:
            backends: Dictionary of {name: backend}.
            n_qubits: Number of qubits.
            n_samples: Number of samples.
            shadow_size: Shadow size.
            grid_size: SOM grid size.
            n_iterations: Training iterations.

        Returns:
            Dictionary of results by backend name.
        """
        results = {}

        for name, backend in backends.items():
            print(f"Benchmarking backend: {name}")
            try:
                result = self.benchmark_full_pipeline(
                    backend=backend,
                    n_qubits=n_qubits,
                    n_samples=n_samples,
                    shadow_size=shadow_size,
                    grid_size=grid_size,
                    n_iterations=n_iterations,
                    name=f"backend_{name}"
                )
                results[name] = result
            except Exception as e:
                print(f"  Error: {e}")
                results[name] = None

        return results

    def save_results(self, filename: str = "benchmark_results.json") -> None:
        """
        Save all results to JSON file.

        Args:
            filename: Output filename.
        """
        if self.results_dir:
            filepath = self.results_dir / filename
        else:
            filepath = Path(filename)

        data = [r.to_dict() for r in self.results]

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"Results saved to {filepath}")

    def print_summary(self) -> None:
        """Print summary of benchmark results."""
        print("\n" + "=" * 70)
        print("QSOM Benchmark Summary")
        print("=" * 70)

        for result in self.results:
            print(f"\n{result.name}")
            print("-" * 40)
            print(f"  Backend: {result.backend_type}")
            print(f"  Qubits: {result.n_qubits}, Samples: {result.n_samples}")
            print(f"  Shadow size: {result.shadow_size}")
            print(f"  Shadow generation: {result.shadow_generation_time:.3f}s")
            print(f"  SOM training: {result.som_training_time:.3f}s")
            print(f"  Total time: {result.total_time:.3f}s")
            print(f"  Final QE: {result.final_quantization_error:.4f}")
            print(f"  Circuits executed: {result.n_circuits_executed}")

        print("\n" + "=" * 70)


def run_standard_benchmarks(backend: Any) -> Dict[str, BenchmarkResult]:
    """
    Run a standard suite of benchmarks.

    Args:
        backend: QuantumBackend to benchmark.

    Returns:
        Dictionary of benchmark results.
    """
    bench = QSOMBenchmark()

    results = {}

    # Small scale test
    print("Running small-scale benchmark...")
    results['small'] = bench.benchmark_full_pipeline(
        backend=backend,
        n_qubits=2,
        n_samples=10,
        shadow_size=20,
        grid_size=(4, 4),
        n_iterations=100,
        name="small_scale"
    )

    # Medium scale test
    print("Running medium-scale benchmark...")
    results['medium'] = bench.benchmark_full_pipeline(
        backend=backend,
        n_qubits=3,
        n_samples=30,
        shadow_size=50,
        grid_size=(6, 6),
        n_iterations=300,
        name="medium_scale"
    )

    # Shadow size comparison
    print("Running shadow size comparison...")
    for shadow_size in [10, 30, 100]:
        results[f'shadow_{shadow_size}'] = bench.benchmark_shadow_generation(
            backend=backend,
            n_qubits=3,
            n_samples=20,
            shadow_size=shadow_size,
            name=f"shadow_size_{shadow_size}"
        )

    bench.print_summary()

    return results
