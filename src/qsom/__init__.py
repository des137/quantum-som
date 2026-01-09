"""
QSOM: Quantum Self-Organizing Maps

A Python library for exploring quantum state spaces using classical shadows
and self-organizing maps.

Key Features:
- Classical shadows generation for efficient quantum state representation
- Self-organizing map implementation with quantum-specific distance metrics
- Visualization tools for exploring quantum state space
- Qiskit 2.0+ integration for real quantum measurements
"""

__version__ = "0.3.0"
__author__ = "Amol Deshmukh"

from .backend import QuantumBackend
from .shadows import ClassicalShadow
from .som import QuantumSOM
from .utils import generate_quantum_states, angle_encoding, amplitude_encoding

# Visualization functions (optional import)
try:
    from .visualization import (
        create_interactive_umatrix,
        create_training_animation,
        create_3d_umatrix,
        create_component_planes,
        create_hit_histogram,
        create_training_progress,
    )
    _VIZ_AVAILABLE = True
except ImportError:
    _VIZ_AVAILABLE = False

# Sparse matrix support (optional - requires scipy)
try:
    from .sparse import SparseClassicalShadow, SparseSOM, estimate_sparsity
    _SPARSE_AVAILABLE = True
except ImportError:
    _SPARSE_AVAILABLE = False

# Error mitigation (optional - requires qiskit)
try:
    from .error_mitigation import (
        MeasurementErrorMitigator,
        ZeroNoiseExtrapolator,
        DynamicalDecoupling,
        TwirledReadoutMitigator,
    )
    _ERROR_MITIGATION_AVAILABLE = True
except ImportError:
    _ERROR_MITIGATION_AVAILABLE = False

# Benchmarking utilities
try:
    from .benchmarks import QSOMBenchmark, BenchmarkResult, run_standard_benchmarks
    _BENCHMARKS_AVAILABLE = True
except ImportError:
    _BENCHMARKS_AVAILABLE = False

# GPU acceleration (optional - requires jax or cupy)
try:
    from .gpu import (
        GPUAcceleratedSOM,
        GPUAcceleratedShadowProcessor,
        get_backend as get_gpu_backend,
        gpu_available,
        create_optimized_som,
    )
    _GPU_AVAILABLE = True
except ImportError:
    _GPU_AVAILABLE = False

# Clifford shadows (optional - requires qiskit)
try:
    from .clifford_shadows import (
        CliffordShadow,
        LocalCliffordShadow,
        compare_shadow_methods,
    )
    _CLIFFORD_AVAILABLE = True
except ImportError:
    _CLIFFORD_AVAILABLE = False

# IBM Quantum job management (optional - requires qiskit-ibm-runtime)
try:
    from .ibm_jobs import (
        IBMJobManager,
        ShadowJobRunner,
        JobInfo,
        estimate_job_time,
    )
    _IBM_JOBS_AVAILABLE = True
except ImportError:
    _IBM_JOBS_AVAILABLE = False

__all__ = [
    "QuantumBackend",
    "ClassicalShadow",
    "QuantumSOM",
    "generate_quantum_states",
    "angle_encoding",
    "amplitude_encoding",
]

if _VIZ_AVAILABLE:
    __all__.extend([
        "create_interactive_umatrix",
        "create_training_animation",
        "create_3d_umatrix",
        "create_component_planes",
        "create_hit_histogram",
        "create_training_progress",
    ])

if _SPARSE_AVAILABLE:
    __all__.extend([
        "SparseClassicalShadow",
        "SparseSOM",
        "estimate_sparsity",
    ])

if _ERROR_MITIGATION_AVAILABLE:
    __all__.extend([
        "MeasurementErrorMitigator",
        "ZeroNoiseExtrapolator",
        "DynamicalDecoupling",
        "TwirledReadoutMitigator",
    ])

if _BENCHMARKS_AVAILABLE:
    __all__.extend([
        "QSOMBenchmark",
        "BenchmarkResult",
        "run_standard_benchmarks",
    ])

if _GPU_AVAILABLE:
    __all__.extend([
        "GPUAcceleratedSOM",
        "GPUAcceleratedShadowProcessor",
        "get_gpu_backend",
        "gpu_available",
        "create_optimized_som",
    ])

if _CLIFFORD_AVAILABLE:
    __all__.extend([
        "CliffordShadow",
        "LocalCliffordShadow",
        "compare_shadow_methods",
    ])

if _IBM_JOBS_AVAILABLE:
    __all__.extend([
        "IBMJobManager",
        "ShadowJobRunner",
        "JobInfo",
        "estimate_job_time",
    ])
