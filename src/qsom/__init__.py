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

__version__ = "0.2.0"
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
