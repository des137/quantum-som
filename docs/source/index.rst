QSOM: Quantum Self-Organizing Maps
===================================

.. image:: https://img.shields.io/badge/python-3.9+-blue.svg
   :target: https://www.python.org/downloads/

.. image:: https://img.shields.io/badge/License-MIT-yellow.svg
   :target: https://opensource.org/licenses/MIT

**QSOM** is a Python library for exploring quantum state spaces using classical
shadows and self-organizing maps.

Overview
--------

QSOM combines two powerful techniques:

* **Classical Shadows**: Efficient representation of quantum states using
  randomized measurements (Huang et al., 2020)
* **Self-Organizing Maps (SOM)**: Unsupervised neural network for
  dimensionality reduction and visualization (Kohonen, 1982)

Together, they enable exploration and visualization of high-dimensional quantum
state spaces on both classical simulators and real quantum hardware.

Features
--------

* Classical shadows with Pauli and Clifford bases
* Quantum SOM with multiple distance metrics
* Interactive Plotly visualizations
* GPU acceleration with JAX/CuPy
* IBM Quantum hardware integration
* Error mitigation strategies

Quick Start
-----------

.. code-block:: python

   from qsom import QuantumBackend, ClassicalShadow, QuantumSOM
   from qsom import generate_quantum_states
   from qiskit import QuantumCircuit

   # Generate quantum states
   states = generate_quantum_states(100, n_qubits=3, state_type='random')

   # Initialize backend and shadow generator
   backend = QuantumBackend(use_simulator=True)
   shadow_gen = ClassicalShadow(n_qubits=3, backend=backend, shadow_size=50)

   # Generate features from quantum states
   circuits = []
   for state in states:
       qc = QuantumCircuit(3)
       qc.prepare_state(state, range(3))
       circuits.append(qc)

   features = shadow_gen.generate_features_batch(circuits)

   # Train SOM and visualize
   som = QuantumSOM(grid_size=(10, 10), input_dim=features.shape[1])
   som.train(features)
   som.visualize()

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   installation
   quickstart
   theory

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/backend
   api/shadows
   api/som
   api/visualization
   api/error_mitigation
   api/gpu
   api/ibm_jobs

.. toctree::
   :maxdepth: 1
   :caption: Additional

   changelog
   contributing

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
