Quick Start
===========

This guide walks through a complete QSOM workflow: generating quantum states,
creating classical shadows, training a SOM, and visualizing the results.

Step 1: Generate Quantum States
-------------------------------

.. code-block:: python

   from qsom import generate_quantum_states

   # Generate 100 random quantum states on 3 qubits
   states = generate_quantum_states(
       n_states=100,
       n_qubits=3,
       state_type='random'  # Options: 'random', 'ghz', 'w', 'product', 'cluster'
   )

   print(f"Generated {len(states)} states of dimension {len(states[0])}")

Step 2: Initialize Backend and Shadow Generator
-----------------------------------------------

.. code-block:: python

   from qsom import QuantumBackend, ClassicalShadow

   # Use local simulator (fast, no IBM account needed)
   backend = QuantumBackend(use_simulator=True)

   # Create shadow generator
   shadow_gen = ClassicalShadow(
       n_qubits=3,
       backend=backend,
       shadow_size=50  # Number of random measurements per state
   )

Step 3: Create Circuits and Generate Features
---------------------------------------------

.. code-block:: python

   from qiskit import QuantumCircuit

   # Create circuits that prepare each state
   circuits = []
   for state in states:
       qc = QuantumCircuit(3)
       qc.prepare_state(state, range(3))
       circuits.append(qc)

   # Generate classical shadow features for all circuits
   features = shadow_gen.generate_features_batch(circuits)

   print(f"Feature matrix shape: {features.shape}")
   # Output: (100, 18) - 100 states, 18-dimensional feature vectors

Step 4: Train the SOM
---------------------

.. code-block:: python

   from qsom import QuantumSOM

   # Create and train SOM
   som = QuantumSOM(
       grid_size=(10, 10),       # 10x10 neuron grid
       input_dim=features.shape[1],
       learning_rate=0.5,
       sigma=1.0,                # Neighborhood radius
       n_iterations=1000,
       distance_metric='quantum'  # Quantum-aware distance
   )

   som.train(features, verbose=True, track_errors=True)

Step 5: Visualize Results
-------------------------

.. code-block:: python

   # Basic visualization
   fig = som.visualize(data=features, title="Quantum State Space")

For interactive visualization:

.. code-block:: python

   from qsom import create_interactive_umatrix, create_3d_umatrix

   # Interactive U-matrix (requires plotly)
   fig = create_interactive_umatrix(som, data=features)
   fig.show()

   # 3D surface plot
   fig3d = create_3d_umatrix(som)
   fig3d.show()

Complete Example
----------------

Here's the full workflow in one script:

.. code-block:: python

   from qsom import (
       QuantumBackend, ClassicalShadow, QuantumSOM,
       generate_quantum_states
   )
   from qiskit import QuantumCircuit

   # 1. Generate states
   states = generate_quantum_states(100, n_qubits=3, state_type='random')

   # 2. Setup backend and shadows
   backend = QuantumBackend(use_simulator=True)
   shadow_gen = ClassicalShadow(n_qubits=3, backend=backend, shadow_size=50)

   # 3. Create circuits and features
   circuits = []
   for state in states:
       qc = QuantumCircuit(3)
       qc.prepare_state(state, range(3))
       circuits.append(qc)

   features = shadow_gen.generate_features_batch(circuits)

   # 4. Train SOM
   som = QuantumSOM(
       grid_size=(10, 10),
       input_dim=features.shape[1],
       n_iterations=1000
   )
   som.train(features, verbose=True)

   # 5. Visualize
   fig = som.visualize(data=features)

Next Steps
----------

* :doc:`theory` - Learn the mathematical foundations
* :doc:`api/shadows` - Deep dive into classical shadows
* :doc:`api/som` - Explore SOM configuration options
* :doc:`api/ibm_jobs` - Run on real quantum hardware
