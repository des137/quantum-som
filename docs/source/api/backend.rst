Quantum Backend
===============

.. automodule:: qsom.backend
   :members:
   :undoc-members:
   :show-inheritance:

Overview
--------

The :class:`QuantumBackend` class provides a unified interface for running
quantum circuits on different backends:

* **Local Simulators**: Qiskit Aer for fast local simulation
* **IBM Quantum Hardware**: Real quantum processors via IBM Quantum
* **Mock Backend**: For testing without quantum resources

Basic Usage
-----------

Using a Simulator
~~~~~~~~~~~~~~~~~

.. code-block:: python

   from qsom import QuantumBackend

   # Create simulator backend
   backend = QuantumBackend(use_simulator=True)

   # Run a circuit
   result = backend.run(circuit)

Using IBM Quantum Hardware
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from qsom import QuantumBackend
   from qiskit_ibm_runtime import QiskitRuntimeService

   # Initialize IBM Quantum service
   service = QiskitRuntimeService(channel="ibm_quantum")

   # Create hardware backend
   backend = QuantumBackend(
       backend_name="ibm_brisbane",
       service=service,
       resilience_level=1
   )

Configuration Options
---------------------

.. list-table::
   :header-rows: 1
   :widths: 20 15 65

   * - Parameter
     - Default
     - Description
   * - ``backend_name``
     - None
     - IBM Quantum backend name (e.g., 'ibm_brisbane')
   * - ``use_simulator``
     - False
     - Use local Aer simulator
   * - ``resilience_level``
     - 1
     - Error mitigation level (0-2)
   * - ``dynamical_decoupling``
     - False
     - Enable dynamical decoupling
   * - ``shots``
     - 1000
     - Default shots per circuit
