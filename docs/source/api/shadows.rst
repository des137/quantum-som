Classical Shadows
=================

.. automodule:: qsom.shadows
   :members:
   :undoc-members:
   :show-inheritance:

Overview
--------

Classical shadows provide an efficient representation of quantum states
using randomized measurements. The :class:`ClassicalShadow` class implements
the protocol from Huang et al. (2020).

Mathematical Background
-----------------------

For an n-qubit state :math:`\rho`, a classical shadow is constructed by:

1. Applying a random unitary :math:`U` from a distribution (e.g., random Paulis)
2. Measuring in the computational basis to get outcome :math:`|b\rangle`
3. Storing the "snapshot" :math:`U^\dagger |b\rangle\langle b| U`

The inverse channel reconstructs the state:

.. math::

   \mathcal{M}^{-1}(|b\rangle\langle b|) = 3|b\rangle\langle b| - I

for single-qubit Pauli measurements.

Basic Usage
-----------

.. code-block:: python

   from qsom import ClassicalShadow, QuantumBackend
   from qiskit import QuantumCircuit

   # Initialize
   backend = QuantumBackend(use_simulator=True)
   shadow = ClassicalShadow(
       n_qubits=3,
       backend=backend,
       shadow_size=100
   )

   # Generate shadow for a circuit
   qc = QuantumCircuit(3)
   qc.h(0)
   qc.cx(0, 1)
   qc.cx(1, 2)

   samples = shadow.generate(qc)
   features = shadow.shadow_to_feature_vector(samples)

Batch Processing
----------------

For multiple circuits, use batch processing for efficiency:

.. code-block:: python

   circuits = [create_circuit(i) for i in range(100)]
   features = shadow.generate_features_batch(circuits)

Clifford Shadows
----------------

For certain applications, Clifford-based shadows can be more efficient:

.. code-block:: python

   from qsom import CliffordShadow

   cliff_shadow = CliffordShadow(
       n_qubits=3,
       shadow_size=100,
       use_global_clifford=True
   )

See :mod:`qsom.clifford_shadows` for details.
