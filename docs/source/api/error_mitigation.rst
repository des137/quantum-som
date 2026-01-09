Error Mitigation
================

.. automodule:: qsom.error_mitigation
   :members:
   :undoc-members:
   :show-inheritance:

Overview
--------

The error mitigation module provides techniques for improving the quality
of quantum measurements on noisy hardware.

Measurement Error Mitigation
----------------------------

Corrects readout errors using a calibration matrix:

.. code-block:: python

   from qsom import MeasurementErrorMitigator

   mitigator = MeasurementErrorMitigator(n_qubits=3)

   # Calibrate on the backend
   calibration_matrix = mitigator.calibrate(backend, shots=8192)

   # Apply mitigation to measurement results
   mitigated_probs = mitigator.mitigate(raw_counts)

Zero Noise Extrapolation (ZNE)
------------------------------

Runs circuits at multiple noise levels and extrapolates to zero noise:

.. code-block:: python

   from qsom import ZeroNoiseExtrapolator

   zne = ZeroNoiseExtrapolator(
       scale_factors=[1.0, 2.0, 3.0],
       extrapolation='linear'
   )

   # Run ZNE workflow
   mitigated_value, details = zne.run_zne(
       circuit=qc,
       backend=backend,
       observable_fn=compute_expectation,
       shots=4096
   )

Dynamical Decoupling
--------------------

Apply dynamical decoupling sequences for coherence protection:

.. code-block:: python

   from qsom import DynamicalDecoupling

   # Available sequences: 'xy4', 'cpmg', 'xy8'
   dd = DynamicalDecoupling(sequence='xy4')

   # Insert DD into circuit
   protected_circuit = dd.insert_dd(circuit, idle_qubits=[0, 1])

TREX (Twirled Readout Error Extinction)
---------------------------------------

Randomize readout errors via Pauli twirling:

.. code-block:: python

   from qsom import TwirledReadoutMitigator

   trex = TwirledReadoutMitigator(n_qubits=3, n_twirls=10)

   # Run with TREX mitigation
   mitigated_probs = trex.mitigate(
       circuit=qc,
       backend=backend,
       shots_per_twirl=1000
   )
