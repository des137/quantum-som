IBM Quantum Jobs
================

.. automodule:: qsom.ibm_jobs
   :members:
   :undoc-members:
   :show-inheritance:

Overview
--------

The IBM jobs module provides utilities for managing quantum jobs on IBM
Quantum hardware, including job submission, monitoring, and recovery.

Setup
-----

First, save your IBM Quantum credentials:

.. code-block:: python

   from qiskit_ibm_runtime import QiskitRuntimeService

   # Save credentials (one time)
   QiskitRuntimeService.save_account(
       channel="ibm_quantum",
       token="YOUR_API_TOKEN"
   )

Basic Usage
-----------

.. code-block:: python

   from qsom import IBMJobManager

   # Initialize job manager
   manager = IBMJobManager()

   # List available backends
   backends = manager.list_backends()
   print(backends)

   # Submit a circuit
   job_info = manager.submit_circuit(
       circuit=qc,
       shots=1000,
       backend_name='ibm_brisbane'
   )
   print(f"Job ID: {job_info.job_id}")

   # Wait for result
   result = manager.wait_for_job(job_info.job_id)

Batch Job Submission
--------------------

Submit multiple circuits efficiently:

.. code-block:: python

   circuits = [create_circuit(i) for i in range(100)]

   job_infos = manager.submit_batch(
       circuits,
       shots=1000,
       batch_name="shadow_measurements"
   )

Session Management
------------------

Use sessions for multiple related jobs:

.. code-block:: python

   # Start session
   with manager.start_session(backend_name='ibm_brisbane') as session:
       for batch in circuit_batches:
           job_info = manager.submit_in_session(batch)
           result = manager.wait_for_job(job_info.job_id)
           process_result(result)

Job Recovery
------------

Recover jobs after script interruption:

.. code-block:: python

   # Jobs are automatically persisted
   manager = IBMJobManager(jobs_dir='~/.qsom_jobs')

   # Recover all saved jobs
   recovered = manager.recover_jobs()

   for job_id, info in recovered.items():
       print(f"{job_id}: {info.status}")

Shadow Job Runner
-----------------

Specialized runner for classical shadow measurements:

.. code-block:: python

   from qsom import ShadowJobRunner

   runner = ShadowJobRunner(
       job_manager=manager,
       n_qubits=3,
       shadow_size=100
   )

   # Generate random Pauli bases
   bases = [''.join(np.random.choice(['X','Y','Z'], 3))
            for _ in range(100)]

   # Submit shadow circuits
   job_info = runner.submit_shadow_circuits(
       state_circuit=qc,
       pauli_bases=bases
   )

   # Collect results
   shadow_samples = runner.collect_shadow_results(
       job_info.job_id,
       pauli_bases=bases
   )

Time Estimation
---------------

Estimate job execution time:

.. code-block:: python

   from qsom import estimate_job_time

   estimate = estimate_job_time(
       n_circuits=100,
       n_qubits=5,
       shots=1000
   )
   print(f"Estimated time: {estimate['estimated_total_minutes']:.1f} minutes")
