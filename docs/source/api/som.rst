Quantum SOM
===========

.. automodule:: qsom.som
   :members:
   :undoc-members:
   :show-inheritance:

Overview
--------

The :class:`QuantumSOM` class implements a Self-Organizing Map (Kohonen, 1982)
optimized for quantum state data with quantum-specific distance metrics.

Basic Usage
-----------

.. code-block:: python

   from qsom import QuantumSOM

   # Create and train SOM
   som = QuantumSOM(
       grid_size=(10, 10),
       input_dim=features.shape[1],
       learning_rate=0.5,
       sigma=1.0,
       n_iterations=1000,
       distance_metric='quantum'
   )

   som.train(features, verbose=True)

   # Visualize
   som.visualize()

Distance Metrics
----------------

Available distance metrics:

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Metric
     - Description
   * - ``euclidean``
     - Standard Euclidean distance (fastest)
   * - ``fidelity``
     - Quantum fidelity-based distance
   * - ``bures``
     - Bures distance (metric on density matrices)
   * - ``trace``
     - Trace distance
   * - ``quantum``
     - Weighted combination of fidelity and trace
   * - ``hilbert_schmidt``
     - Hilbert-Schmidt inner product based
   * - ``cosine``
     - Cosine similarity based
   * - ``manhattan``
     - L1 (Manhattan) distance

Mini-Batch Training
-------------------

For large datasets, use mini-batch training:

.. code-block:: python

   som.train_minibatch(
       features,
       n_epochs=20,
       batch_size=32,
       checkpoint_dir='checkpoints',
       checkpoint_freq=5
   )

Checkpointing
-------------

Save and load trained models:

.. code-block:: python

   # Save
   som.save_checkpoint('models/my_som')

   # Load
   loaded_som = QuantumSOM.load_checkpoint('models/my_som')

GPU Acceleration
----------------

For large-scale training, use GPU acceleration:

.. code-block:: python

   from qsom import GPUAcceleratedSOM, create_optimized_som

   # Automatically select best backend (JAX > CuPy > NumPy)
   gpu_som = create_optimized_som(
       grid_size=(20, 20),
       input_dim=features.shape[1]
   )

   gpu_som.train(features, n_iterations=5000)
