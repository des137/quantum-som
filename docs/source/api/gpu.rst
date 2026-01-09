GPU Acceleration
================

.. automodule:: qsom.gpu
   :members:
   :undoc-members:
   :show-inheritance:

Overview
--------

The GPU module provides optional acceleration using JAX or CuPy backends.
This can provide significant speedups for large-scale SOM training.

Requirements
------------

Install one of the GPU backends:

.. code-block:: bash

   # JAX (recommended)
   pip install jax jaxlib

   # CuPy (NVIDIA GPUs)
   pip install cupy-cuda12x  # Adjust for your CUDA version

Checking GPU Availability
-------------------------

.. code-block:: python

   from qsom import gpu_available, get_gpu_backend

   if gpu_available():
       print("GPU acceleration available!")
       xp = get_gpu_backend()  # Returns jax.numpy, cupy, or numpy
   else:
       print("Using NumPy backend")

GPU-Accelerated SOM
-------------------

.. code-block:: python

   from qsom import GPUAcceleratedSOM, create_optimized_som

   # Manually specify backend
   gpu_som = GPUAcceleratedSOM(
       grid_size=(20, 20),
       input_dim=100,
       backend='jax'  # or 'cupy', 'numpy'
   )

   # Or automatically select best available
   gpu_som = create_optimized_som(
       grid_size=(20, 20),
       input_dim=100
   )

   # Train
   gpu_som.train(features, n_iterations=5000, verbose=True)

   # Get weights as NumPy array
   weights = gpu_som.get_weights_numpy()

Performance Benchmarking
------------------------

Compare backends on your hardware:

.. code-block:: python

   from qsom.gpu import benchmark_backends

   results = benchmark_backends(
       grid_size=(30, 30),
       input_dim=200,
       n_samples=2000,
       n_iterations=1000
   )

   for backend, stats in results.items():
       print(f"{backend}: {stats['time']:.2f}s "
             f"({stats['iterations_per_second']:.0f} iter/s)")

Typical Speedups
----------------

Expected performance improvements with JAX on GPU:

.. list-table::
   :header-rows: 1
   :widths: 30 30 40

   * - Grid Size
     - Input Dim
     - JAX vs NumPy
   * - 10x10
     - 50
     - ~2x
   * - 20x20
     - 100
     - ~5-10x
   * - 50x50
     - 500
     - ~20-50x
