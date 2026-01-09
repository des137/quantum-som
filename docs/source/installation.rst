Installation
============

Requirements
------------

* Python 3.9 or higher
* NumPy
* Qiskit 2.0+

Basic Installation
------------------

Install from source:

.. code-block:: bash

   git clone https://github.com/amoldeshmukh/quantum-som.git
   cd quantum-som
   pip install -e .

Optional Dependencies
---------------------

QSOM has several optional dependencies for extended functionality:

IBM Quantum Hardware
~~~~~~~~~~~~~~~~~~~~

For running on real quantum computers:

.. code-block:: bash

   pip install qsom[ibm]
   # or
   pip install qiskit-ibm-runtime qiskit-aer

Interactive Visualization
~~~~~~~~~~~~~~~~~~~~~~~~~

For Plotly-based visualizations:

.. code-block:: bash

   pip install qsom[viz]
   # or
   pip install plotly kaleido

GPU Acceleration
~~~~~~~~~~~~~~~~

For JAX-based acceleration:

.. code-block:: bash

   pip install jax jaxlib

For CuPy (NVIDIA GPUs):

.. code-block:: bash

   pip install cupy-cuda12x  # Adjust for your CUDA version

Development Installation
~~~~~~~~~~~~~~~~~~~~~~~~

For development with testing tools:

.. code-block:: bash

   pip install qsom[dev]
   # or
   pip install pytest pytest-cov mypy ruff

All Dependencies
~~~~~~~~~~~~~~~~

Install everything:

.. code-block:: bash

   pip install qsom[all]

Verifying Installation
----------------------

.. code-block:: python

   import qsom

   print(f"QSOM version: {qsom.__version__}")

   # Check available features
   from qsom import QuantumBackend, ClassicalShadow, QuantumSOM

   # Test basic functionality
   backend = QuantumBackend(use_simulator=True)
   print("Installation successful!")

Troubleshooting
---------------

Qiskit Import Errors
~~~~~~~~~~~~~~~~~~~~

If you see Qiskit import errors, ensure you have Qiskit 2.0+:

.. code-block:: bash

   pip install --upgrade qiskit qiskit-aer

JAX Installation Issues
~~~~~~~~~~~~~~~~~~~~~~~

For M1/M2 Macs:

.. code-block:: bash

   pip install jax-metal
   pip install jax jaxlib

For Linux with CUDA:

.. code-block:: bash

   pip install jax[cuda12_pip] -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
