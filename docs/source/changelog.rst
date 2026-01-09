Changelog
=========

Version 0.3.0
-------------

**Features**

* Added Clifford-based classical shadows (``CliffordShadow``, ``LocalCliffordShadow``)
* Added GPU acceleration with JAX/CuPy backends (``GPUAcceleratedSOM``)
* Added IBM Quantum job management (``IBMJobManager``, ``ShadowJobRunner``)
* Added sparse matrix support for large qubit systems (``SparseSOM``)
* Added comprehensive error mitigation strategies:
  - Zero Noise Extrapolation (ZNE)
  - Measurement Error Mitigation
  - Dynamical Decoupling
  - TREX (Twirled Readout Error Extinction)
* Added benchmarking utilities (``QSOMBenchmark``)

**Improvements**

* New distance metrics: Hilbert-Schmidt, cosine, Manhattan
* Mini-batch training for large datasets
* Model checkpointing and resumption
* Interactive Plotly visualizations
* 3D surface plots
* Training animations
* Batch processing for shadow generation

**Documentation**

* Added Sphinx documentation
* Added mathematical background section
* Created tutorial Jupyter notebook
* Updated README with examples

Version 0.2.0
-------------

**Features**

* Multiple distance metrics (euclidean, fidelity, bures, trace, quantum)
* Adaptive learning rate scheduling
* Proper inverse channel reconstruction for Pauli shadows
* Qiskit 2.0+ compatibility

**Improvements**

* Better error handling
* Improved visualization

Version 0.1.0
-------------

Initial release with:

* Basic classical shadow generation
* Self-organizing map implementation
* Qiskit backend integration
* Basic visualization
