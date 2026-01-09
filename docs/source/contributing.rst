Contributing
============

We welcome contributions to QSOM! This guide explains how to contribute.

Getting Started
---------------

1. Fork the repository on GitHub
2. Clone your fork:

   .. code-block:: bash

      git clone https://github.com/YOUR_USERNAME/quantum-som.git
      cd quantum-som

3. Install development dependencies:

   .. code-block:: bash

      pip install -e ".[dev]"

4. Create a branch for your changes:

   .. code-block:: bash

      git checkout -b feature/your-feature-name

Development Workflow
--------------------

Running Tests
~~~~~~~~~~~~~

.. code-block:: bash

   # Run all tests
   pytest tests/ -v

   # Run with coverage
   pytest tests/ --cov=src/qsom --cov-report=html

   # Run specific test file
   pytest tests/test_som.py -v

Code Quality
~~~~~~~~~~~~

.. code-block:: bash

   # Format code
   black src/ tests/

   # Lint
   ruff check src/ tests/

   # Type checking
   mypy src/qsom

Building Documentation
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   cd docs
   make html
   # Open docs/build/html/index.html

Pull Request Guidelines
-----------------------

1. **Write tests** for new functionality
2. **Update documentation** for API changes
3. **Follow code style** (run black and ruff)
4. **Add type hints** to new code
5. **Write clear commit messages**
6. **Keep PRs focused** - one feature/fix per PR

Code Style
----------

* Follow PEP 8
* Use type hints
* Write docstrings (Google style)
* Maximum line length: 100 characters

Example docstring:

.. code-block:: python

   def estimate_observable(
       self,
       shadow_samples: List[Tuple[np.ndarray, np.ndarray]],
       observable: np.ndarray
   ) -> float:
       """
       Estimate expectation value from shadow samples.

       Args:
           shadow_samples: List of (pauli_indices, outcomes) tuples.
           observable: Observable matrix to estimate.

       Returns:
           Estimated expectation value.

       Raises:
           ValueError: If observable dimensions don't match.
       """

Testing Guidelines
------------------

* Write unit tests for all public functions
* Use pytest fixtures for common setup
* Mock external dependencies (Qiskit backends)
* Test edge cases and error conditions

Example test:

.. code-block:: python

   def test_som_finds_bmu():
       som = QuantumSOM(grid_size=(5, 5), input_dim=10)
       x = np.random.randn(10)
       bmu = som.find_bmu(x)

       assert 0 <= bmu[0] < 5
       assert 0 <= bmu[1] < 5

Reporting Issues
----------------

When reporting bugs, please include:

* Python version
* QSOM version
* Qiskit version (if applicable)
* Minimal code to reproduce
* Full error traceback

Feature Requests
----------------

For feature requests, please describe:

* The use case
* Expected behavior
* Any relevant references (papers, etc.)

Questions
---------

For questions about using QSOM, please open a GitHub Discussion rather
than an Issue.
