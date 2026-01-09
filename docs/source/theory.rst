Mathematical Background
=======================

This section provides the mathematical foundations underlying QSOM, including
classical shadows, self-organizing maps, and quantum distance metrics.

Classical Shadows
-----------------

Classical shadows provide an efficient classical representation of quantum
states that enables prediction of many properties from few measurements.

Quantum State Representation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

An n-qubit quantum state is described by a density matrix
:math:`\rho \in \mathbb{C}^{2^n \times 2^n}` satisfying:

* :math:`\rho = \rho^\dagger` (Hermitian)
* :math:`\rho \geq 0` (positive semi-definite)
* :math:`\mathrm{Tr}(\rho) = 1` (unit trace)

For pure states: :math:`\rho = |\psi\rangle\langle\psi|`

The Shadow Protocol
~~~~~~~~~~~~~~~~~~~

The classical shadow protocol (Huang et al., 2020) works as follows:

1. **Random Unitary**: Apply a random unitary :math:`U` from an ensemble
   (e.g., random Paulis or Cliffords)

2. **Measurement**: Measure in the computational basis to obtain outcome
   :math:`|b\rangle`

3. **Classical Storage**: Store the classical description :math:`(U, b)`

4. **Inverse Channel**: Reconstruct via :math:`\hat{\rho} = \mathcal{M}^{-1}(U^\dagger|b\rangle\langle b|U)`

Inverse Channel for Pauli Measurements
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For random Pauli measurements, the inverse channel is:

.. math::

   \mathcal{M}^{-1}(|b\rangle\langle b|) = 3|b\rangle\langle b| - I

For n qubits with independent Pauli measurements on each qubit:

.. math::

   \mathcal{M}^{-1} = \bigotimes_{i=1}^{n} (3|\hat{b}_i\rangle\langle\hat{b}_i| - I)

where :math:`|\hat{b}_i\rangle` is the post-measurement state of qubit :math:`i`.

Inverse Channel for Clifford Measurements
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For global random Clifford measurements:

.. math::

   \mathcal{M}^{-1}(|b\rangle\langle b|) = (2^n + 1)|b\rangle\langle b| - I

Sample Complexity
~~~~~~~~~~~~~~~~~

To estimate an observable :math:`O` with :math:`\|O\| \leq 1` to error
:math:`\epsilon`:

* **Pauli shadows**: :math:`\mathcal{O}(4^k \log(M) / \epsilon^2)` samples
  for k-local observables
* **Clifford shadows**: :math:`\mathcal{O}(\|O\|_{\mathrm{shadow}}^2 / \epsilon^2)` samples

Self-Organizing Maps
--------------------

Self-Organizing Maps (Kohonen, 1982) are unsupervised neural networks for
dimensionality reduction that preserve topological structure.

Architecture
~~~~~~~~~~~~

A SOM consists of:

* **Input space**: :math:`\mathbf{x} \in \mathbb{R}^d`
* **Map space**: 2D grid of neurons at positions :math:`(i, j)`
* **Weight vectors**: :math:`\mathbf{w}_{ij} \in \mathbb{R}^d` for each neuron

Training Algorithm
~~~~~~~~~~~~~~~~~~

1. **Best Matching Unit (BMU)**: For input :math:`\mathbf{x}`, find:

   .. math::

      (i^*, j^*) = \arg\min_{i,j} \|\mathbf{x} - \mathbf{w}_{ij}\|

2. **Neighborhood Function**: Gaussian influence:

   .. math::

      h_{ij}(t) = \exp\left(-\frac{(i-i^*)^2 + (j-j^*)^2}{2\sigma(t)^2}\right)

3. **Weight Update**:

   .. math::

      \mathbf{w}_{ij}(t+1) = \mathbf{w}_{ij}(t) + \alpha(t) h_{ij}(t)
      (\mathbf{x} - \mathbf{w}_{ij}(t))

4. **Parameter Decay**:

   .. math::

      \alpha(t) = \alpha_0 \exp(-t/\tau), \quad
      \sigma(t) = \sigma_0 \exp(-t/\tau)

Quantum Distance Metrics
------------------------

QSOM supports several quantum-aware distance metrics.

Euclidean Distance
~~~~~~~~~~~~~~~~~~

Standard :math:`L_2` distance on feature vectors:

.. math::

   d_E(\mathbf{x}, \mathbf{y}) = \|\mathbf{x} - \mathbf{y}\|_2

Fidelity-Based Distance
~~~~~~~~~~~~~~~~~~~~~~~

Derived from quantum fidelity :math:`F(\rho, \sigma)`:

.. math::

   d_F(\rho, \sigma) = 1 - F(\rho, \sigma)

where for pure states:

.. math::

   F(|\psi\rangle, |\phi\rangle) = |\langle\psi|\phi\rangle|^2

Bures Distance
~~~~~~~~~~~~~~

The Bures metric on density matrices:

.. math::

   d_B(\rho, \sigma) = \sqrt{2 - 2\sqrt{F(\rho, \sigma)}}

Trace Distance
~~~~~~~~~~~~~~

The trace distance (total variation distance for quantum states):

.. math::

   d_T(\rho, \sigma) = \frac{1}{2}\|\rho - \sigma\|_1
   = \frac{1}{2}\mathrm{Tr}|\rho - \sigma|

Hilbert-Schmidt Distance
~~~~~~~~~~~~~~~~~~~~~~~~

Based on the Hilbert-Schmidt inner product:

.. math::

   d_{HS}(\rho, \sigma) = \sqrt{\mathrm{Tr}[(\rho - \sigma)^2]}

Combined Quantum Metric
~~~~~~~~~~~~~~~~~~~~~~~

QSOM's ``quantum`` metric combines multiple measures:

.. math::

   d_Q(\mathbf{x}, \mathbf{y}) = w_F \cdot d_F + w_T \cdot d_T + w_E \cdot d_E

Error Mitigation Theory
-----------------------

Zero Noise Extrapolation
~~~~~~~~~~~~~~~~~~~~~~~~

ZNE amplifies noise and extrapolates to zero:

1. **Noise Scaling**: Scale noise by factor :math:`\lambda` using circuit folding:
   :math:`U \rightarrow U(U^\dagger U)^n`

2. **Extrapolation**: Fit :math:`E(\lambda)` and extrapolate:

   .. math::

      E_{\mathrm{ideal}} = \lim_{\lambda \to 0} E(\lambda)

Measurement Error Mitigation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Characterize confusion matrix :math:`A` where :math:`A_{ij} = P(\text{measure } i | \text{prepared } j)`:

.. math::

   \mathbf{p}_{\mathrm{noisy}} = A \mathbf{p}_{\mathrm{ideal}}

Mitigate by applying pseudo-inverse:

.. math::

   \mathbf{p}_{\mathrm{mitigated}} = A^+ \mathbf{p}_{\mathrm{noisy}}

References
----------

1. Huang, H.-Y., Kueng, R., & Preskill, J. (2020). Predicting many properties
   of a quantum system from very few measurements. *Nature Physics*, 16(10),
   1050-1057.

2. Kohonen, T. (1982). Self-organized formation of topologically correct
   feature maps. *Biological Cybernetics*, 43(1), 59-69.

3. Nielsen, M. A., & Chuang, I. L. (2010). *Quantum Computation and Quantum
   Information*. Cambridge University Press.

4. Temme, K., Bravyi, S., & Gambetta, J. M. (2017). Error mitigation for
   short-depth quantum circuits. *Physical Review Letters*, 119(18), 180509.

5. Elben, A., et al. (2022). The randomized measurement toolbox.
   *Nature Reviews Physics*, 5, 9-24.
