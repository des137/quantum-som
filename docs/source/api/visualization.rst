Visualization
=============

.. automodule:: qsom.visualization
   :members:
   :undoc-members:
   :show-inheritance:

Overview
--------

QSOM provides both static (Matplotlib) and interactive (Plotly) visualizations
for exploring trained SOMs.

U-Matrix Visualization
----------------------

The U-matrix shows distances between neighboring neurons, revealing cluster
boundaries.

Static U-Matrix
~~~~~~~~~~~~~~~

.. code-block:: python

   from qsom import QuantumSOM

   som = QuantumSOM(grid_size=(10, 10), input_dim=100)
   som.train(features)

   # Built-in visualization
   fig = som.visualize(data=features, labels=labels)

Interactive U-Matrix
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from qsom import create_interactive_umatrix

   fig = create_interactive_umatrix(
       som,
       data=features,
       labels=labels,
       title="Quantum State Space"
   )
   fig.show()

3D Surface Plot
---------------

.. code-block:: python

   from qsom import create_3d_umatrix

   fig = create_3d_umatrix(som, colorscale='Viridis')
   fig.show()

Component Planes
----------------

Visualize how individual feature dimensions are distributed across the map:

.. code-block:: python

   from qsom import create_component_planes

   fig = create_component_planes(
       som,
       feature_names=['X0', 'Y0', 'Z0', 'X1', 'Y1', 'Z1'],
       n_cols=3
   )
   fig.show()

Training Animation
------------------

Create an animation of the training process:

.. code-block:: python

   from qsom import create_training_animation

   # Train with history tracking
   som.train(features, track_errors=True)

   # Create animation (requires training history)
   ani = create_training_animation(
       som,
       data=features,
       interval=100
   )
   ani.save('training.gif', writer='pillow')

Hit Histogram
-------------

Show how data points are distributed across neurons:

.. code-block:: python

   from qsom import create_hit_histogram

   fig = create_hit_histogram(som, data=features)
   fig.show()
