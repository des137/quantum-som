"""
Visualization Module for QSOM.

This module provides advanced visualization capabilities including:
- Interactive Plotly visualizations
- Training animations
- 3D visualizations
"""

from typing import Optional, List, Tuple, Dict, Any
import numpy as np

# Check for optional visualization libraries
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    from matplotlib import animation
    from matplotlib.colors import Normalize
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


def create_interactive_umatrix(
    som: 'QuantumSOM',
    data: Optional[np.ndarray] = None,
    labels: Optional[np.ndarray] = None,
    title: str = "Interactive U-Matrix"
) -> 'go.Figure':
    """
    Create an interactive U-matrix visualization using Plotly.

    Args:
        som: Trained QuantumSOM instance.
        data: Optional data to overlay on the map.
        labels: Optional labels for data points.
        title: Figure title.

    Returns:
        Plotly Figure object.
    """
    if not PLOTLY_AVAILABLE:
        raise ImportError("Plotly is required for interactive visualizations. "
                         "Install with: pip install plotly")

    umatrix = som.get_umatrix()

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('U-Matrix (Distance Map)', 'Data Projection'),
        horizontal_spacing=0.1
    )

    # U-matrix heatmap
    fig.add_trace(
        go.Heatmap(
            z=umatrix,
            colorscale='Viridis',
            colorbar=dict(title='Avg Distance', x=0.45),
            hovertemplate='X: %{x}<br>Y: %{y}<br>Distance: %{z:.4f}<extra></extra>'
        ),
        row=1, col=1
    )

    # Data projection
    if data is not None:
        predictions = som.predict_batch(data)
        hit_map = som.get_hit_map(data)

        # Background heatmap
        fig.add_trace(
            go.Heatmap(
                z=hit_map,
                colorscale='Hot',
                colorbar=dict(title='Hit Count', x=1.0),
                opacity=0.7,
                hovertemplate='X: %{x}<br>Y: %{y}<br>Hits: %{z}<extra></extra>'
            ),
            row=1, col=2
        )

        # Scatter points
        if labels is not None:
            unique_labels = np.unique(labels)
            colors = px.colors.qualitative.Set1[:len(unique_labels)]

            for label, color in zip(unique_labels, colors):
                mask = labels == label
                proj_label = predictions[mask]
                jitter = np.random.uniform(-0.3, 0.3, proj_label.shape)

                fig.add_trace(
                    go.Scatter(
                        x=proj_label[:, 1] + jitter[:, 1],
                        y=proj_label[:, 0] + jitter[:, 0],
                        mode='markers',
                        marker=dict(
                            size=10,
                            color=color,
                            line=dict(width=1, color='white')
                        ),
                        name=f'Class {label}',
                        hovertemplate=f'Class {label}<br>X: %{{x:.1f}}<br>Y: %{{y:.1f}}<extra></extra>'
                    ),
                    row=1, col=2
                )
        else:
            jitter = np.random.uniform(-0.2, 0.2, predictions.shape)
            fig.add_trace(
                go.Scatter(
                    x=predictions[:, 1] + jitter[:, 1],
                    y=predictions[:, 0] + jitter[:, 0],
                    mode='markers',
                    marker=dict(size=8, color='blue', opacity=0.6),
                    name='Data',
                    hovertemplate='X: %{x:.1f}<br>Y: %{y:.1f}<extra></extra>'
                ),
                row=1, col=2
            )

    fig.update_layout(
        title=dict(text=title, x=0.5, font=dict(size=16)),
        showlegend=True,
        legend=dict(x=1.05, y=0.5),
        height=500,
        width=1000
    )

    fig.update_xaxes(title_text='Grid X', row=1, col=1)
    fig.update_yaxes(title_text='Grid Y', row=1, col=1)
    fig.update_xaxes(title_text='Grid X', row=1, col=2)
    fig.update_yaxes(title_text='Grid Y', row=1, col=2)

    return fig


def create_training_animation(
    som: 'QuantumSOM',
    data: np.ndarray,
    n_iterations: int = 500,
    frame_interval: int = 50,
    figsize: Tuple[int, int] = (10, 8)
) -> 'animation.FuncAnimation':
    """
    Create an animation of the SOM training process.

    Args:
        som: QuantumSOM instance (will be modified during training).
        data: Training data.
        n_iterations: Number of training iterations.
        frame_interval: Capture frame every N iterations.
        figsize: Figure size.

    Returns:
        Matplotlib animation object.
    """
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError("Matplotlib is required for animations.")

    # Store weight snapshots
    snapshots = []
    umatrix_snapshots = []

    # Initialize weights if needed
    if som.weights is None:
        som.initialize_weights(data)

    # Capture initial state
    snapshots.append(som.weights.copy())
    umatrix_snapshots.append(som.get_umatrix())

    n_samples = len(data)

    for t in range(n_iterations):
        current_lr = som._adaptive_learning_rate(t, n_iterations)
        current_sigma = som._adaptive_sigma(t, n_iterations)

        idx = np.random.randint(0, n_samples)
        x = data[idx]
        bmu = som._find_bmu(x)
        som._update_weights(x, bmu, current_lr, current_sigma)

        if (t + 1) % frame_interval == 0:
            snapshots.append(som.weights.copy())
            umatrix_snapshots.append(som.get_umatrix())

    # Create animation
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    def animate(frame):
        axes[0].clear()
        axes[1].clear()

        umat = umatrix_snapshots[frame]
        weights = snapshots[frame]

        # U-matrix
        im1 = axes[0].imshow(umat, cmap='viridis', origin='lower')
        axes[0].set_title(f'U-Matrix (Frame {frame})')
        axes[0].set_xlabel('Grid X')
        axes[0].set_ylabel('Grid Y')

        # Component plane
        component = weights[:, :, 0]
        im2 = axes[1].imshow(component, cmap='coolwarm', origin='lower')
        axes[1].set_title(f'Component Plane (Dim 0)')
        axes[1].set_xlabel('Grid X')
        axes[1].set_ylabel('Grid Y')

        plt.tight_layout()
        return [im1, im2]

    anim = animation.FuncAnimation(
        fig, animate, frames=len(snapshots),
        interval=200, blit=False
    )

    return anim


def create_3d_umatrix(
    som: 'QuantumSOM',
    title: str = "3D U-Matrix"
) -> 'go.Figure':
    """
    Create a 3D surface plot of the U-matrix.

    Args:
        som: Trained QuantumSOM instance.
        title: Figure title.

    Returns:
        Plotly Figure object.
    """
    if not PLOTLY_AVAILABLE:
        raise ImportError("Plotly is required for 3D visualizations.")

    umatrix = som.get_umatrix()

    x = np.arange(umatrix.shape[1])
    y = np.arange(umatrix.shape[0])
    X, Y = np.meshgrid(x, y)

    fig = go.Figure(data=[
        go.Surface(
            x=X, y=Y, z=umatrix,
            colorscale='Viridis',
            colorbar=dict(title='Distance'),
            hovertemplate='X: %{x}<br>Y: %{y}<br>Distance: %{z:.4f}<extra></extra>'
        )
    ])

    fig.update_layout(
        title=dict(text=title, x=0.5),
        scene=dict(
            xaxis_title='Grid X',
            yaxis_title='Grid Y',
            zaxis_title='Avg Distance',
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.2))
        ),
        height=600,
        width=800
    )

    return fig


def create_component_planes(
    som: 'QuantumSOM',
    n_components: int = 4,
    title: str = "Component Planes"
) -> 'go.Figure':
    """
    Create interactive component plane visualizations.

    Args:
        som: Trained QuantumSOM instance.
        n_components: Number of components to display.
        title: Figure title.

    Returns:
        Plotly Figure object.
    """
    if not PLOTLY_AVAILABLE:
        raise ImportError("Plotly is required for interactive visualizations.")

    components = som.get_component_planes()
    n_show = min(n_components, components.shape[0])

    # Create subplot grid
    cols = 2
    rows = (n_show + 1) // 2

    fig = make_subplots(
        rows=rows, cols=cols,
        subplot_titles=[f'Component {i}' for i in range(n_show)]
    )

    for i in range(n_show):
        row = i // cols + 1
        col = i % cols + 1

        fig.add_trace(
            go.Heatmap(
                z=components[i],
                colorscale='RdBu',
                showscale=(i == 0),
                hovertemplate=f'Component {i}<br>X: %{{x}}<br>Y: %{{y}}<br>Value: %{{z:.4f}}<extra></extra>'
            ),
            row=row, col=col
        )

    fig.update_layout(
        title=dict(text=title, x=0.5),
        height=300 * rows,
        width=800
    )

    return fig


def create_hit_histogram(
    som: 'QuantumSOM',
    data: np.ndarray,
    labels: Optional[np.ndarray] = None,
    title: str = "Hit Histogram"
) -> 'go.Figure':
    """
    Create an interactive hit histogram showing BMU frequency.

    Args:
        som: Trained QuantumSOM instance.
        data: Data to project onto the SOM.
        labels: Optional labels for coloring.
        title: Figure title.

    Returns:
        Plotly Figure object.
    """
    if not PLOTLY_AVAILABLE:
        raise ImportError("Plotly is required for interactive visualizations.")

    hit_map = som.get_hit_map(data)

    fig = go.Figure(data=[
        go.Heatmap(
            z=hit_map,
            colorscale='YlOrRd',
            colorbar=dict(title='Hit Count'),
            hovertemplate='X: %{x}<br>Y: %{y}<br>Hits: %{z}<extra></extra>'
        )
    ])

    fig.update_layout(
        title=dict(text=title, x=0.5),
        xaxis_title='Grid X',
        yaxis_title='Grid Y',
        height=500,
        width=600
    )

    return fig


def create_training_progress(
    som: 'QuantumSOM',
    title: str = "Training Progress"
) -> 'go.Figure':
    """
    Create an interactive training progress visualization.

    Args:
        som: Trained QuantumSOM instance with training history.
        title: Figure title.

    Returns:
        Plotly Figure object.
    """
    if not PLOTLY_AVAILABLE:
        raise ImportError("Plotly is required for interactive visualizations.")

    if not som.training_history:
        raise ValueError("No training history available. Train the SOM first.")

    iterations = [h['iteration'] for h in som.training_history]
    qe = [h['quantization_error'] for h in som.training_history]
    lr = [h['learning_rate'] for h in som.training_history]
    sigma = [h['sigma'] for h in som.training_history]

    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=('Quantization Error', 'Learning Rate', 'Neighborhood Radius')
    )

    fig.add_trace(
        go.Scatter(
            x=iterations, y=qe,
            mode='lines+markers',
            name='QE',
            line=dict(color='blue', width=2),
            hovertemplate='Iteration: %{x}<br>QE: %{y:.4f}<extra></extra>'
        ),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=iterations, y=lr,
            mode='lines+markers',
            name='LR',
            line=dict(color='green', width=2),
            hovertemplate='Iteration: %{x}<br>LR: %{y:.4f}<extra></extra>'
        ),
        row=1, col=2
    )

    fig.add_trace(
        go.Scatter(
            x=iterations, y=sigma,
            mode='lines+markers',
            name='Sigma',
            line=dict(color='red', width=2),
            hovertemplate='Iteration: %{x}<br>Sigma: %{y:.4f}<extra></extra>'
        ),
        row=1, col=3
    )

    fig.update_layout(
        title=dict(text=title, x=0.5),
        showlegend=False,
        height=400,
        width=1200
    )

    fig.update_xaxes(title_text='Iteration', row=1, col=1)
    fig.update_xaxes(title_text='Iteration', row=1, col=2)
    fig.update_xaxes(title_text='Iteration', row=1, col=3)
    fig.update_yaxes(title_text='Error', row=1, col=1)
    fig.update_yaxes(title_text='Rate', row=1, col=2)
    fig.update_yaxes(title_text='Radius', row=1, col=3)

    return fig
