"""
Optimized neural network inference and PCA reconstruction using cached data.
"""
import jax
import jax.numpy as jnp
from .data_cache import (
    parameters_NN, mean, scale, 
    pca_data_exponent, pca_data_scaling, 
    pca_data_eigenvectors, pca_data_eigenvalues, pca_data_mean,
    amp_points, phase_points, pc_exponent
)

# Pre-unpack neural network parameters for maximum efficiency
W0, W1, W2 = parameters_NN[0]
b0, b1, b2 = parameters_NN[1]

@jax.jit
def mlp_forward_optimized(x):
    """Optimized MLP forward pass with pre-unpacked weights."""
    # Scale input
    x_scaled = (x - mean) / scale

    # Forward pass with pre-unpacked parameters

    # Use more efficient activation patterns
    z1 = jnp.dot(x_scaled, W0) + b0
    a1 = jnp.tanh(z1)

    z2 = jnp.dot(a1, W1) + b1
    a2 = jnp.tanh(z2)

    z3 = jnp.dot(a2, W2) + b2
    return z3

@jax.jit
def pca_reconstruct_optimized(pca_reduced_data):
    """Optimized PCA reconstruction with pre-loaded parameters."""
    # Apply eigenvalue scaling
    reduced_data = pca_reduced_data / (pca_data_eigenvalues ** pc_exponent)
    
    # Apply scaling
    scaled_data = reduced_data * pca_data_scaling
    
    # Matrix multiplication for reconstruction
    zero_mean_data = jnp.dot(scaled_data, pca_data_eigenvectors.T)
    
    return zero_mean_data + pca_data_mean

@jax.jit
def full_reconstruct_data_NN_optimized(x):
    """Optimized full reconstruction pipeline."""
    pca_reduced_data = mlp_forward_optimized(x)
    return pca_reconstruct_optimized(pca_reduced_data)

@jax.jit
def from_combined_residuals_optimized(combined_residuals):
    """Optimized residual splitting with pre-computed split points."""
    return combined_residuals[:, :amp_points], combined_residuals[:, -phase_points:]
