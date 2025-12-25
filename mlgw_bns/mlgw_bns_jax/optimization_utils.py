"""
Additional optimization utilities for memory layout and computational efficiency.
"""
import jax
import jax.numpy as jnp

@jax.jit
def optimize_memory_layout(arrays):
    """Optimize memory layout for better cache performance."""
    return [jnp.asarray(arr, order='C') for arr in arrays]

@jax.jit  
def batch_interpolation_optimized(x_points, x_new, y_batch):
    """
    Optimized batch interpolation that processes multiple signals efficiently.
    Uses vectorized operations instead of loops.
    """
    # Use JAX's native vectorized interpolation
    return jax.vmap(
        lambda y: jnp.interp(x_new, x_points, y),
        in_axes=0, out_axes=0
    )(y_batch)

# Consider using lower precision if accuracy allows
@jax.jit
def convert_to_float32_if_appropriate(x):
    """Convert to float32 if the precision loss is acceptable."""
    return x.astype(jnp.float32)

# Pre-compute frequently used constants
PI_2 = 2 * jnp.pi

@jax.jit
def fast_phase_computation(phase_base, reference_phase, time_shift, frequency):
    """Optimized phase computation with pre-computed constants."""
    return phase_base + reference_phase + PI_2 * time_shift * frequency
