import jax
import jax.numpy as jnp
from jax import jit, vmap


# ---------------------------------------------------------
# 1. Compute natural cubic spline coefficient *matrices*
#    These depend ONLY on x_ds, not on y_ds.
# ---------------------------------------------------------
@jit
def cubic_spline_system(x):
    """
    Build the tridiagonal system matrix components for the cubic spline.
    These depend only on x, not on y.
    """
    n = x.shape[0]
    h = x[1:] - x[:-1]

    # Diagonals of the tridiagonal system
    lower = jnp.zeros(n)        # sub-diagonal (L)
    diag = jnp.zeros(n)         # main diagonal (D)
    upper = jnp.zeros(n)        # super-diagonal (U)

    # Natural boundary conditions
    diag = diag.at[0].set(1.0)
    diag = diag.at[-1].set(1.0)

    # Middle equations
    diag = diag.at[1:-1].set(2 * (x[2:] - x[:-2]))
    upper = upper.at[1:-1].set(h[1:])
    lower = lower.at[1:-1].set(h[:-1])

    return lower, diag, upper, h


# ---------------------------------------------------------
# 2. Compute spline coefficients a,b,c,d for a single y array
# ---------------------------------------------------------
@jit
def spline_coeffs_single(x, y, lower, diag, upper, h):
    n = x.shape[0]

    # α for RHS (size n-2)
    alpha = (3/h[1:] * (y[2:] - y[1:-1])
             - 3/h[:-1] * (y[1:-1] - y[:-2]))

    # Full RHS vector (size n)
    rhs = jnp.zeros_like(y)
    rhs = rhs.at[1:-1].set(alpha)

    # ---- FIX: tridiagonal_solve expects 2-D RHS ----
    rhs2d = rhs[:, None]          # shape (n, 1)

    # Solve
    c2d = jax.lax.linalg.tridiagonal_solve(lower, diag, upper, rhs2d)

    # Back to 1-D
    c = c2d[:, 0]

    # Compute spline coefficients
    a = y[:-1]
    b = (y[1:] - y[:-1]) / h - h * (2*c[:-1] + c[1:]) / 3
    d = (c[1:] - c[:-1]) / (3*h)

    return a, b, c[:-1], d

# ---------------------------------------------------------
# 3. Vectorized version for arbitrary batch size
# ---------------------------------------------------------
@jit
def cubic_spline_coeffs_batched(x, y_batch):
    """
    Compute cubic spline coefficients for batched y_ds:
        y_batch.shape = (batch, N)
    """
    lower, diag, upper, h = cubic_spline_system(x)

    # vmap over the batch dimension
    batched_fn = vmap(
        lambda y: spline_coeffs_single(x, y, lower, diag, upper, h),
        in_axes=0,  # batch in first axis
        out_axes=0  # return batched coefficients
    )

    return batched_fn(y_batch)


# ---------------------------------------------------------
# 4. Spline evaluation (supports batches)
# ---------------------------------------------------------
@jit
def cubic_spline_eval_batched(new_x, x, a, b, c, d):
    """
    Evaluate batched splines with coefficients:
        a,b,c,d all have shape (batch, N-1)
    Result shape: (batch, len(new_x))
    """

    idx = jnp.clip(jnp.searchsorted(x, new_x) - 1,
                   0, x.shape[0] - 2)

    dx = new_x - x[idx]

    # vmap over batch dimension
    def eval_one(a_i, b_i, c_i, d_i):
        return (a_i[idx]
                + b_i[idx] * dx
                + c_i[idx] * dx**2
                + d_i[idx] * dx**3)

    return vmap(eval_one, in_axes=(0,0,0,0))(a, b, c, d)


# ---------------------------------------------------------
# 5. Final resample function (drop-in replacement)
# ---------------------------------------------------------
@jit
def resample(x_ds, new_x, y_ds):
    """
    y_ds can be shape (batch, N) or (N,) (automatically promoted to batch=1)
    """
    if y_ds.ndim == 1:
        y_ds = y_ds[None, :]   # promote to batch=1

    if x_ds.shape[0] != y_ds.shape[1]:
        raise ValueError(
            f"Shape mismatch: x_ds={x_ds.shape}, y_ds={y_ds.shape}"
        )

    # Compute cubic spline coefficients for the batch
    a, b, c, d = cubic_spline_coeffs_batched(x_ds, y_ds)

    # Evaluate spline
    return cubic_spline_eval_batched(new_x, x_ds, a, b, c, d)

@jit
def linear_resample_jax(x_ds, new_x, y_ds):
    """
    Fast linear interpolation using jnp.interp.
    y_ds may be (N,) or (B, N). Returns shape (B, M) or (1, M).
    """
    if y_ds.ndim == 1:
        # single signal -> use jnp.interp directly
        return jnp.interp(new_x, x_ds, y_ds)[None, :]
    else:
        # batched: map jnp.interp over batch axis with vmap (cheap)
        interp_one = jax.vmap(lambda y: jnp.interp(new_x, x_ds, y), in_axes=0, out_axes=0)
        return interp_one(y_ds)
