#!/usr/bin/env python3
"""
JAX-based Reduced Order Quadrature (ROQ) basis builder.

This module re-implements the ROQ basis construction algorithm used by
JenpyROQ but relies entirely on JAX for waveform generation and linear
algebra, exploiting ``jax.vmap`` for batch parallelism and ``jax.jit``
for compiled kernel performance.

Algorithm
---------
The construction follows Field et al., Phys. Rev. X 4, 031006 (2014)
and the JenpyROQ implementation:

1. **Pre-selection** (corners + random seed): generate a small initial
   training set, normalise the waveforms, and greedily select basis
   vectors by choosing the waveform with the largest projection error.

2. **Enrichment** (iterative cycles): generate increasingly large fresh
   training sets and augment the basis with any waveform whose
   representation error exceeds the tolerance.

3. **Empirical Interpolation Method (EIM)**: given the reduced basis,
   identify a sparse set of *empirical nodes* (frequency indices) and
   construct the interpolation matrix (basis interpolant) that allows
   reconstructing any representable waveform from its values at those
   nodes only.

The procedure is run separately for the *linear* basis (acting on h(f))
and the *quadratic* basis (acting on |h(f)|²).

Usage
-----
    python roq_builder_jax.py                        # default config
    python roq_builder_jax.py my_config.ini          # custom config

Dependencies
------------
    jax, jaxlib, numpy, h5py
    jax_import_n_predict  (local, for waveform generation)

Author
------
Auto-generated JAX ROQ builder for the mlgw_bns_jax project.
"""

from __future__ import annotations

import configparser
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

# Force CPU and enable float64 before any JAX import
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)


# ─────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────

@dataclass
class ROQConfig:
    """All parameters controlling an ROQ basis build."""

    # Waveform / frequency grid
    f_min: float = 23.0
    f_max: float = 2000.0
    seglen: float = 128.0

    # Tolerances
    tolerance_lin: float = 1e-4
    tolerance_qua: float = 1e-6

    # Pre-basis
    n_pre_basis_lin: int = 100
    n_pre_basis_qua: int = 5
    n_pre_basis_search_iter: int = 100

    # Enrichment
    n_training_set_cycles: int = 3
    training_set_sizes: list[int] = field(
        default_factory=lambda: [10_000, 50_000, 100_000]
    )

    # Training range (intrinsic params: mc, q, s1z, s2z, lambda1, lambda2)
    mc_range: tuple[float, float] = (1.18, 1.21)
    q_range: tuple[float, float] = (1.0, 2.0)
    s1z_range: tuple[float, float] = (-0.5, 0.5)
    s2z_range: tuple[float, float] = (-0.5, 0.5)
    lambda1_range: tuple[float, float] = (5.0, 5000.0)
    lambda2_range: tuple[float, float] = (5.0, 5000.0)
    iota_range: tuple[float, float] = (0.0, np.pi)
    phiref_range: tuple[float, float] = (0.0, 2 * np.pi)

    # I/O
    output_dir: str = "./roq_basis_mlgw_bns_jax"
    model_path: str = "mlgw_bns_jax_model.h5"
    random_seed: int = 170817
    verbose: int = 1

    # Batching
    waveform_batch_size: int = 512

    @property
    def delta_f(self) -> float:
        return 1.0 / self.seglen

    @property
    def n_freq(self) -> int:
        return int((self.f_max - self.f_min) / self.delta_f) + 1

    @property
    def frequencies(self) -> np.ndarray:
        return np.arange(self.f_min, self.f_max + self.delta_f / 2, self.delta_f)

    @classmethod
    def from_ini(cls, path: str) -> "ROQConfig":
        """Load from a JenpyROQ-style .ini config file."""
        c = configparser.ConfigParser()
        c.read(path)
        wf = c["Waveform_and_parametrisation"]
        roq = c["ROQ"]
        tr = c["Training_range"]
        io_sec = c["I/O"]

        sizes_str = roq.get("training-set-sizes", "10000,50000,100000")
        sizes = [int(s.strip()) for s in sizes_str.split(",")]

        return cls(
            f_min=float(wf.get("f-min", 23.0)),
            f_max=float(wf.get("f-max", 2000.0)),
            seglen=float(wf.get("seglen", 128.0)),
            tolerance_lin=float(roq.get("tolerance-lin", 1e-4)),
            tolerance_qua=float(roq.get("tolerance-qua", 1e-6)),
            n_pre_basis_lin=int(roq.get("n-pre-basis-lin", 100)),
            n_pre_basis_qua=int(roq.get("n-pre-basis-qua", 5)),
            n_pre_basis_search_iter=int(roq.get("n-pre-basis-search-iter", 100)),
            n_training_set_cycles=int(roq.get("n-training-set-cycles", 3)),
            training_set_sizes=sizes,
            mc_range=(float(tr.get("mc-min", 1.18)), float(tr.get("mc-max", 1.21))),
            q_range=(float(tr.get("q-min", 1.0)), float(tr.get("q-max", 2.0))),
            s1z_range=(float(tr.get("s1z-min", -0.5)), float(tr.get("s1z-max", 0.5))),
            s2z_range=(float(tr.get("s2z-min", -0.5)), float(tr.get("s2z-max", 0.5))),
            lambda1_range=(float(tr.get("lambda1-min", 5.0)), float(tr.get("lambda1-max", 5000.0))),
            lambda2_range=(float(tr.get("lambda2-min", 5.0)), float(tr.get("lambda2-max", 5000.0))),
            iota_range=(float(tr.get("iota-min", 0.0)), float(tr.get("iota-max", str(np.pi)))),
            phiref_range=(float(tr.get("phiref-min", 0.0)), float(tr.get("phiref-max", str(2 * np.pi)))),
            output_dir=io_sec.get("output", "./roq_basis_mlgw_bns_jax"),
            random_seed=int(io_sec.get("random-seed", 170817)),
            verbose=int(io_sec.get("verbose", 1)),
        )


# ─────────────────────────────────────────────────────────────────────
# Waveform generation (JAX, vmap-ready)
# ─────────────────────────────────────────────────────────────────────

def _load_predictor(model_path: str):
    """Load the JAX waveform predictor from the HDF5 model."""
    from jax_import_n_predict import load_predict
    return load_predict(model_path)


def _mcq_to_m1m2(mc: float, q: float):
    """Convert chirp mass and mass ratio to component masses."""
    factor = mc * (1.0 + q) ** 0.2
    m1 = factor * q ** (-0.6)
    m2 = factor * q ** 0.4
    return m1, m2


def sample_parameters(rng: np.random.Generator, n: int, cfg: ROQConfig) -> np.ndarray:
    """Draw n random parameter sets from the training range.

    Returns shape ``(n, 8)`` = ``[mc, q, s1z, s2z, lambda1, lambda2, iota, phiref]``.
    """
    params = np.column_stack([
        rng.uniform(cfg.mc_range[0], cfg.mc_range[1], n),
        rng.uniform(cfg.q_range[0], cfg.q_range[1], n),
        rng.uniform(cfg.s1z_range[0], cfg.s1z_range[1], n),
        rng.uniform(cfg.s2z_range[0], cfg.s2z_range[1], n),
        rng.uniform(cfg.lambda1_range[0], cfg.lambda1_range[1], n),
        rng.uniform(cfg.lambda2_range[0], cfg.lambda2_range[1], n),
        rng.uniform(cfg.iota_range[0], cfg.iota_range[1], n),
        rng.uniform(cfg.phiref_range[0], cfg.phiref_range[1], n),
    ])
    return params


def corner_parameters(cfg: ROQConfig) -> np.ndarray:
    """Generate parameters at the corners of the training range.

    Returns 2^6 corners for the 6 intrinsic parameters (mc, q, s1z, s2z,
    lambda1, lambda2), with fixed iota=0 and phiref=0.
    """
    corners = []
    ranges = [cfg.mc_range, cfg.q_range, cfg.s1z_range, cfg.s2z_range,
              cfg.lambda1_range, cfg.lambda2_range]
    # Generate all 2^6 = 64 corners
    for bits in range(2 ** len(ranges)):
        point = []
        for j, r in enumerate(ranges):
            point.append(r[(bits >> j) & 1])
        point.extend([0.0, 0.0])  # iota=0, phiref=0
        corners.append(point)
    return np.array(corners)


def generate_waveforms_batch(
    predict_fn,
    params_batch: np.ndarray,
    frequencies: np.ndarray,
    distance_mpc: float = 1.0,
    batch_size: int = 512,
) -> np.ndarray:
    """Generate normalised h_plus waveforms for a batch of parameters.

    Uses vmap for parallelism within each sub-batch.

    Parameters
    ----------
    predict_fn : callable
        The JAX predict function from ``load_predict``.
    params_batch : np.ndarray, shape ``(n, 8)``
        Parameters: [mc, q, s1z, s2z, lambda1, lambda2, iota, phiref].
    frequencies : np.ndarray, shape ``(n_freq,)``
        Frequency grid in Hz.
    distance_mpc : float
        Reference distance (doesn't matter much since we normalise).
    batch_size : int
        Number of waveforms to evaluate in parallel via vmap.

    Returns
    -------
    waveforms : np.ndarray, shape ``(n, n_freq)``, complex128
        Normalised h_plus waveforms.
    """
    n = len(params_batch)
    freqs_jax = jnp.array(frequencies, dtype=jnp.float64)

    def _single_waveform(p8):
        mc, q, s1z, s2z, lam1, lam2, iota, phiref = p8
        m1, m2 = mc * (1.0 + q) ** 0.2 * q ** (-0.6), mc * (1.0 + q) ** 0.2 * q ** 0.4
        total_mass = m1 + m2
        mlgw_params = jnp.array([q, lam1, lam2, s1z, s2z])
        hp, _ = predict_fn(
            mlgw_params, freqs_jax,
            total_mass=total_mass,
            distance_mpc=jnp.array(distance_mpc),
            inclination=iota,
        )
        # Apply phase factor
        hp = hp * jnp.exp(-1j * phiref)
        return hp

    _batch_waveform = jax.jit(jax.vmap(_single_waveform))

    all_waveforms = []
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        chunk = params_batch[start:end]
        actual_size = len(chunk)

        # Pad to batch_size for consistent JIT shapes
        if actual_size < batch_size:
            pad = np.zeros((batch_size - actual_size, 8))
            # Repeat last valid entry for padding
            pad[:] = chunk[-1]
            chunk = np.concatenate([chunk, pad], axis=0)

        chunk_jax = jnp.array(chunk, dtype=jnp.float64)
        hp_batch = _batch_waveform(chunk_jax)
        hp_np = np.array(hp_batch[:actual_size], dtype=np.complex128)
        all_waveforms.append(hp_np)

    return np.concatenate(all_waveforms, axis=0)


# ─────────────────────────────────────────────────────────────────────
# Linear algebra helpers
# ─────────────────────────────────────────────────────────────────────

def normalise(h: np.ndarray, delta_f: float) -> np.ndarray:
    """Normalise a waveform vector so that <h|h> = 1.

    For complex waveforms: <h|h> = delta_f * sum(|h|^2).
    """
    norm = np.sqrt(delta_f * np.sum(np.abs(h) ** 2, axis=-1, keepdims=True))
    norm = np.where(norm > 0, norm, 1.0)
    return h / norm


def scalar_product_batch(a: np.ndarray, b: np.ndarray, delta_f: float) -> np.ndarray:
    """Compute <a|b> = delta_f * sum(conj(a) * b) for batched inputs.

    a: shape (n, k) or (k,)
    b: shape (m, k) or (k,)
    Returns shape (n,) or (n, m) or scalar.
    """
    return delta_f * np.sum(np.conj(a) * b, axis=-1)


def projection_error(
    h: np.ndarray,
    basis: np.ndarray,
    delta_f: float,
) -> np.ndarray:
    """Compute the representation error ||h - P_V h||^2 for each h.

    Parameters
    ----------
    h : np.ndarray, shape ``(n, k)``
        Normalised waveforms to project.
    basis : np.ndarray, shape ``(m, k)``
        Orthonormalised basis vectors.
    delta_f : float
        Frequency spacing.

    Returns
    -------
    errors : np.ndarray, shape ``(n,)``
        Squared projection errors.
    """
    # Overlap matrix: (n, m) = <h_i | e_j>
    overlaps = delta_f * (np.conj(h) @ basis.T.conj())  # wrong
    # Actually: <h_i | e_j> = delta_f * sum_k conj(h_i[k]) * e_j[k]
    # Wait, for complex inner product we use h . conj(e):
    # <h|e> = delta_f * sum(conj(h) * e)
    # But for projection onto orthonormal basis: P h = sum_j <e_j|h> e_j
    # <e_j|h> = delta_f * sum(conj(e_j) * h)
    overlaps = delta_f * (h @ np.conj(basis).T)  # shape (n, m); overlaps[i,j] = <e_j | h_i>

    # ||P h||^2 = sum_j |<e_j|h>|^2
    proj_norm_sq = np.sum(np.abs(overlaps) ** 2, axis=-1)

    # ||h||^2 = 1 (already normalised)
    # error = ||h||^2 - ||P h||^2 = 1 - proj_norm_sq
    errors = 1.0 - proj_norm_sq
    # Clamp to avoid numerical negatives
    return np.maximum(errors, 0.0)


def projection_error_jax(
    h_batch: jnp.ndarray,
    basis: jnp.ndarray,
    delta_f: float,
) -> jnp.ndarray:
    """JAX version of projection_error for GPU/JIT acceleration.

    Parameters
    ----------
    h_batch : jnp.ndarray, shape ``(n, k)``
    basis : jnp.ndarray, shape ``(m, k)``
    delta_f : float

    Returns
    -------
    errors : jnp.ndarray, shape ``(n,)``
    """
    # <e_j | h_i> = delta_f * sum_k conj(e_j[k]) * h_i[k]
    overlaps = delta_f * (h_batch @ jnp.conj(basis).T)  # (n, m)
    proj_norm_sq = jnp.sum(jnp.abs(overlaps) ** 2, axis=-1)
    return jnp.maximum(1.0 - proj_norm_sq, 0.0)


_projection_error_jax_jit = jax.jit(projection_error_jax, static_argnums=(2,))


def gram_schmidt_add(basis: np.ndarray, new_vec: np.ndarray, delta_f: float) -> np.ndarray:
    """Add a vector to an orthonormal basis via modified Gram-Schmidt.

    Parameters
    ----------
    basis : np.ndarray, shape ``(m, k)``
        Existing orthonormal basis (may be empty, shape ``(0, k)``).
    new_vec : np.ndarray, shape ``(k,)``
        Vector to add (must not be in the span of basis).
    delta_f : float
        Frequency spacing for inner product.

    Returns
    -------
    new_basis : np.ndarray, shape ``(m+1, k)``
        Updated orthonormal basis.
    """
    v = new_vec.copy()
    for e in basis:
        # <e|v> = delta_f * sum(conj(e) * v)
        overlap = delta_f * np.sum(np.conj(e) * v)
        v = v - overlap * e

    # Re-orthogonalise (second pass for numerical stability)
    for e in basis:
        overlap = delta_f * np.sum(np.conj(e) * v)
        v = v - overlap * e

    norm = np.sqrt(delta_f * np.sum(np.abs(v) ** 2))
    if norm < 1e-15:
        raise ValueError("Vector is linearly dependent on the basis.")
    v = v / norm

    if len(basis) == 0:
        return v.reshape(1, -1)
    return np.vstack([basis, v.reshape(1, -1)])


# ─────────────────────────────────────────────────────────────────────
# Greedy basis construction
# ─────────────────────────────────────────────────────────────────────

def greedy_basis(
    waveforms: np.ndarray,
    params: np.ndarray,
    delta_f: float,
    tolerance: float,
    max_basis: int = 500,
    verbose: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Greedy selection of reduced basis vectors.

    Parameters
    ----------
    waveforms : np.ndarray, shape ``(n, k)``
        Normalised training waveforms.
    params : np.ndarray, shape ``(n, p)``
        Corresponding parameters.
    delta_f : float
        Frequency spacing.
    tolerance : float
        Stop when max error < tolerance.
    max_basis : int
        Maximum number of basis vectors.
    verbose : int
        Verbosity level.

    Returns
    -------
    basis : np.ndarray, shape ``(m, k)``
        Orthonormal reduced basis.
    basis_params : np.ndarray, shape ``(m, p)``
        Parameters of the selected basis waveforms.
    errors_history : np.ndarray, shape ``(m,)``
        Maximum error at each greedy step.
    """
    n, k = waveforms.shape
    if n == 0:
        raise ValueError("No waveforms to build basis from.")

    # Start with first waveform
    basis = np.empty((0, k), dtype=waveforms.dtype)
    basis = gram_schmidt_add(basis, waveforms[0], delta_f)
    basis_params = [params[0]]
    errors_history = []

    if verbose:
        print(f"  Greedy basis construction: {n} training waveforms, "
              f"tolerance={tolerance:.1e}")

    for step in range(1, max_basis):
        # Compute projection errors for all training waveforms
        # Use JAX for speed on large batches
        basis_jax = jnp.array(basis)
        wf_jax = jnp.array(waveforms)

        errors = np.array(_projection_error_jax_jit(wf_jax, basis_jax, delta_f))

        max_err = errors.max()
        max_idx = errors.argmax()
        errors_history.append(max_err)

        if verbose:
            print(f"    Step {step:4d}: basis size = {len(basis)}, "
                  f"max error = {max_err:.6e}")

        if max_err < tolerance:
            if verbose:
                print(f"  ✓ Converged at step {step} with {len(basis)} basis vectors "
                      f"(error {max_err:.2e} < {tolerance:.1e})")
            break

        # Add worst-represented waveform to the basis
        basis = gram_schmidt_add(basis, waveforms[max_idx], delta_f)
        basis_params.append(params[max_idx])

    else:
        if verbose:
            print(f"  ⚠ Reached max basis size {max_basis} without convergence "
                  f"(error {errors_history[-1]:.2e})")

    return basis, np.array(basis_params), np.array(errors_history)


# ─────────────────────────────────────────────────────────────────────
# Enrichment
# ─────────────────────────────────────────────────────────────────────

def enrich_basis(
    basis: np.ndarray,
    basis_params: np.ndarray,
    predict_fn,
    frequencies: np.ndarray,
    cfg: ROQConfig,
    tolerance: float,
    quadratic: bool = False,
    rng: Optional[np.random.Generator] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Iteratively enrich basis with new training waveforms.

    Parameters
    ----------
    basis : np.ndarray, shape ``(m, k)``
        Current orthonormal basis.
    basis_params : np.ndarray, shape ``(m, p)``
        Parameters of current basis elements.
    predict_fn : callable
        JAX waveform predictor.
    frequencies : np.ndarray
        Frequency grid.
    cfg : ROQConfig
        Configuration.
    tolerance : float
        Greedy tolerance.
    quadratic : bool
        If True, build quadratic basis (|h|^2).
    rng : np.random.Generator, optional
        Random generator (defaults to cfg.random_seed + 1000).

    Returns
    -------
    basis : np.ndarray, shape ``(m', k)``
        Enriched orthonormal basis.
    basis_params : np.ndarray, shape ``(m', p)``
        Updated parameters.
    """
    if rng is None:
        rng = np.random.default_rng(cfg.random_seed + 1000)

    delta_f = cfg.delta_f
    kind = "quadratic" if quadratic else "linear"

    for cycle_idx, n_train in enumerate(cfg.training_set_sizes):
        t0 = time.time()
        if cfg.verbose:
            print(f"\n  Enrichment cycle {cycle_idx + 1}/{len(cfg.training_set_sizes)}: "
                  f"{n_train} waveforms ({kind})")

        # Generate fresh training set
        train_params = sample_parameters(rng, n_train, cfg)
        if cfg.verbose:
            print(f"    Generating {n_train} waveforms...")
        train_wf = generate_waveforms_batch(
            predict_fn, train_params, frequencies,
            batch_size=cfg.waveform_batch_size,
        )

        if quadratic:
            train_wf = np.abs(train_wf) ** 2

        train_wf = normalise(train_wf, delta_f)

        # Compute errors
        basis_jax = jnp.array(basis)
        train_jax = jnp.array(train_wf)
        errors = np.array(_projection_error_jax_jit(train_jax, basis_jax, delta_f))

        n_outliers = np.sum(errors > tolerance)
        max_err = errors.max()

        if cfg.verbose:
            print(f"    Max error: {max_err:.6e}, outliers: {n_outliers}/{n_train}")

        if max_err < tolerance:
            if cfg.verbose:
                print(f"    ✓ No enrichment needed.")
            continue

        # Sort by error (descending) and add the worst ones greedily
        order = np.argsort(errors)[::-1]
        n_added = 0
        for idx in order:
            if errors[idx] < tolerance:
                break

            # Recompute error with current basis (it may have changed)
            h = train_wf[idx]
            err = projection_error(h.reshape(1, -1), basis, delta_f)[0]
            if err < tolerance:
                continue

            basis = gram_schmidt_add(basis, h, delta_f)
            basis_params = np.vstack([basis_params, train_params[idx]])
            n_added += 1

        elapsed = time.time() - t0
        if cfg.verbose:
            print(f"    Added {n_added} vectors (basis size: {len(basis)}) "
                  f"in {elapsed:.1f}s")

    return basis, basis_params


# ─────────────────────────────────────────────────────────────────────
# Empirical Interpolation Method (EIM)
# ─────────────────────────────────────────────────────────────────────

def empirical_interpolation(
    basis: np.ndarray,
    delta_f: float,
    verbose: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute empirical interpolation nodes and interpolant matrix.

    Implements the standard EIM algorithm:
    1. Find the frequency index where |e_0| is maximal → first node.
    2. For each subsequent basis vector e_j:
       a. Solve for interpolation coefficients using the existing nodes.
       b. Form the interpolation residual.
       c. The index of the maximum residual becomes the next node.
    3. Construct the interpolant matrix B such that for any representable
       waveform h, h ≈ B @ h[nodes].

    Parameters
    ----------
    basis : np.ndarray, shape ``(m, k)``
        Orthonormal reduced basis.
    delta_f : float
        Frequency spacing.
    verbose : int
        Verbosity level.

    Returns
    -------
    nodes : np.ndarray, shape ``(m,)``, int
        Indices of the empirical interpolation nodes.
    interpolant : np.ndarray, shape ``(k, m)``
        Interpolant matrix B. For a waveform h:
        ``h_approx = B @ h[nodes]``.
    """
    m, k = basis.shape
    if m == 0:
        raise ValueError("Basis is empty.")

    nodes = np.empty(m, dtype=int)
    # V_ij = basis[j, nodes[i]]  (interpolation matrix, built incrementally)

    if verbose:
        print(f"\n  EIM: computing empirical nodes for {m} basis vectors "
              f"on {k} frequency points")

    # Step 1: first node
    nodes[0] = np.argmax(np.abs(basis[0]))

    for j in range(1, m):
        # Solve V[:j, :j] c = basis[:j, :] at new point candidates
        # V is the interpolation matrix: V[a, b] = basis[b, nodes[a]]
        V = basis[:j, :][:, nodes[:j]]  # shape (j, j)

        # For each frequency index f, solve V c = basis[:j, f]
        # Then r[f] = basis[j, f] - basis[:j, f]^T @ c
        # But basis[:j, f] = V[a, f] for column... let me be careful.

        # Actually: V[a,b] = e_b(f_{nodes[a]}) where e_b is basis vector b
        # We want c such that V c = e_0..j-1 evaluated at new freq
        # The standard EIM residual is:
        # r(f) = e_j(f) - sum_i c_i * e_i(f) where c = V^{-1} e_j(nodes)

        # Let's do it clearly:
        V_mat = basis[:j, :][:, nodes[:j]].T  # shape (j, j); V_mat[a,b] = e_a(f_{nodes[b]})
        # Actually need to be careful about indexing.
        # V_mat[a,b] = basis[a, nodes[b]]

        # c = V_mat^{-1} @ [e_j(nodes[0]), ..., e_j(nodes[j-1])]
        rhs = basis[j, nodes[:j]]  # shape (j,)
        c = np.linalg.solve(V_mat, rhs)  # shape (j,)

        # Residual at every frequency: r(f) = e_j(f) - sum_a c_a * e_a(f)
        residual = basis[j] - c @ basis[:j]  # shape (k,)

        # New node: where |residual| is maximal
        nodes[j] = np.argmax(np.abs(residual))

        if verbose and (j + 1) % 50 == 0:
            print(f"    EIM step {j+1}/{m}: node at freq index {nodes[j]}")

    # Build the full interpolant matrix B
    # B is shape (k, m) such that h ≈ B @ h[nodes]
    # B = basis.T @ V_full^{-1}
    V_full = basis[:, nodes].T  # shape (m, m); V_full[a,b] = basis[a, nodes[b]]
    V_inv = np.linalg.inv(V_full)

    # B[f, j] = sum_a basis[a, f] * V_inv[a, j]
    interpolant = basis.T @ V_inv  # shape (k, m)

    if verbose:
        print(f"  ✓ EIM complete: {m} nodes selected")
        print(f"    Node frequency range: [{nodes.min()}, {nodes.max()}] "
              f"(out of {k} points)")

    return nodes, interpolant


# ─────────────────────────────────────────────────────────────────────
# Full build pipeline
# ─────────────────────────────────────────────────────────────────────

def build_roq_basis(
    cfg: ROQConfig,
    kind: str = "linear",
) -> dict:
    """Full pipeline: pre-selection → enrichment → EIM.

    Parameters
    ----------
    cfg : ROQConfig
        Configuration.
    kind : str
        ``"linear"`` or ``"quadratic"``.

    Returns
    -------
    results : dict
        Dictionary with keys:
        - ``basis``: orthonormal basis, shape ``(m, k)``
        - ``basis_params``: parameters, shape ``(m, p)``
        - ``nodes``: empirical node indices, shape ``(m,)``
        - ``interpolant``: interpolant matrix, shape ``(k, m)``
        - ``frequencies``: frequency grid
        - ``errors_history``: greedy error history
    """
    quadratic = kind == "quadratic"
    tolerance = cfg.tolerance_qua if quadratic else cfg.tolerance_lin
    n_pre = cfg.n_pre_basis_qua if quadratic else cfg.n_pre_basis_lin

    frequencies = cfg.frequencies
    delta_f = cfg.delta_f

    if cfg.verbose:
        print(f"\n{'='*65}")
        print(f"  Building {kind.upper()} ROQ basis")
        print(f"{'='*65}")
        print(f"  Frequency range: [{cfg.f_min}, {cfg.f_max}] Hz")
        print(f"  Frequency points: {cfg.n_freq}")
        print(f"  Tolerance: {tolerance:.1e}")
        print(f"  Pre-basis size target: {n_pre}")

    t_start = time.time()

    # Load waveform model
    if cfg.verbose:
        print(f"\n  Loading waveform model from {cfg.model_path}...")
    predict_fn = _load_predictor(cfg.model_path)

    # ── Phase 1: Pre-selection ──────────────────────────────────────
    if cfg.verbose:
        print(f"\n  Phase 1: Pre-selection")

    rng = np.random.default_rng(cfg.random_seed)

    # Generate corner waveforms + random seed waveforms
    corner_params = corner_parameters(cfg)
    n_random_seed = max(0, cfg.n_pre_basis_search_iter * n_pre - len(corner_params))
    random_seed_params = sample_parameters(rng, n_random_seed, cfg)
    pre_params = np.vstack([corner_params, random_seed_params])

    if cfg.verbose:
        print(f"    Generating {len(pre_params)} pre-selection waveforms "
              f"({len(corner_params)} corners + {n_random_seed} random)...")

    pre_wf = generate_waveforms_batch(
        predict_fn, pre_params, frequencies,
        batch_size=cfg.waveform_batch_size,
    )

    if quadratic:
        pre_wf = np.abs(pre_wf) ** 2

    pre_wf = normalise(pre_wf, delta_f)

    # Greedy selection
    basis, basis_params, errors_history = greedy_basis(
        pre_wf, pre_params, delta_f, tolerance,
        max_basis=n_pre, verbose=cfg.verbose,
    )

    # Save pre-selection checkpoint
    out_dir = Path(cfg.output_dir) / "ROQ_data" / kind
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / f"preselection_{kind}_basis.npy", basis)
    np.save(out_dir / f"preselection_{kind}_basis_waveform_params.npy", basis_params)
    np.save(out_dir / f"preselection_{kind}_basis_residual_modula.npy", errors_history)
    if cfg.verbose:
        print(f"    Saved pre-selection checkpoint to {out_dir}")

    # ── Phase 2: Enrichment ─────────────────────────────────────────
    if cfg.verbose:
        print(f"\n  Phase 2: Enrichment")

    rng_enrich = np.random.default_rng(cfg.random_seed + 2000)
    basis, basis_params = enrich_basis(
        basis, basis_params, predict_fn, frequencies, cfg,
        tolerance=tolerance, quadratic=quadratic, rng=rng_enrich,
    )

    # Save enriched basis checkpoint
    np.save(out_dir / f"basis_{kind}.npy", basis)
    np.save(out_dir / f"basis_waveform_params_{kind}.npy", basis_params)
    if cfg.verbose:
        print(f"    Saved enriched basis to {out_dir}")

    # ── Phase 3: EIM ────────────────────────────────────────────────
    if cfg.verbose:
        print(f"\n  Phase 3: Empirical Interpolation")

    nodes, interpolant = empirical_interpolation(basis, delta_f, verbose=cfg.verbose)

    # Save EIM outputs
    empirical_freqs = frequencies[nodes]
    np.save(out_dir / f"empirical_nodes_{kind}.npy", nodes)
    np.save(out_dir / f"empirical_frequencies_{kind}.npy", empirical_freqs)
    np.save(out_dir / f"basis_interpolant_{kind}.npy", interpolant)

    t_total = time.time() - t_start
    if cfg.verbose:
        print(f"\n  ✓ {kind.upper()} basis complete:")
        print(f"    Basis size         : {len(basis)}")
        print(f"    Frequency points   : {cfg.n_freq}")
        print(f"    Empirical nodes    : {len(nodes)}")
        print(f"    Compression ratio  : {cfg.n_freq / len(nodes):.0f}x")
        print(f"    Total time         : {t_total:.1f}s")

    return {
        "basis": basis,
        "basis_params": basis_params,
        "nodes": nodes,
        "interpolant": interpolant,
        "frequencies": frequencies,
        "empirical_frequencies": empirical_freqs,
        "errors_history": errors_history,
    }


# ─────────────────────────────────────────────────────────────────────
# Validation
# ─────────────────────────────────────────────────────────────────────

def validate_basis(
    results: dict,
    predict_fn,
    cfg: ROQConfig,
    n_test: int = 200,
    quadratic: bool = False,
) -> tuple[np.ndarray, bool]:
    """Validate the ROQ basis against random unseen waveforms.

    Parameters
    ----------
    results : dict
        Output from ``build_roq_basis``.
    predict_fn : callable
        JAX predict function.
    cfg : ROQConfig
        Configuration.
    n_test : int
        Number of test waveforms.
    quadratic : bool
        If True, test quadratic basis.

    Returns
    -------
    errors : np.ndarray, shape ``(n_test,)``
        Representation errors.
    passed : bool
        Whether all errors are below 10x tolerance.
    """
    tolerance = cfg.tolerance_qua if quadratic else cfg.tolerance_lin
    delta_f = cfg.delta_f
    frequencies = cfg.frequencies
    interpolant = results["interpolant"]
    nodes = results["nodes"]

    rng = np.random.default_rng(42)
    test_params = sample_parameters(rng, n_test, cfg)
    test_wf = generate_waveforms_batch(
        predict_fn, test_params, frequencies,
        batch_size=cfg.waveform_batch_size,
    )

    if quadratic:
        test_wf = np.abs(test_wf) ** 2

    test_wf = normalise(test_wf, delta_f)

    errors = []
    for i in range(n_test):
        h = test_wf[i]
        h_approx = interpolant @ h[nodes]
        residual = h - h_approx
        err = delta_f * np.sum(np.abs(residual) ** 2)
        errors.append(err)

    errors = np.array(errors)
    max_err = errors.max()
    passed = max_err < tolerance * 10

    kind = "QUADRATIC" if quadratic else "LINEAR"
    print(f"\n  Validation ({kind}): {n_test} waveforms")
    print(f"    Max error  : {max_err:.2e}")
    print(f"    Mean error : {errors.mean():.2e}")
    print(f"    Tolerance  : {tolerance:.1e} (pass threshold: {tolerance * 10:.1e})")
    print(f"    Result     : {'✓ PASSED' if passed else '✗ FAILED'}")

    return errors, passed


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────

def main():
    """Main entry point: build linear + quadratic ROQ bases."""
    config_file = sys.argv[1] if len(sys.argv) > 1 else "config_roq_mlgw_bns_jax_gw170817.ini"

    if os.path.isfile(config_file):
        print(f"Loading config from {config_file}")
        cfg = ROQConfig.from_ini(config_file)
    else:
        print(f"Config file {config_file} not found, using defaults.")
        cfg = ROQConfig()

    print(f"\n{'='*65}")
    print(f"  JAX ROQ Basis Builder")
    print(f"{'='*65}")
    print(f"  Frequency range  : [{cfg.f_min}, {cfg.f_max}] Hz")
    print(f"  Segment length   : {cfg.seglen} s")
    print(f"  Frequency points : {cfg.n_freq}")
    print(f"  Tolerance (lin)  : {cfg.tolerance_lin:.1e}")
    print(f"  Tolerance (qua)  : {cfg.tolerance_qua:.1e}")
    print(f"  Training cycles  : {cfg.training_set_sizes}")
    print(f"  vmap batch size  : {cfg.waveform_batch_size}")

    t0 = time.time()

    # ── LINEAR ──────────────────────────────────────────────────────
    results_lin = build_roq_basis(cfg, kind="linear")

    # Free memory before quadratic
    predict_fn_for_validate = _load_predictor(cfg.model_path)
    errors_lin, lin_ok = validate_basis(
        results_lin, predict_fn_for_validate, cfg,
        n_test=200, quadratic=False,
    )

    # ── QUADRATIC ───────────────────────────────────────────────────
    results_qua = build_roq_basis(cfg, kind="quadratic")

    errors_qua, qua_ok = validate_basis(
        results_qua, predict_fn_for_validate, cfg,
        n_test=200, quadratic=True,
    )

    t_total = time.time() - t0

    # ── Summary ─────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"  ROQ BUILD SUMMARY")
    print(f"{'='*65}")
    print(f"  LINEAR  : {len(results_lin['basis']):4d} basis vectors, "
          f"{len(results_lin['nodes']):4d} nodes "
          f"({cfg.n_freq / len(results_lin['nodes']):.0f}x speedup) "
          f"{'✓' if lin_ok else '✗'}")
    print(f"  QUAD    : {len(results_qua['basis']):4d} basis vectors, "
          f"{len(results_qua['nodes']):4d} nodes "
          f"({cfg.n_freq / len(results_qua['nodes']):.0f}x speedup) "
          f"{'✓' if qua_ok else '✗'}")
    print(f"  Total time: {t_total:.1f}s ({t_total / 60:.1f} min)")
    print(f"  Output dir: {cfg.output_dir}")

    # Save build metadata
    meta = {
        "linear_basis_size": int(len(results_lin["basis"])),
        "linear_nodes": int(len(results_lin["nodes"])),
        "quadratic_basis_size": int(len(results_qua["basis"])),
        "quadratic_nodes": int(len(results_qua["nodes"])),
        "n_freq": int(cfg.n_freq),
        "f_min": cfg.f_min,
        "f_max": cfg.f_max,
        "seglen": cfg.seglen,
        "tolerance_lin": cfg.tolerance_lin,
        "tolerance_qua": cfg.tolerance_qua,
        "total_time_s": t_total,
        "linear_validation_passed": bool(lin_ok),
        "quadratic_validation_passed": bool(qua_ok),
    }
    meta_path = Path(cfg.output_dir) / "build_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\n  Build metadata saved to {meta_path}")


if __name__ == "__main__":
    main()
