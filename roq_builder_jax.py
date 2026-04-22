#!/usr/bin/env python3
"""
JAX-based Reduced Order Quadrature (ROQ) basis builder.

This module re-implements the ROQ basis construction algorithm used by
JenpyROQ but relies entirely on JAX for waveform generation and linear
algebra.  It uses ``jax.jit`` for compiled waveform kernels and
``jax.vmap`` (with small batch sizes) for parallelism.

**Key design decisions for memory efficiency:**

- Waveform kernels are **JIT-compiled once** (warm-up call) before any
  batch evaluation.  Subsequent calls reuse the compiled XLA code.
- ``vmap`` batch size is kept small (default 8) so that peak memory for
  253 k frequency points stays under ~2 GB of intermediates.
- Training sets are **never materialised entirely** in RAM.  Both the
  greedy pre-selection and enrichment phases stream through waveforms in
  small batches, keeping only the current basis (~100 × 253 k × 16 B ≈
  400 MB) and one batch at a time in memory.

Algorithm
---------
The construction follows Field et al., Phys. Rev. X 4, 031006 (2014)
and the JenpyROQ implementation:

1. **Pre-selection** (corners + random seed): generate a small initial
   training set in streaming batches, normalise the waveforms, and
   greedily select basis vectors by choosing the waveform with the
   largest projection error.

2. **Enrichment** (iterative cycles): generate increasingly large fresh
   training sets (streamed) and augment the basis with any waveform
   whose representation error exceeds the tolerance.

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
import gc
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
    # Per-cycle relative tolerance: effective_tol = tolerance * rel_tol[i].
    # A value of 1.0 means use the absolute tolerance as-is.
    # A value < 1.0 (e.g. 0.1) is more aggressive: adds waveforms whose
    # error exceeds 10% of the absolute tolerance.
    training_set_rel_tol: list[float] = field(
        default_factory=lambda: [1.0, 1.0, 1.0]
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

    # Batching  –  keep small for 253 k-point grids to avoid OOM
    waveform_batch_size: int = 8

    # How many waveforms to hold in RAM at once for projection-error
    # computation during streaming greedy / enrichment.
    projection_batch_size: int = 200

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

        rel_tol_str = roq.get("training-set-rel-tol", ",".join(["1.0"] * len(sizes)))
        rel_tols = [float(s.strip()) for s in rel_tol_str.split(",")]
        # Pad or trim to match number of cycles
        while len(rel_tols) < len(sizes):
            rel_tols.append(1.0)
        rel_tols = rel_tols[: len(sizes)]

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
            training_set_rel_tol=rel_tols,
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
# Waveform generation (JAX)
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


def _build_single_waveform_fn(predict_fn, freqs_jax):
    """Build and return the JIT-compiled single-waveform function.

    Parameters
    ----------
    predict_fn : callable
        Raw (un-jitted) predictor from ``load_predict``.
    freqs_jax : jnp.ndarray
        Frequency grid (frozen).

    Returns
    -------
    single_fn : callable
        A JIT-compiled function ``p8 -> hp`` where ``p8`` is shape (8,)
        and ``hp`` is shape ``(n_freq,)`` complex128.
    """
    @jax.jit
    def _single(p8):
        mc  = p8[0]; q    = p8[1]
        s1z = p8[2]; s2z  = p8[3]
        lam1 = p8[4]; lam2 = p8[5]
        iota = p8[6]; phiref = p8[7]
        m1 = mc * (1.0 + q) ** 0.2 * q ** (-0.6)
        m2 = mc * (1.0 + q) ** 0.2 * q ** 0.4
        total_mass = m1 + m2
        mlgw_params = jnp.array([q, lam1, lam2, s1z, s2z])
        hp, _ = predict_fn(
            mlgw_params, freqs_jax,
            total_mass=total_mass,
            distance_mpc=jnp.array(1.0),
            inclination=iota,
        )
        hp = hp * jnp.exp(-1j * phiref)
        return hp
    return _single


def _build_vmap_waveform_fn(predict_fn, freqs_jax, batch_size: int):
    """Build a vmap-batched waveform function with fixed batch_size.

    The returned function accepts a padded array of shape
    ``(batch_size, 8)`` and returns ``(batch_size, n_freq)``.
    """
    single_fn = _build_single_waveform_fn(predict_fn, freqs_jax)
    return jax.jit(jax.vmap(single_fn))


def warmup_jit(predict_fn, frequencies: np.ndarray, batch_size: int,
               verbose: int = 1) -> tuple:
    """JIT-compile both the single-waveform and vmap-batched functions.

    Performs a throw-away evaluation so that all XLA compilation happens
    here.  Subsequent calls reuse the cached compiled code.

    Returns
    -------
    single_fn, batch_fn : callables
        JIT-compiled single and batch waveform generators.
    """
    freqs_jax = jnp.array(frequencies, dtype=jnp.float64)

    single_fn = _build_single_waveform_fn(predict_fn, freqs_jax)
    batch_fn = _build_vmap_waveform_fn(predict_fn, freqs_jax, batch_size)

    # ── Warm-up: compile single ─────────────────────────────────────
    if verbose:
        print("    JIT warm-up: compiling single-waveform kernel...")
    t0 = time.time()
    dummy_p = jnp.array([1.19, 1.0, 0.0, 0.0, 300.0, 300.0, 0.0, 0.0])
    _ = single_fn(dummy_p).block_until_ready()
    if verbose:
        print(f"    ✓ single compiled in {time.time() - t0:.1f}s")

    # ── Warm-up: compile vmap batch ─────────────────────────────────
    if verbose:
        print(f"    JIT warm-up: compiling vmap batch (size={batch_size}) kernel...")
    t0 = time.time()
    dummy_batch = jnp.tile(dummy_p, (batch_size, 1))
    _ = batch_fn(dummy_batch).block_until_ready()
    if verbose:
        print(f"    ✓ batch compiled in {time.time() - t0:.1f}s")

    return single_fn, batch_fn


def generate_waveforms_batch(
    predict_fn,
    params_batch: np.ndarray,
    frequencies: np.ndarray,
    distance_mpc: float = 1.0,
    batch_size: int = 8,
    _precompiled: tuple | None = None,
) -> np.ndarray:
    """Generate h_plus waveforms for a batch of parameters.

    Uses a pre-compiled vmap kernel with small ``batch_size`` to keep
    peak memory low.  If ``_precompiled`` is provided it must be
    ``(single_fn, batch_fn)`` from ``warmup_jit`` and will be reused
    without re-tracing.

    Parameters
    ----------
    predict_fn : callable
        The JAX predict function from ``load_predict``.
    params_batch : np.ndarray, shape ``(n, 8)``
        Parameters: [mc, q, s1z, s2z, lambda1, lambda2, iota, phiref].
    frequencies : np.ndarray, shape ``(n_freq,)``
        Frequency grid in Hz.
    distance_mpc : float
        Reference distance (irrelevant since we normalise later).
    batch_size : int
        vmap width.  8–16 is safe for 253 k-point grids on 15 GB RAM.
    _precompiled : tuple or None
        ``(single_fn, batch_fn)`` from ``warmup_jit``.

    Returns
    -------
    waveforms : np.ndarray, shape ``(n, n_freq)``, complex128
    """
    n = len(params_batch)
    freqs_jax = jnp.array(frequencies, dtype=jnp.float64)

    if _precompiled is not None:
        single_fn, batch_fn = _precompiled
    else:
        single_fn = _build_single_waveform_fn(predict_fn, freqs_jax)
        batch_fn = _build_vmap_waveform_fn(predict_fn, freqs_jax, batch_size)

    all_waveforms = []
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        chunk = params_batch[start:end]
        actual_size = len(chunk)

        if actual_size == batch_size:
            chunk_jax = jnp.array(chunk, dtype=jnp.float64)
            hp_batch = batch_fn(chunk_jax)
            hp_np = np.array(hp_batch, dtype=np.complex128)
        elif actual_size > 1:
            # Pad to batch_size so we reuse the same compiled kernel
            pad = np.tile(chunk[-1:], (batch_size - actual_size, 1))
            padded = np.concatenate([chunk, pad], axis=0)
            chunk_jax = jnp.array(padded, dtype=jnp.float64)
            hp_batch = batch_fn(chunk_jax)
            hp_np = np.array(hp_batch[:actual_size], dtype=np.complex128)
        else:
            # Single waveform – use the single-call kernel
            hp = single_fn(jnp.array(chunk[0], dtype=jnp.float64))
            hp_np = np.array(hp, dtype=np.complex128).reshape(1, -1)

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
    # <e_j | h_i> = delta_f * sum(conj(e_j) * h_i)
    overlaps = delta_f * (h @ np.conj(basis).T)  # shape (n, m)

    # ||P h||^2 = sum_j |<e_j|h>|^2
    proj_norm_sq = np.sum(np.abs(overlaps) ** 2, axis=-1)

    # error = ||h||^2 - ||P h||^2 = 1 - proj_norm_sq  (h normalised)
    errors = 1.0 - proj_norm_sq
    return np.maximum(errors, 0.0)


def projection_error_jax(
    h_batch: jnp.ndarray,
    basis: jnp.ndarray,
    delta_f: float,
) -> jnp.ndarray:
    """JAX version of projection_error for JIT acceleration.

    Parameters
    ----------
    h_batch : jnp.ndarray, shape ``(n, k)``
    basis : jnp.ndarray, shape ``(m, k)``
    delta_f : float

    Returns
    -------
    errors : jnp.ndarray, shape ``(n,)``
    """
    overlaps = delta_f * (h_batch @ jnp.conj(basis).T)
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
# Greedy basis construction  (in-memory, for small training sets)
# ─────────────────────────────────────────────────────────────────────

def greedy_basis(
    waveforms: np.ndarray,
    params: np.ndarray,
    delta_f: float,
    tolerance: float,
    max_basis: int = 500,
    verbose: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Greedy selection of reduced basis vectors (in-memory version).

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

    basis = np.empty((0, k), dtype=waveforms.dtype)
    basis = gram_schmidt_add(basis, waveforms[0], delta_f)
    basis_params = [params[0]]
    errors_history = []

    if verbose:
        print(f"  Greedy basis construction: {n} training waveforms, "
              f"tolerance={tolerance:.1e}")

    for step in range(1, max_basis):
        errors = projection_error(waveforms, basis, delta_f)
        max_err = float(errors.max())
        max_idx = int(errors.argmax())
        errors_history.append(max_err)

        if verbose:
            print(f"    Step {step:4d}: basis size = {len(basis)}, "
                  f"max error = {max_err:.6e}")

        if max_err < tolerance:
            if verbose:
                print(f"  ✓ Converged at step {step} with {len(basis)} basis vectors "
                      f"(error {max_err:.2e} < {tolerance:.1e})")
            break

        basis = gram_schmidt_add(basis, waveforms[max_idx], delta_f)
        basis_params.append(params[max_idx])
    else:
        if verbose:
            print(f"  ⚠ Reached max basis size {max_basis} without convergence "
                  f"(error {errors_history[-1]:.2e})")

    return basis, np.array(basis_params), np.array(errors_history)


# ─────────────────────────────────────────────────────────────────────
# Streaming greedy basis  (memory-efficient, for large training sets)
# ─────────────────────────────────────────────────────────────────────

def greedy_basis_streaming(
    all_params: np.ndarray,
    predict_fn,
    frequencies: np.ndarray,
    delta_f: float,
    tolerance: float,
    max_basis: int = 500,
    quadratic: bool = False,
    batch_size: int = 8,
    proj_batch: int = 200,
    verbose: int = 1,
    precompiled: tuple | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Greedy basis construction that streams waveforms in batches.

    On each greedy iteration the **entire** training parameter set is
    scanned in batches of ``proj_batch`` waveforms.  Only the current
    basis and one batch of waveforms are in RAM at the same time.

    Parameters
    ----------
    all_params : np.ndarray, shape ``(n, 8)``
        Full parameter set to scan.
    predict_fn : callable
        JAX predict function.
    frequencies : np.ndarray
        Frequency grid.
    delta_f : float
        Frequency spacing.
    tolerance : float
        Greedy stopping tolerance.
    max_basis : int
        Maximum number of basis vectors.
    quadratic : bool
        If True, operate on |h|^2 instead of h.
    batch_size : int
        vmap batch width for waveform generation.
    proj_batch : int
        Number of waveforms to generate and project at once.
    verbose : int
        Verbosity level.
    precompiled : tuple or None
        ``(single_fn, batch_fn)`` from ``warmup_jit``.

    Returns
    -------
    basis, basis_params, errors_history
    """
    n = len(all_params)
    k = len(frequencies)

    # Seed with first waveform
    first_wf = generate_waveforms_batch(
        predict_fn, all_params[:1], frequencies,
        batch_size=batch_size, _precompiled=precompiled,
    )
    if quadratic:
        first_wf = np.abs(first_wf) ** 2
    first_wf = normalise(first_wf, delta_f)

    basis = np.empty((0, k), dtype=first_wf.dtype)
    basis = gram_schmidt_add(basis, first_wf[0], delta_f)
    basis_params_list = [all_params[0]]
    errors_history = []

    if verbose:
        print(f"  Streaming greedy: {n} params, tolerance={tolerance:.1e}, "
              f"proj_batch={proj_batch}")

    for step in range(1, max_basis):
        t_step = time.time()
        global_max_err = 0.0
        global_max_wf = None
        global_max_p = None

        # Scan all training params in batches
        for bstart in range(0, n, proj_batch):
            bend = min(bstart + proj_batch, n)
            chunk_params = all_params[bstart:bend]

            # Generate waveforms for this chunk
            chunk_wf = generate_waveforms_batch(
                predict_fn, chunk_params, frequencies,
                batch_size=batch_size, _precompiled=precompiled,
            )
            if quadratic:
                chunk_wf = np.abs(chunk_wf) ** 2
            chunk_wf = normalise(chunk_wf, delta_f)

            # Project and find worst
            errs = projection_error(chunk_wf, basis, delta_f)
            local_max_idx = int(errs.argmax())
            local_max_err = float(errs[local_max_idx])

            if local_max_err > global_max_err:
                global_max_err = local_max_err
                global_max_wf = chunk_wf[local_max_idx].copy()
                global_max_p = chunk_params[local_max_idx].copy()

            del chunk_wf, errs
            gc.collect()

        errors_history.append(global_max_err)
        elapsed = time.time() - t_step

        if verbose:
            print(f"    Step {step:4d}: basis={len(basis)}, "
                  f"max_err={global_max_err:.6e}  ({elapsed:.1f}s)")

        if global_max_err < tolerance:
            if verbose:
                print(f"  ✓ Converged: {len(basis)} basis vectors "
                      f"(error {global_max_err:.2e} < {tolerance:.1e})")
            break

        basis = gram_schmidt_add(basis, global_max_wf, delta_f)
        basis_params_list.append(global_max_p)
    else:
        if verbose:
            print(f"  ⚠ Reached max basis {max_basis} "
                  f"(error {errors_history[-1]:.2e})")

    return basis, np.array(basis_params_list), np.array(errors_history)


# ─────────────────────────────────────────────────────────────────────
# Enrichment  (streaming)
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
    precompiled: tuple | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Iteratively enrich basis with streamed training waveforms.

    Training waveforms are generated and projected in batches of
    ``cfg.projection_batch_size``; only one batch is in RAM at a time.

    The effective threshold for adding a waveform to the basis in cycle *i*
    is ``tolerance * cfg.training_set_rel_tol[i]``.  A rel_tol < 1.0 means
    more waveforms are admitted (aggressive cycle); 1.0 is the standard threshold.

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
        Absolute greedy tolerance.
    quadratic : bool
        If True, build quadratic basis (|h|^2).
    rng : np.random.Generator, optional
        Random generator.
    precompiled : tuple or None
        ``(single_fn, batch_fn)`` from ``warmup_jit``.

    Returns
    -------
    basis, basis_params
    """
    if rng is None:
        rng = np.random.default_rng(cfg.random_seed + 1000)

    delta_f = cfg.delta_f
    kind = "quadratic" if quadratic else "linear"
    proj_bs = cfg.projection_batch_size

    # Pad rel_tol list to match number of cycles
    rel_tols = list(cfg.training_set_rel_tol)
    while len(rel_tols) < len(cfg.training_set_sizes):
        rel_tols.append(1.0)

    for cycle_idx, n_train in enumerate(cfg.training_set_sizes):
        t0 = time.time()
        cycle_tol = tolerance * rel_tols[cycle_idx]
        if cfg.verbose:
            print(f"\n  Enrichment cycle {cycle_idx + 1}/"
                  f"{len(cfg.training_set_sizes)}: "
                  f"{n_train} waveforms ({kind}), "
                  f"effective_tol={cycle_tol:.2e} "
                  f"(rel={rel_tols[cycle_idx]:.1f})")

        train_params = sample_parameters(rng, n_train, cfg)

        n_scanned = 0
        n_added = 0
        max_err_cycle = 0.0

        for bstart in range(0, n_train, proj_bs):
            bend = min(bstart + proj_bs, n_train)
            chunk_params = train_params[bstart:bend]

            chunk_wf = generate_waveforms_batch(
                predict_fn, chunk_params, frequencies,
                batch_size=cfg.waveform_batch_size,
                _precompiled=precompiled,
            )
            if quadratic:
                chunk_wf = np.abs(chunk_wf) ** 2
            chunk_wf = normalise(chunk_wf, delta_f)

            errs = projection_error(chunk_wf, basis, delta_f)
            local_max = float(errs.max())
            if local_max > max_err_cycle:
                max_err_cycle = local_max

            # Add outliers sorted by error (descending)
            outlier_idx = np.where(errs > cycle_tol)[0]
            if len(outlier_idx) > 0:
                order = outlier_idx[np.argsort(errs[outlier_idx])[::-1]]
                for idx in order:
                    h = chunk_wf[idx]
                    err = projection_error(h.reshape(1, -1), basis, delta_f)[0]
                    if err < cycle_tol:
                        continue
                    basis = gram_schmidt_add(basis, h, delta_f)
                    basis_params = np.vstack([basis_params, chunk_params[idx]])
                    n_added += 1

            n_scanned += len(chunk_params)
            del chunk_wf, errs
            gc.collect()

            if cfg.verbose and n_scanned % (proj_bs * 10) == 0:
                print(f"      scanned {n_scanned}/{n_train}, "
                      f"added {n_added}, basis={len(basis)}")

        elapsed = time.time() - t0
        if cfg.verbose:
            print(f"    Cycle done: scanned={n_scanned}, added={n_added}, "
                  f"basis={len(basis)}, max_err={max_err_cycle:.2e}, "
                  f"time={elapsed:.1f}s")

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
        Interpolant matrix B.  ``h_approx = B @ h[nodes]``.
    """
    m, k = basis.shape
    if m == 0:
        raise ValueError("Basis is empty.")

    nodes = np.empty(m, dtype=int)

    if verbose:
        print(f"\n  EIM: computing empirical nodes for {m} basis vectors "
              f"on {k} frequency points")

    # Step 1: first node
    nodes[0] = np.argmax(np.abs(basis[0]))

    for j in range(1, m):
        # V_mat[a,b] = basis[a, nodes[b]]  — shape (j, j)
        V_mat = basis[:j, :][:, nodes[:j]].T

        # c = V_mat^{-1} @ e_j(nodes)
        rhs = basis[j, nodes[:j]]
        c = np.linalg.solve(V_mat, rhs)

        # Residual: r(f) = e_j(f) - sum_a c_a e_a(f)
        residual = basis[j] - c @ basis[:j]

        nodes[j] = np.argmax(np.abs(residual))

        if verbose and (j + 1) % 50 == 0:
            print(f"    EIM step {j+1}/{m}: node at freq index {nodes[j]}")

    # Build full interpolant  B = basis.T @ V_full^{-1}
    V_full = basis[:, nodes].T  # (m, m)
    V_inv = np.linalg.inv(V_full)
    interpolant = basis.T @ V_inv  # (k, m)

    if verbose:
        print(f"  ✓ EIM complete: {m} nodes selected")
        print(f"    Node frequency range: [{nodes.min()}, {nodes.max()}] "
              f"(out of {k} points)")

    return nodes, interpolant


# ─────────────────────────────────────────────────────────────────────
# Full build pipeline
# ─────────────────────────────────────────────────────────────────────

def _save_phase_status(out_dir: Path, kind: str, phase: str, info: dict | None = None):
    """Write a JSON marker recording that *phase* completed successfully."""
    status_path = out_dir / f"_status_{kind}.json"
    status: dict = {}
    if status_path.exists():
        with open(status_path) as f:
            status = json.load(f)
    status[phase] = {
        "completed": True,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        **(info or {}),
    }
    with open(status_path, "w") as f:
        json.dump(status, f, indent=2)


def _phase_completed(out_dir: Path, kind: str, phase: str) -> bool:
    """Return True if *phase* was already marked as completed."""
    status_path = out_dir / f"_status_{kind}.json"
    if not status_path.exists():
        return False
    with open(status_path) as f:
        status = json.load(f)
    return status.get(phase, {}).get("completed", False)


def build_roq_basis(
    cfg: ROQConfig,
    kind: str = "linear",
    resume: bool = True,
) -> dict:
    """Full pipeline: JIT warm-up → pre-selection → enrichment → EIM.

    When ``resume=True`` (default), each phase checks for existing
    checkpoint files on disk.  If a phase's outputs already exist **and**
    its status marker says it completed successfully, that phase is
    skipped and the saved data are loaded instead.  This allows the
    build to be interrupted and restarted without losing progress.

    Resume points
    -------------
    * **After Phase 1 (pre-selection):**
      ``preselection_{kind}_basis.npy`` + status marker.
    * **After Phase 2 (enrichment):**
      ``basis_{kind}.npy`` + status marker.
    * **After Phase 3 (EIM):**
      ``empirical_nodes_{kind}.npy`` + status marker.

    Parameters
    ----------
    cfg : ROQConfig
        Configuration.
    kind : str
        ``"linear"`` or ``"quadratic"``.
    resume : bool
        If True, skip phases whose checkpoints already exist.

    Returns
    -------
    results : dict
        ``basis``, ``basis_params``, ``nodes``, ``interpolant``,
        ``frequencies``, ``empirical_frequencies``, ``errors_history``.
    """
    quadratic = kind == "quadratic"
    tolerance = cfg.tolerance_qua if quadratic else cfg.tolerance_lin
    n_pre = cfg.n_pre_basis_qua if quadratic else cfg.n_pre_basis_lin

    frequencies = cfg.frequencies
    delta_f = cfg.delta_f

    out_dir = Path(cfg.output_dir) / "ROQ_data" / kind
    out_dir.mkdir(parents=True, exist_ok=True)

    if cfg.verbose:
        print(f"\n{'='*65}")
        print(f"  Building {kind.upper()} ROQ basis")
        print(f"{'='*65}")
        print(f"  Frequency range: [{cfg.f_min}, {cfg.f_max}] Hz")
        print(f"  Frequency points: {cfg.n_freq}")
        print(f"  Tolerance: {tolerance:.1e}")
        print(f"  Pre-basis size target: {n_pre}")
        print(f"  Resume mode: {'ON' if resume else 'OFF'}")

    t_start = time.time()

    # ── Check if the entire build is already done ───────────────────
    if resume and _phase_completed(out_dir, kind, "eim"):
        if cfg.verbose:
            print(f"\n  ✓ {kind.upper()} basis already fully built — loading from disk")
        basis = np.load(out_dir / f"basis_{kind}.npy")
        basis_params = np.load(out_dir / f"basis_waveform_params_{kind}.npy")
        nodes = np.load(out_dir / f"empirical_nodes_{kind}.npy")
        interpolant = np.load(out_dir / f"basis_interpolant_{kind}.npy")
        empirical_freqs = np.load(out_dir / f"empirical_frequencies_{kind}.npy")
        err_path = out_dir / f"preselection_{kind}_basis_residual_modula.npy"
        errors_history = np.load(err_path) if err_path.exists() else np.array([])
        return {
            "basis": basis,
            "basis_params": basis_params,
            "nodes": nodes,
            "interpolant": interpolant,
            "frequencies": frequencies,
            "empirical_frequencies": empirical_freqs,
            "errors_history": errors_history,
        }

    # ── Load model + JIT compile ────────────────────────────────────
    if cfg.verbose:
        print(f"\n  Loading waveform model from {cfg.model_path}...")
    predict_fn = _load_predictor(cfg.model_path)

    if cfg.verbose:
        print(f"\n  Phase 0: JIT compilation")
    precompiled = warmup_jit(
        predict_fn, frequencies, cfg.waveform_batch_size,
        verbose=cfg.verbose,
    )

    # ── Phase 1: Pre-selection (streaming greedy) ───────────────────
    if resume and _phase_completed(out_dir, kind, "preselection"):
        if cfg.verbose:
            print(f"\n  Phase 1: Pre-selection — RESUMING from checkpoint")
        basis = np.load(out_dir / f"preselection_{kind}_basis.npy")
        basis_params = np.load(out_dir / f"preselection_{kind}_basis_waveform_params.npy")
        err_path = out_dir / f"preselection_{kind}_basis_residual_modula.npy"
        errors_history = np.load(err_path) if err_path.exists() else np.array([])
        if cfg.verbose:
            print(f"    Loaded pre-selection basis: {len(basis)} vectors")
    else:
        if cfg.verbose:
            print(f"\n  Phase 1: Pre-selection (streaming)")

        rng = np.random.default_rng(cfg.random_seed)

        corner_p = corner_parameters(cfg)
        n_random_seed = max(0, cfg.n_pre_basis_search_iter * n_pre - len(corner_p))
        random_seed_p = sample_parameters(rng, n_random_seed, cfg)
        pre_params = np.vstack([corner_p, random_seed_p])

        if cfg.verbose:
            print(f"    Pre-selection pool: {len(pre_params)} parameter sets "
                  f"({len(corner_p)} corners + {n_random_seed} random)")

        basis, basis_params, errors_history = greedy_basis_streaming(
            pre_params, predict_fn, frequencies, delta_f, tolerance,
            max_basis=n_pre,
            quadratic=quadratic,
            batch_size=cfg.waveform_batch_size,
            proj_batch=cfg.projection_batch_size,
            verbose=cfg.verbose,
            precompiled=precompiled,
        )

        # Save pre-selection checkpoint
        np.save(out_dir / f"preselection_{kind}_basis.npy", basis)
        np.save(out_dir / f"preselection_{kind}_basis_waveform_params.npy", basis_params)
        np.save(out_dir / f"preselection_{kind}_basis_residual_modula.npy", errors_history)
        _save_phase_status(out_dir, kind, "preselection",
                           {"basis_size": int(len(basis))})
        if cfg.verbose:
            print(f"    Saved pre-selection checkpoint to {out_dir}")

    # ── Phase 2: Enrichment (streaming) ─────────────────────────────
    if resume and _phase_completed(out_dir, kind, "enrichment"):
        if cfg.verbose:
            print(f"\n  Phase 2: Enrichment — RESUMING from checkpoint")
        basis = np.load(out_dir / f"basis_{kind}.npy")
        basis_params = np.load(out_dir / f"basis_waveform_params_{kind}.npy")
        if cfg.verbose:
            print(f"    Loaded enriched basis: {len(basis)} vectors")
    else:
        if cfg.verbose:
            print(f"\n  Phase 2: Enrichment (streaming)")

        rng_enrich = np.random.default_rng(cfg.random_seed + 2000)
        basis, basis_params = enrich_basis(
            basis, basis_params, predict_fn, frequencies, cfg,
            tolerance=tolerance, quadratic=quadratic, rng=rng_enrich,
            precompiled=precompiled,
        )

        # Save enriched basis checkpoint
        np.save(out_dir / f"basis_{kind}.npy", basis)
        np.save(out_dir / f"basis_waveform_params_{kind}.npy", basis_params)
        _save_phase_status(out_dir, kind, "enrichment",
                           {"basis_size": int(len(basis))})
        if cfg.verbose:
            print(f"    Saved enriched basis to {out_dir}")

    # ── Phase 3: EIM ────────────────────────────────────────────────
    if cfg.verbose:
        print(f"\n  Phase 3: Empirical Interpolation")

    nodes, interpolant = empirical_interpolation(basis, delta_f, verbose=cfg.verbose)

    empirical_freqs = frequencies[nodes]
    np.save(out_dir / f"empirical_nodes_{kind}.npy", nodes)
    np.save(out_dir / f"empirical_frequencies_{kind}.npy", empirical_freqs)
    np.save(out_dir / f"basis_interpolant_{kind}.npy", interpolant)
    _save_phase_status(out_dir, kind, "eim",
                       {"n_nodes": int(len(nodes))})

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
    precompiled: tuple | None = None,
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
    precompiled : tuple or None
        ``(single_fn, batch_fn)`` from ``warmup_jit``.

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

    errors = []
    # Stream validation waveforms in batches
    vbs = cfg.projection_batch_size
    for bstart in range(0, n_test, vbs):
        bend = min(bstart + vbs, n_test)
        chunk_p = test_params[bstart:bend]
        chunk_wf = generate_waveforms_batch(
            predict_fn, chunk_p, frequencies,
            batch_size=cfg.waveform_batch_size,
            _precompiled=precompiled,
        )
        if quadratic:
            chunk_wf = np.abs(chunk_wf) ** 2
        chunk_wf = normalise(chunk_wf, delta_f)

        for i in range(len(chunk_wf)):
            h = chunk_wf[i]
            h_approx = interpolant @ h[nodes]
            residual = h - h_approx
            err = delta_f * np.sum(np.abs(residual) ** 2)
            errors.append(err)

        del chunk_wf
        gc.collect()

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
    """Main entry point: build linear + quadratic ROQ bases.

    Supports ``--no-resume`` flag to force a fresh build.  By default,
    completed phases and completed basis kinds (linear / quadratic) are
    detected from checkpoint files on disk and skipped.
    """
    # ── CLI parsing ─────────────────────────────────────────────────
    resume = "--no-resume" not in sys.argv
    argv_rest = [a for a in sys.argv[1:] if a != "--no-resume"]
    config_file = argv_rest[0] if argv_rest else "config_roq_mlgw_bns_jax_gw170817.ini"

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
    print(f"  proj batch size  : {cfg.projection_batch_size}")
    print(f"  Resume mode      : {'ON' if resume else 'OFF'}")

    t0 = time.time()

    # ── LINEAR ──────────────────────────────────────────────────────
    results_lin = build_roq_basis(cfg, kind="linear", resume=resume)

    predict_fn = _load_predictor(cfg.model_path)
    precompiled = warmup_jit(
        predict_fn, cfg.frequencies, cfg.waveform_batch_size, verbose=0,
    )
    errors_lin, lin_ok = validate_basis(
        results_lin, predict_fn, cfg,
        n_test=200, quadratic=False, precompiled=precompiled,
    )

    # Free linear data before quadratic
    del results_lin
    gc.collect()

    # ── QUADRATIC ───────────────────────────────────────────────────
    results_qua = build_roq_basis(cfg, kind="quadratic", resume=resume)

    errors_qua, qua_ok = validate_basis(
        results_qua, predict_fn, cfg,
        n_test=200, quadratic=True, precompiled=precompiled,
    )

    t_total = time.time() - t0

    # ── Summary ─────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"  ROQ BUILD SUMMARY")
    print(f"{'='*65}")
    n_lin_nodes = len(np.load(
        Path(cfg.output_dir) / "ROQ_data/linear/empirical_nodes_linear.npy"))
    n_qua_nodes = len(results_qua["nodes"])
    n_lin_basis = len(np.load(
        Path(cfg.output_dir) / "ROQ_data/linear/basis_linear.npy"))
    print(f"  LINEAR  : {n_lin_basis:4d} basis vectors, "
          f"{n_lin_nodes:4d} nodes "
          f"({cfg.n_freq / n_lin_nodes:.0f}x speedup) "
          f"{'✓' if lin_ok else '✗'}")
    print(f"  QUAD    : {len(results_qua['basis']):4d} basis vectors, "
          f"{n_qua_nodes:4d} nodes "
          f"({cfg.n_freq / n_qua_nodes:.0f}x speedup) "
          f"{'✓' if qua_ok else '✗'}")
    print(f"  Total time: {t_total:.1f}s ({t_total / 60:.1f} min)")
    print(f"  Output dir: {cfg.output_dir}")

    # Save build metadata
    meta = {
        "linear_basis_size": int(n_lin_basis),
        "linear_nodes": int(n_lin_nodes),
        "quadratic_basis_size": int(len(results_qua["basis"])),
        "quadratic_nodes": int(n_qua_nodes),
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
