"""ROQ-optimised predictor for mlgw_bns_jax.

Wraps ``load_predict`` from ``jax_import_n_predict`` and adds
``load_predict_nodes``, which returns a waveform function that evaluates
the model **only at the ROQ empirical nodes** instead of the full
~253 k frequency grid.

Why this is faster
------------------
The standard ``predict`` path inside ``jax_import_n_predict.py`` works as
follows:

1. MLP forward pass on 5 intrinsic parameters → 11 847 PCA coefficients.
2. PCA back-projection → amplitude residuals on ~3 000 amp-grid points +
   phase residuals on ~6 000 phi-grid points (fixed "downsampled" grids
   baked into the HDF5 model file).
3. PN functions evaluated on those same ~6 000 pts (expensive for each call).
4. Spline of (residual × PN) to the *caller's* frequency array.

When the caller passes 253 k frequencies (full grid), step 4 is the
bottleneck.  But steps 2–3 always run on ~6 000 pts even if you pass only
224 ROQ nodes — so the ROQ gives almost no advantage in the standard path.

``load_predict_nodes`` fixes this:

* Steps 1–2 are unchanged (unavoidable).
* Step 3: PN is evaluated **directly** at the N_ROQ ≪ 6 000 query
  frequencies.
* Step 4: only the smooth ML residuals are splined to those query
  frequencies (cheaper and just as accurate, since the residuals are
  slowly varying).

Net result:
    PN cost  : O(N_ROQ) ≈ 224 instead of O(N_model_grid) ≈ 6 200
    Spline   : O(N_ROQ log N_model_grid) instead of O(N_full log N_model_grid)
    MLP/PCA  : unchanged (dominates anyway).

Usage
-----
    from jax_predict_roq import load_predict_nodes
    predict_nodes = load_predict_nodes("mlgw_bns_jax_model.h5")

    # Use exactly like the standard predict:
    hp, hc = predict_nodes(params, f_lin_jax,
                           total_mass=total_mass,
                           distance_mpc=dist_mpc,
                           inclination=inclination)

The signature, units and return values are **identical** to the standard
``predict`` from ``jax_import_n_predict``.
"""

from __future__ import annotations

import h5py
import jax
import jax.numpy as jnp
import numpy as np

from jax_import_n_predict import (
    _ACTIVATIONS,
    _AMP_SI_BASE,
    _TF2_BASE,
    _Af3hPN_jax,
    _PhifT7hPNComplete_jax,
    _PhifQM3hPN_jax,
    _Phif5hPN_jax,
    _compute_lambda_tilde_jax,
    _compute_delta_lambda_jax,
    _make_cubic_spline_jax,
    _smoothly_connect_with_zero_jax,
)

jax.config.update("jax_enable_x64", True)


def load_predict_nodes(path: str):
    """Load HDF5 model and return an ROQ-node-optimised predictor.

    Parameters
    ----------
    path : str
        Path to the HDF5 file produced by ``jax_export.py``.

    Returns
    -------
    predict_nodes : callable
        Signature identical to the standard ``predict`` from
        ``jax_import_n_predict.load_predict``::

            predict_nodes(params, frequencies_hz, total_mass,
                          distance_mpc, inclination) -> (hp, hc)

        But evaluates PN terms directly at ``frequencies_hz`` rather than
        on the model's internal ~6 200-point grid.  Pass the ROQ empirical
        node frequencies here for maximum speed.
    """
    with h5py.File(path, "r") as f:
        activation_name = f["mlp"].attrs["activation"]
        n_layers = int(f["mlp"].attrs["n_layers"])
        coefs = [jnp.array(f[f"mlp/coef_{i}"][...], dtype=jnp.float64)
                 for i in range(n_layers)]
        intercepts = [jnp.array(f[f"mlp/intercept_{i}"][...], dtype=jnp.float64)
                      for i in range(n_layers)]

        scaler_mean  = jnp.array(f["scaler/mean"][...],  dtype=jnp.float64)
        scaler_scale = jnp.array(f["scaler/scale"][...], dtype=jnp.float64)

        eigenvectors = jnp.array(f["pca/eigenvectors"][...],                   dtype=jnp.float64)
        eigenvalues  = jnp.array(f["pca/eigenvalues"][...],                    dtype=jnp.float64)
        pca_mean     = jnp.array(f["pca/mean"][...],                           dtype=jnp.float64)
        pca_scaling  = jnp.array(f["pca/principal_components_scaling"][...],   dtype=jnp.float64)
        pc_exponent  = float(f["pca"].attrs["pc_exponent"])

        frequencies_hz_np       = f["grid/frequencies_hz"][...]
        frequencies_natural_np  = f["grid/frequencies_natural"][...]
        amp_idx = f["grid/amplitude_indices"][...]
        phi_idx = f["grid/phase_indices"][...]
        M_ref   = float(f["grid"].attrs["total_mass"])

    activation         = _ACTIVATIONS[activation_name]
    eigenvalue_scaling = eigenvalues ** pc_exponent
    n_amp              = len(amp_idx)

    amp_freqs_hz_np      = frequencies_hz_np[amp_idx]
    amp_freqs_natural_np = frequencies_natural_np[amp_idx]
    phi_freqs_hz_np      = frequencies_hz_np[phi_idx]

    amp_freqs_hz_jax      = jnp.array(amp_freqs_hz_np,      dtype=jnp.float64)
    amp_freqs_natural_jax = jnp.array(amp_freqs_natural_np, dtype=jnp.float64)
    phi_freqs_hz_jax      = jnp.array(phi_freqs_hz_np,      dtype=jnp.float64)

    # Reference frequency for phase normalisation (= phi_freqs_hz[0])
    phi_ref_freq_jax = phi_freqs_hz_jax[0:1]

    # Spline evaluators for the *smooth ML residuals* on model grids
    amp_residual_spline = _make_cubic_spline_jax(amp_freqs_hz_np)
    phi_residual_spline = _make_cubic_spline_jax(phi_freqs_hz_np)

    # ------------------------------------------------------------------ #
    # MLP + PCA back-projection (identical to jax_import_n_predict)
    # ------------------------------------------------------------------ #

    def _mlp_forward(x: jnp.ndarray) -> jnp.ndarray:
        x = (x - scaler_mean) / scaler_scale
        for W, b in zip(coefs[:-1], intercepts[:-1]):
            x = activation(x @ W + b)
        return x @ coefs[-1] + intercepts[-1]

    def _nn_pca_predict(x: jnp.ndarray) -> jnp.ndarray:
        scaled_pca   = _mlp_forward(x)
        pca_comps    = scaled_pca / eigenvalue_scaling
        scaled_data  = pca_comps * pca_scaling
        zero_mean    = scaled_data @ eigenvectors.T
        return zero_mean + pca_mean

    # ------------------------------------------------------------------ #
    # ROQ-optimised predictor
    # ------------------------------------------------------------------ #

    def predict_nodes(
        params: jnp.ndarray,
        frequencies_hz: jnp.ndarray,
        total_mass: jnp.ndarray,
        distance_mpc: jnp.ndarray,
        inclination: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Evaluate (hp, hc) at arbitrary ``frequencies_hz``, optimised for
        small arrays (ROQ empirical nodes).

        Instead of evaluating all PN terms on the model's internal ~6 200-pt
        grid and then splining to the caller's frequencies, this function:

        1. Splines only the smooth MLP residuals from the model grid to the
           query frequencies (cheap, since residuals vary slowly).
        2. Evaluates all PN terms **directly** at the query frequencies
           (O(N_ROQ) ≪ O(N_model_grid)).

        The MLP + PCA forward pass is unchanged (unavoidable fixed cost).
        """
        q    = params[0]; lam1 = params[1]; lam2 = params[2]
        chi1 = params[3]; chi2 = params[4]
        eta  = q / (1.0 + q) ** 2
        m1   = M_ref / (1.0 + 1.0 / q)
        m2   = M_ref / (1.0 + q)

        lambdatilde = _compute_lambda_tilde_jax(m1, m2, lam1, lam2)
        dlambda     = _compute_delta_lambda_jax(m1, m2, lam1, lam2)

        # ── Step 1: MLP + PCA (fixed cost, independent of N_freq) ────────
        combined    = _nn_pca_predict(jnp.expand_dims(params, 0))
        log_amp_res = combined[0, :n_amp]   # residuals on amp model grid
        phi_res     = combined[0, n_amp:]   # residuals on phi model grid

        # ── Step 2: rescale query freqs to model reference mass frame ─────
        rescaled_freqs = frequencies_hz * (total_mass / M_ref)

        # ── Step 3: spline only smooth ML residuals to query freqs ────────
        log_amp_res_q = amp_residual_spline(log_amp_res, rescaled_freqs)
        phi_res_q     = phi_residual_spline(phi_res,     rescaled_freqs)

        # ── Step 4: PN amplitude directly at query frequencies ────────────
        f_nat_q  = jnp.interp(rescaled_freqs, amp_freqs_hz_jax, amp_freqs_natural_jax)
        pn_amp_q = _Af3hPN_jax(rescaled_freqs, M_ref, eta, chi1, chi2,
                                lambdatilde, dlambda)
        pn_amp_q = pn_amp_q * (_TF2_BASE * _AMP_SI_BASE / eta / M_ref ** 2)
        pn_amp_q = _smoothly_connect_with_zero_jax(f_nat_q, pn_amp_q)

        # ── Step 5: PN phase directly at query frequencies ────────────────
        phi_5pn_q   = _Phif5hPN_jax(rescaled_freqs, M_ref, eta, chi1, chi2)
        phi_tidal_q = _PhifT7hPNComplete_jax(rescaled_freqs, M_ref, eta, lam1, lam2)
        phi_qm_q    = _PhifQM3hPN_jax(rescaled_freqs, M_ref, eta, chi1, chi2,
                                       lam1, lam2)
        pn_phase_q  = -(phi_5pn_q + phi_tidal_q + phi_qm_q)

        # ── Step 6: PN phase reference at phi_freqs_hz[0] ────────────────
        # The original model sets pn_phase[0] = 0 (subtracts pn_phase at the
        # first phi-grid frequency).  We replicate that with one scalar call.
        phi_5pn_ref   = _Phif5hPN_jax(phi_ref_freq_jax, M_ref, eta, chi1, chi2)
        phi_tidal_ref = _PhifT7hPNComplete_jax(phi_ref_freq_jax, M_ref, eta,
                                               lam1, lam2)
        phi_qm_ref    = _PhifQM3hPN_jax(phi_ref_freq_jax, M_ref, eta, chi1, chi2,
                                         lam1, lam2)
        pn_phase_ref  = -(phi_5pn_ref + phi_tidal_ref + phi_qm_ref)[0]

        # ── Step 7: combine ───────────────────────────────────────────────
        amp = jnp.exp(log_amp_res_q) * pn_amp_q
        phi = phi_res_q + (pn_phase_q - pn_phase_ref)

        # ── Step 8: physical scaling ──────────────────────────────────────
        pre = total_mass ** 2 / _AMP_SI_BASE * eta / distance_mpc
        amp = amp * pre

        h_real = amp * jnp.cos(phi)
        h_imag = amp * jnp.sin(phi)

        cosi      = jnp.cos(inclination)
        pre_plus  = (1.0 + cosi ** 2) / 2.0
        pre_cross = cosi

        hp = pre_plus  * h_real + 1j * pre_plus  * h_imag
        hc = pre_cross * h_imag - 1j * pre_cross * h_real

        return hp, hc

    return predict_nodes
