"""roq_likelihood_mlgw_bns_jax.py
ROQ likelihood helpers for mlgw_bns_jax — drop-in replacement for the
cell that defines _template_mlgw_bns and _roq_logL_single_det in
roq_pe_mlgw_bns_jax.ipynb.

Key difference from the notebook's original implementation
----------------------------------------------------------
The original notebook patched ``_gw_mod.template`` with a function that
calls the standard ``predict`` (from ``jax_import_n_predict``).  That
function always evaluates full PN terms on the model's internal ~6 200-pt
grid regardless of how many ROQ nodes you pass.  The ROQ therefore only
saved the *spline interpolation* step (from 6 200 → N_ROQ pts) but not the
PN evaluation (still O(6 200)).

This module instead uses ``load_predict_nodes`` (from ``jax_predict_roq``)
which evaluates PN directly at the N_ROQ ≈ 224 query frequencies, giving:

    PN cost  : O(224) instead of O(6 200) → ~28× cheaper per call
    Spline   : O(224 · log 6200) instead of O(6200 · log 6200)
    MLP/PCA  : unchanged

Usage in the notebook
---------------------
Replace cells 4 and 7 (template + likelihood) with:

    from roq_likelihood_mlgw_bns_jax import build_roq_template, build_roq_likelihood

    # After loading the model:
    template_nodes = build_roq_template("mlgw_bns_jax_model.h5")

    # After loading ROQ basis and building the tc-grid weights:
    roq_logL = build_roq_likelihood(
        template_nodes, f_lin_jax, n_det,
        batched_det, data_lin_grid_jax, M_hh_jax, dd_jax,
        tc_grid, N_TC_GRID,
    )
    # roq_logL(params_11) → scalar log-likelihood
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from jax_predict_roq import load_predict_nodes
from sharpy.utils import McQ2Masses


def build_roq_template(model_path: str):
    """Load model and return a SHARPy-compatible ROQ template function.

    The returned function has the same signature as the template in the
    notebook::

        template(params_13, frequency_array) -> (hp, hc)

    but uses ``predict_nodes`` internally, so evaluations at small
    ``frequency_array`` (ROQ nodes) are significantly faster.

    Parameters
    ----------
    model_path : str
        Path to the HDF5 model file.

    Returns
    -------
    template_nodes : callable
    """
    _predict_nodes = load_predict_nodes(model_path)

    def template_nodes(params, frequency_array):
        mc, q   = params[6], params[7]
        m1, m2  = McQ2Masses(mc, q)
        total_mass = m1 + m2

        chi1, chi2         = params[9], params[10]
        lambda_1, lambda_2 = params[11], params[12]
        phic               = params[4]
        dist_mpc           = jnp.exp(params[2])
        inclination        = params[3]

        mlgw_params = jnp.array([q, lambda_1, lambda_2, chi1, chi2])
        hp, hc = _predict_nodes(
            mlgw_params, frequency_array,
            total_mass=total_mass,
            distance_mpc=dist_mpc,
            inclination=inclination,
        )
        phase_factor = jnp.exp(-1j * phic)
        return hp * phase_factor, hc * phase_factor

    return template_nodes


def build_roq_likelihood(
    template_nodes,
    f_lin_jax,
    n_det: int,
    batched_det,
    data_lin_grid_jax,
    M_hh_jax: list,
    dd_jax,
    tc_grid,
    N_TC_GRID: int,
    FIXED_RA: float,
    FIXED_DEC: float,
):
    """Build the ROQ log-likelihood functions.

    Returns
    -------
    log_likelihood_roq_full : callable
        Full 13-parameter ROQ log-likelihood.
    log_likelihood_roq_reduced : callable
        11-parameter version with fixed RA/Dec.
    """
    from sharpy.GW_likelihood import antenna_pattern_functions
    from sharpy.utils import TimeDelayFromEarthCenter

    tc_min  = float(tc_grid[0])
    tc_step = float(tc_grid[1] - tc_grid[0])

    def _roq_logL_single_det(params_13, det_idx,
                              data_lin_grid_det, M_hh_det, dd):
        lat  = batched_det.latitude[det_idx]
        lon  = batched_det.longitude[det_idx]
        gam  = batched_det.gamma[det_idx]
        zeta = batched_det.zeta[det_idx]
        elev = batched_det.elevation[det_idx]
        trig = batched_det.trigtime[det_idx]

        fplus, fcross = antenna_pattern_functions(
            params_13, lat, lon, gam, zeta, trig)

        ra, dec = params_13[0], params_13[1]
        tc      = trig + params_13[8]
        timedelay = TimeDelayFromEarthCenter(lat, lon, elev, ra, dec, tc)
        timeshift = timedelay + params_13[8]

        idx_f = (timeshift - tc_min) / tc_step
        idx_lo = jnp.clip(jnp.floor(idx_f).astype(jnp.int32), 0, N_TC_GRID - 2)
        frac = idx_f - idx_lo
        d_lin_tc = ((1.0 - frac) * data_lin_grid_det[idx_lo]
                    +        frac  * data_lin_grid_det[idx_lo + 1])

        # Waveform evaluated ONLY at ROQ nodes (N_ROQ ≈ 224 pts)
        hp_lin, hc_lin = template_nodes(params_13, f_lin_jax)
        h_lin = fplus * hp_lin + fcross * hc_lin

        dh = jnp.real(jnp.sum(d_lin_tc * h_lin))
        hh = jnp.real(jnp.dot(jnp.conj(h_lin), M_hh_det @ h_lin))

        return dh - 0.5 * hh - 0.5 * dd

    def log_likelihood_roq_full(params_13):
        logL = jnp.float64(0.0)
        for i in range(n_det):
            logL = logL + _roq_logL_single_det(
                params_13, i,
                data_lin_grid_jax[i], M_hh_jax[i], dd_jax[i],
            )
        return logL

    def log_likelihood_roq_reduced(params_11):
        params_13 = jnp.concatenate([
            jnp.array([FIXED_RA, FIXED_DEC]),
            params_11[:4],
            params_11[4:9],
            params_11[9:11],
        ])
        return log_likelihood_roq_full(params_13)

    return log_likelihood_roq_full, log_likelihood_roq_reduced
