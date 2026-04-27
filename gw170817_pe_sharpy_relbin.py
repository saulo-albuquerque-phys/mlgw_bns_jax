"""GW170817 parameter estimation with SHARPy SMC + mlgw_bns_jax + relative binning.

Relative binning (Zackay+2018) reduces each likelihood evaluation from
O(N_freq ~ 3000) to O(N_bins ~ 400), enabling far more SMC particles to
run simultaneously in memory.

Fixed parameters (known from electro-magnetic observations):
    ra  = 3.44616 rad  (NGC 4993)
    dec = -0.408084 rad

Sampled parameters (6):
    logdistance, inclination, mc, q, chi_eff, lambda_tilde

Usage
-----
    python gw170817_pe_sharpy_relbin.py

Requirements
------------
    pip install sharpy ripplegw
    mlgw_bns_jax_model.h5 in the repository root.
    GW170817 data files in gw170817_data/.
"""

from __future__ import annotations

import os
import sys
import time
from functools import partial

import numpy as np

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

# ── Load mlgw_bns_jax ─────────────────────────────────────────────────
# ── ripplegw compatibility shim ───────────────────────────────────────
try:
    import ripplegw as _rw
    if not hasattr(_rw, "ms_to_Mc_eta"):
        from ripplegw.conversions import ms_to_Mc_eta as _ms2mc
        _rw.ms_to_Mc_eta = _ms2mc
except Exception:
    pass

from jax_import_n_predict import load_predict

MODEL_PATH = os.path.join(SCRIPT_DIR, "mlgw_bns_jax_model.h5")
_predict = load_predict(MODEL_PATH)

# ── Patch SHARPy template ─────────────────────────────────────────────
import sharpy.GW_likelihood as _gw_mod
from sharpy.utils import McQ2Masses


def _template_mlgw(params, f):
    """mlgw_bns_jax template for SHARPy.

    Applies the coalescence phase as an explicit rotation exp(-i·φ_c).
    The waveform polarisations are NOT conjugated — they are used directly
    in the inner product with the detector data.

    Parameters
    ----------
    params : array, shape (13,)
    f : array, shape (K,)

    Returns
    -------
    (hp, hc) : tuple of complex arrays, shape (K,)
    """
    mc, q = params[6], params[7]
    m1, m2 = McQ2Masses(mc, q)
    total_mass = m1 + m2
    chi1, chi2 = params[9], params[10]
    lam1, lam2 = params[11], params[12]
    phic = params[4]
    dist_mpc = jnp.exp(params[2])
    inclination = params[3]

    mlgw_params = jnp.array([q, lam1, lam2, chi1, chi2])
    hp, hc = _predict(
        mlgw_params, f,
        total_mass=total_mass,
        distance_mpc=dist_mpc,
        inclination=inclination,
    )
    # Coalescence-phase rotation — no conjugation
    phase_factor = jnp.exp(-1j * phic)
    return hp * phase_factor, hc * phase_factor


_gw_mod.template = _template_mlgw

from sharpy.GW_likelihood import GWNetwork, log_likelihood_det, antenna_pattern_functions
from sharpy.utils import TimeDelayFromEarthCenter
from sharpy.smc_functions import run_sharpy

from relative_binning import build_rb_likelihood

# ── Event configuration ───────────────────────────────────────────────
TRIGGER_TIME = 1187008882.43
SEGMENT_DURATION = 128.0
SAMPLING_RATE = 4096
F_LOWER = 23.0
F_UPPER = 2000.0
N_FREQ_RESAMPLED = 3000     # Lower grid: still fine for mlgw_bns_jax
DATA_START_GPS = 1187008114
DATA_DURATION = 1024

FIXED_RA = 3.44616
FIXED_DEC = -0.408084
FIXED_POL = 0.0
FIXED_PHIC = 0.0
FIXED_TC = 0.0

DATA_DIR = os.path.join(SCRIPT_DIR, "gw170817_data")
OUTDIR = "outdir_GW170817_sharpy_relbin"
LABEL = "GW170817_sharpy_relbin"

# Relative-binning configuration
N_RB_BINS = 400   # bins: accuracy vs memory tradeoff

# Fiducial parameters for relative binning
# (approximate MAP from previous analyses)
FIDUCIAL_PARAMS_13 = np.array([
    FIXED_RA,         # [0] ra
    FIXED_DEC,        # [1] dec
    np.log(40.0),     # [2] logdist ~ 40 Mpc
    2.5,              # [3] incl ~ 2.5 rad (near edge-on)
    0.0,              # [4] phic
    0.0,              # [5] pol
    1.186,            # [6] mc [M_sun]
    0.87,             # [7] q
    0.0,              # [8] tc
    0.0,              # [9] chi1
    0.0,              # [10] chi2
    300.0,            # [11] lambda_1
    300.0,            # [12] lambda_2
], dtype=np.float64)


# ── Tidal parameter conversion ─────────────────────────────────────────

def lambda_tilde_to_lambdas(lambda_tilde, m1, m2, delta_lambda_tilde=0.0):
    """Convert (Λ̃, δΛ̃) → (λ₁, λ₂)."""
    M = m1 + m2
    m1_4 = m1 ** 4
    m2_4 = m2 ** 4
    eta = (m1 * m2) / M ** 2
    X = jnp.sqrt(1.0 - 4.0 * eta)

    c1 = (16.0 / 13.0) * (m1 + 12.0 * m2) * m1_4 / M ** 5
    c2 = (16.0 / 13.0) * (m2 + 12.0 * m1) * m2_4 / M ** 5
    a_coeff = 1690.0 * eta / 1319.0 - 4843.0 / 1319.0
    b_coeff = 6162.0 * X / 1319.0
    d1 = (a_coeff + b_coeff) * m1_4 / M ** 4
    d2 = (-a_coeff + b_coeff) * m2_4 / M ** 4

    det = c1 * d2 - c2 * d1
    l1 = (d2 * lambda_tilde - c2 * delta_lambda_tilde) / det
    l2 = (c1 * delta_lambda_tilde - d1 * lambda_tilde) / det
    return l1, l2


def main() -> None:
    os.makedirs(OUTDIR, exist_ok=True)

    # ── 1. Load data, resample, build network ────────────────────────
    prefix = "BWCLEANED"
    test = os.path.join(DATA_DIR,
        f"H-H1_{prefix}_4KHZ-{DATA_START_GPS}-{DATA_DURATION}.txt")
    if not os.path.isfile(test):
        prefix = "CLEANED"
        print(f"BWCLEANED not found, using {prefix}")

    data_files = {
        det: os.path.join(DATA_DIR,
                f"{det[0]}-{det}_{prefix}_4KHZ-{DATA_START_GPS}-{DATA_DURATION}.txt")
        for det in ["H1", "L1", "V1"]
    }
    for det, fn in data_files.items():
        if not os.path.isfile(fn):
            print(f"WARNING: {fn} not found — skipping {det}")
            del data_files[det]

    det_settings = {
        det: dict(
            data_file=data_files[det], channel="GWOSC",
            trigger_time=TRIGGER_TIME, duration=SEGMENT_DURATION,
            sampling_rate=SAMPLING_RATE,
            # SHARPy reads 'f_lowr' (typo in source) for flow
            # and 'f_high' for fhigh — the 'f_lower'/'f_upper' keys are ignored
            f_lowr=F_LOWER, f_high=F_UPPER,
            f_lower=F_LOWER, f_upper=F_UPPER,  # kept for documentation
            psd_file=None, psd_method="welch",
            download_data=False, zero_noise=False,
        )
        for det in data_files
    }
    if not det_settings:
        raise FileNotFoundError(f"No data found in {DATA_DIR}")

    print(f"Building GW network: {list(det_settings.keys())} …")
    t0 = time.time()
    network = GWNetwork(det_settings, injection_parameters=None)
    print(f"  Done in {time.time()-t0:.1f} s")

    batched = network.batched_detector
    n_det = len(batched.latitude)

    # ── Resample to lower grid ───────────────────────────────────────
    from scipy.interpolate import interp1d as _interp1d

    f_new = np.linspace(F_LOWER, F_UPPER, N_FREQ_RESAMPLED)
    df_new = f_new[1] - f_new[0]
    new_data, new_psd = [], []
    for i in range(n_det):
        f_det = np.array(batched.Frequency[i])
        T_i = float(batched.T[i])
        phase_corr = 2.0 * np.pi * f_det * (T_i - 1.0)
        sf_rot = np.array(batched.FrequencySeries[i]) * np.exp(1j * phase_corr)
        r = _interp1d(f_det, sf_rot.real, kind="cubic",
                      bounds_error=False, fill_value=0.0)(f_new)
        im = _interp1d(f_det, sf_rot.imag, kind="cubic",
                       bounds_error=False, fill_value=0.0)(f_new)
        new_data.append(r + 1j * im)
        psd_i = np.array(batched.PowerSpectralDensity[i])
        lp = np.log(np.where(psd_i > 0, psd_i, 1e-100))
        new_psd.append(np.exp(_interp1d(f_det, lp, kind="cubic",
                                bounds_error=False, fill_value=np.log(1e-100))(f_new)))

    batched_rs = batched.replace(
        Frequency=jnp.stack([jnp.array(f_new, dtype=jnp.float64)] * n_det),
        FrequencySeries=jnp.stack([jnp.array(d, dtype=jnp.complex128) for d in new_data]),
        PowerSpectralDensity=jnp.stack([jnp.array(p, dtype=jnp.float64) for p in new_psd]),
        sigmasq=jnp.stack([jnp.array(p, dtype=jnp.float64) for p in new_psd]),
        TwoDeltaTOverN=jnp.stack([jnp.float64(2.0 * df_new)] * n_det),
        T=jnp.ones(n_det, dtype=jnp.float64),   # T=1: epoch shift absorbed into data
    )

    # Patch project_waveform for the resampled (epoch-rotated) convention
    def _project_rs(params, det_dict):
        f = det_dict.Frequency
        hp, hc = _gw_mod.template(params, f)
        fplus, fcross = antenna_pattern_functions(
            params,
            det_dict.latitude, det_dict.longitude,
            det_dict.gamma, det_dict.zeta, det_dict.trigtime,
        )
        ra, dec = params[0], params[1]
        tc = det_dict.trigtime + params[8]
        timedelay = TimeDelayFromEarthCenter(
            det_dict.latitude, det_dict.longitude, det_dict.elevation,
            ra, dec, tc,
        )
        timeshift = timedelay + params[8]  # no T-1 term (absorbed into data)
        shift = 2.0 * jnp.pi * f * timeshift
        return (fplus * hp + fcross * hc) * (jnp.cos(shift) - 1j * jnp.sin(shift))

    _gw_mod.project_waveform = _project_rs

    # ── 2. Build relative-binning likelihood ─────────────────────────
    print(f"\nBuilding relative-binning likelihood ({N_RB_BINS} bins) …")
    t0 = time.time()
    log_likelihood_rb, rb_network = build_rb_likelihood(
        batched_rs,
        FIDUCIAL_PARAMS_13,
        _template_mlgw,
        n_bins=N_RB_BINS,
    )
    print(f"  Done in {time.time()-t0:.1f} s")

    # ── 3. Reduced 6-parameter likelihood ───────────────────────────
    # Sampled: [logdist, incl, mc, q, chi_eff, lambda_tilde]
    def log_likelihood_6(params_6):
        logdist = params_6[0]
        incl = params_6[1]
        mc = params_6[2]
        q = params_6[3]
        chi_eff = params_6[4]
        lambda_tilde = params_6[5]

        chi1 = chi_eff
        chi2 = chi_eff

        m1, m2 = McQ2Masses(mc, q)
        l1, l2 = lambda_tilde_to_lambdas(lambda_tilde, m1, m2)
        l1 = jnp.clip(l1, 0.0)
        l2 = jnp.clip(l2, 0.0)

        params_13 = jnp.array([
            FIXED_RA, FIXED_DEC,
            logdist, incl,
            FIXED_PHIC, FIXED_POL,
            mc, q, FIXED_TC,
            chi1, chi2, l1, l2,
        ])
        return log_likelihood_rb(params_13)

    # ── 4. Prior bounds and sampler settings ─────────────────────────
    prior_bounds = jnp.array([
        [jnp.log(10.0), jnp.log(100.0)],   # logdist (10–100 Mpc)
        [0.0,            jnp.pi],            # inclination
        [1.18,           1.21],              # mc [M_sun]
        [0.5,            1.0],               # q
        [-0.05,          0.05],              # chi_eff
        [0.0,            5000.0],            # lambda_tilde
    ])
    boundary_conditions = jnp.array([0, 0, 0, 0, 0, 0])

    parameter_names = ["logdist", "incl", "mc", "q", "chi_eff", "lambda_tilde"]

    def prior(_p):
        return 0.0

    number_of_particles = 1000  # can be large thanks to RB memory saving
    step_size = 0.3
    alpha = 0.95
    seed = 42

    # ── 5. Run SMC sampler ───────────────────────────────────────────
    print(f"\nStarting SHARPy SMC ({number_of_particles} particles, RB likelihood) …")
    t_start = time.time()
    result = run_sharpy(
        log_likelihood_6,
        prior,
        prior_bounds,
        boundary_conditions,
        alpha,
        number_of_particles,
        step_size,
        jax.random.PRNGKey(seed),
        folder=OUTDIR,
        label=LABEL,
    )
    t_total = time.time() - t_start
    print(f"\nSampling done in {t_total:.1f} s")

    samples = result["posterior_samples"]
    logZ = result["logZ"]
    dlogZ = result["dlogZ"]
    print(f"log Z = {logZ:.2f} ± {dlogZ:.2f}")

    # ── 6. Corner plot ───────────────────────────────────────────────
    try:
        from corner import corner

        fig = corner(
            np.array(samples),
            labels=parameter_names,
            show_titles=True,
            title_kwargs={"fontsize": 11},
        )
        plot_path = os.path.join(OUTDIR, f"{LABEL}_corner.png")
        fig.savefig(plot_path, dpi=150)
        print(f"Corner plot: {plot_path}")
    except ImportError:
        print("corner not installed — skipping corner plot.")

    print(f"Total wall time: {t_total:.0f} s")


if __name__ == "__main__":
    main()
