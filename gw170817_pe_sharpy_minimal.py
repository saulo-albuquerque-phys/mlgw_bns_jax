"""Parameter estimation of GW170817 with SHARPy — minimal 6-parameter model.

Maximally reduced parameter space for fast PE:
  - Sky location fixed to NGC 4993 (ra, dec)
  - Polarization fixed (psi = 0)
  - Coalescence phase fixed (phic = 0)
  - Coalescence time fixed (tc = 0)
  - Two spins reduced to effective spin chi_eff (chi1 = chi2 = chi_eff)
  - Two tidal deformabilities reduced to Lambda_tilde (delta_Lambda_tilde = 0)

Sampled parameters (6):
    logdistance, inclination, mc, q, chi_eff, lambda_tilde

Usage
-----
    python gw170817_pe_sharpy_minimal.py

Requirements
------------
    pip install sharpy ripplegw
    The mlgw_bns_jax_model.h5 file must be accessible.
    GW170817 data files must be in gw170817_data/.
"""

from __future__ import annotations

import os
import sys
import time
from functools import partial

import numpy as np

# ── JAX configuration ────────────────────────────────────────────────
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

# ── Make sure jax_import_n_predict is importable ─────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from jax_import_n_predict import load_predict

# ── Load the mlgw_bns_jax waveform model ─────────────────────────────
MODEL_PATH = os.path.join(SCRIPT_DIR, "mlgw_bns_jax_model.h5")
_mlgw_predict = load_predict(MODEL_PATH)

# ── Monkey-patch sharpy's template to use mlgw_bns_jax ───────────────
import sharpy.GW_likelihood as _gw_mod
from sharpy.utils import McQ2Masses


def _template_mlgw_bns(params, frequency_array):
    """mlgw_bns_jax replacement for sharpy's template function."""
    mc = params[6]
    q = params[7]
    m1_msun, m2_msun = McQ2Masses(mc, q)
    total_mass = m1_msun + m2_msun
    chi1 = params[9]
    chi2 = params[10]
    lambda_1 = params[11]
    lambda_2 = params[12]
    phic = params[4]
    dist_mpc = jnp.exp(params[2])
    inclination = params[3]

    mlgw_params = jnp.array([q, lambda_1, lambda_2, chi1, chi2])

    hp, hc = _mlgw_predict(
        mlgw_params, frequency_array,
        total_mass=total_mass,
        distance_mpc=dist_mpc,
        inclination=inclination,
    )

    phase_factor = jnp.exp(-1j * phic)
    hp = hp * phase_factor
    hc = hc * phase_factor

    return hp, hc


_gw_mod.template = _template_mlgw_bns

from sharpy.GW_likelihood import GWNetwork, log_likelihood_det
from sharpy.smc_functions import run_sharpy
import sharpy.PSDs


# ── Tidal parameter conversion ───────────────────────────────────────

def lambda_tilde_to_lambdas(lambda_tilde, m1, m2, delta_lambda_tilde=0.0):
    r"""Convert (Lambda_tilde, delta_Lambda_tilde) to individual (lambda_1, lambda_2)."""
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


# ── GW170817 event parameters ────────────────────────────────────────
TRIGGER_TIME = 1187008882.43
SEGMENT_DURATION = 4.0               # analysis chunk around trigger (seconds)
SAMPLING_RATE = 4096                 # Hz
F_LOWER = 20.0                       # Hz
F_UPPER = 2000.0                     # Hz

# ── Fixed parameters ─────────────────────────────────────────────────
FIXED_RA = 3.44616       # rad  (NGC 4993)
FIXED_DEC = -0.408084    # rad  (NGC 4993)
FIXED_POL = 0.0          # rad  (polarization)
FIXED_PHIC = 0.0         # rad  (coalescence phase)
FIXED_TC = 0.0           # s    (coalescence time relative to trigger)

DATA_DIR = os.path.join(SCRIPT_DIR, "gw170817_data")
OUTDIR = "outdir_GW170817_sharpy_minimal"
LABEL = "GW170817_sharpy_minimal"


def main() -> None:
    os.makedirs(OUTDIR, exist_ok=True)

    # ── 1. Set up detector network with real GW170817 data ───────
    import glob

    detector_names = ["H1", "L1", "V1"]
    detector_settings = {}

    for det in detector_names:
        pattern = os.path.join(DATA_DIR, f"*{det}*GWOSC*.txt")
        matches = sorted(glob.glob(pattern))
        if not matches:
            print(f"WARNING: No data file found for {det} in {DATA_DIR}, skipping.")
            continue

        detector_settings[det] = {
            "data_file": matches[0],
            "channel": "GWOSC",
            "trigger_time": TRIGGER_TIME,
            "duration": SEGMENT_DURATION,
            "sampling_rate": SAMPLING_RATE,
            "f_lower": F_LOWER,
            "f_upper": F_UPPER,
            "psd_file": None,
            "psd_method": "welch",
            "download_data": False,
            "zero_noise": False,
        }

    if not detector_settings:
        raise FileNotFoundError(
            f"No data files found in {DATA_DIR}. "
            "Download from https://gwosc.org/eventapi/html/GWTC-1-confident/GW170817/"
        )

    print(f"Building GW network with detectors: {list(detector_settings.keys())}")
    t0 = time.time()
    gw_network = GWNetwork(detector_settings, injection_parameters=None)
    print(f"Network built in {time.time() - t0:.2f} s")

    batched_detector = gw_network.batched_detector

    # ── Full 13-param likelihood (SHARPy internal convention) ────
    log_likelihood_full = partial(log_likelihood_det, detector_list=batched_detector)

    # ── Wrapper: 6-param reduced vector → 13-param full vector ───
    # Reduced vector (what the sampler sees):
    #   [0] logdist, [1] incl,
    #   [2] mc, [3] q,
    #   [4] chi_eff, [5] lambda_tilde
    #
    # Full vector (what SHARPy expects):
    #   [0] ra, [1] dec, [2] logdist, [3] incl, [4] phic,
    #   [5] pol, [6] mc, [7] q, [8] tc,
    #   [9] chi1, [10] chi2, [11] lambda_1, [12] lambda_2

    def log_likelihood_reduced(params_6):
        """Expand 6 sampled params → full 13-param vector."""
        logdist = params_6[0]
        incl = params_6[1]
        mc = params_6[2]
        q = params_6[3]
        chi_eff = params_6[4]
        lambda_tilde = params_6[5]

        # chi_eff → chi1 = chi2 = chi_eff  (equal-spin assumption)
        chi1 = chi_eff
        chi2 = chi_eff

        # Lambda_tilde → (lambda_1, lambda_2) with delta_Lambda_tilde = 0
        m1, m2 = McQ2Masses(mc, q)
        l1, l2 = lambda_tilde_to_lambdas(lambda_tilde, m1, m2)
        l1 = jnp.clip(l1, 0.0)
        l2 = jnp.clip(l2, 0.0)

        params_13 = jnp.array([
            FIXED_RA,       # [0]  ra
            FIXED_DEC,      # [1]  dec
            logdist,        # [2]  logdistance
            incl,           # [3]  inclination
            FIXED_PHIC,     # [4]  phic
            FIXED_POL,      # [5]  pol
            mc,             # [6]  mc
            q,              # [7]  q
            FIXED_TC,       # [8]  tc
            chi1,           # [9]  chi1
            chi2,           # [10] chi2
            l1,             # [11] lambda_1
            l2,             # [12] lambda_2
        ])
        return log_likelihood_full(params_13)

    # ── 2. Define prior bounds (6 parameters) ────────────────────
    prior_bounds = jnp.array([
        [jnp.log(10.0), jnp.log(100.0)],  # logdistance (10–100 Mpc)
        [0.0,           jnp.pi],           # inclination
        [1.18,          1.21],             # mc  (chirp mass, M☉)
        [0.5,           1.0],              # q   (mass ratio)
        [-0.05,         0.05],             # chi_eff
        [0.0,           5000.0],           # lambda_tilde
    ])

    # 1 = periodic, 0 = reflective
    boundary_conditions = jnp.array([0, 0, 0, 0, 0, 0])

    parameter_names = [
        "logdistance", "theta_jn",
        "mc", "q", "chi_eff", "lambda_tilde",
    ]

    def prior(params):
        """Uniform prior (log-prior = 0 inside bounds)."""
        return 0.0

    # ── 3. Sampler settings ──────────────────────────────────────
    number_of_particles = 500
    step_size = 0.3
    alpha = 0.95
    seed = 42

    # ── 4. Run the SMC sampler ───────────────────────────────────
    print(f"\nFixed: ra={FIXED_RA:.5f}, dec={FIXED_DEC:.6f}, "
          f"pol={FIXED_POL:.1f}, phic={FIXED_PHIC:.1f}, tc={FIXED_TC:.1f}")
    print(f"Spin: chi1 = chi2 = chi_eff (equal-spin)")
    print(f"Tidal: delta_Lambda_tilde = 0")
    print(f"Sampling {len(parameter_names)} parameters")
    print(f"Starting SHARPy SMC sampler with {number_of_particles} particles...")
    start = time.time()

    result_dict = run_sharpy(
        log_likelihood_reduced,
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

    sampling_time = time.time() - start
    print(f"\nSampling completed in {sampling_time:.1f} s")

    samples = result_dict["posterior_samples"]
    logZ = result_dict["logZ"]
    dlogZ = result_dict["dlogZ"]
    print(f"log Z = {logZ:.2f} +/- {dlogZ:.2f}")

    # ── 5. Corner plot ───────────────────────────────────────────
    try:
        from corner import corner

        fig = corner(
            np.array(samples),
            show_titles=True,
            labels=parameter_names,
            title_kwargs={"fontsize": 12},
        )
        plot_path = os.path.join(OUTDIR, f"{LABEL}_corner.png")
        fig.savefig(plot_path)
        print(f"Corner plot saved to {plot_path}")
    except ImportError:
        print("corner package not installed — skipping corner plot.")

    print(f"Total time: {sampling_time:.1f} s")


if __name__ == "__main__":
    main()
