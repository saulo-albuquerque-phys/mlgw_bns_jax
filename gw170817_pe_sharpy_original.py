"""Full parameter estimation of GW170817 using the original SHARPy (ripplegw/IMRPhenomD).

This script works with the ORIGINAL (unmodified) SHARPy from
https://github.com/gabrieledemasi/sharpy, which uses ripplegw's
IMRPhenomD waveform model internally.

Since IMRPhenomD is a BBH waveform (no tidal effects), the parameter
vector has 9 components instead of 13.  The script monkey-patches the
``template`` function in sharpy.GW_likelihood to inject the mlgw_bns_jax
waveform (with tidal parameters) so that the full BNS physics is captured
while keeping the rest of SHARPy untouched.

Usage
-----
    python gw170817_pe_sharpy_original.py

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
# The original SHARPy uses ripplegw/IMRPhenomD (a BBH waveform with no
# tidal parameters).  We replace the template function to call
# mlgw_bns_jax instead, keeping the same 13-parameter vector convention:
#   [0] ra, [1] dec, [2] logdistance, [3] inclination,
#   [4] phic, [5] pol, [6] mc, [7] q, [8] tc,
#   [9] chi1, [10] chi2, [11] lambda_1, [12] lambda_2

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


# Replace the template used by the likelihood
_gw_mod.template = _template_mlgw_bns

from sharpy.GW_likelihood import GWNetwork, log_likelihood_det
from sharpy.smc_functions import run_sharpy
import sharpy.PSDs

# ── GW170817 event parameters ────────────────────────────────────────
TRIGGER_TIME = 1187008882.43
SEGMENT_DURATION = 4.0               # analysis chunk around trigger (seconds)
SAMPLING_RATE = 4096                 # Hz
F_LOWER = 20.0                       # Hz
F_UPPER = 2000.0                     # Hz

DATA_DIR = os.path.join(SCRIPT_DIR, "gw170817_data")
OUTDIR = "outdir_GW170817_sharpy"
LABEL = "GW170817_sharpy"


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
    log_likelihood = partial(log_likelihood_det, detector_list=batched_detector)

    # ── 2. Define prior bounds and boundary conditions ───────────
    # 13-parameter vector:
    #   [0] ra, [1] dec, [2] logdist, [3] incl, [4] phic,
    #   [5] pol, [6] mc, [7] q, [8] tc,
    #   [9] chi1, [10] chi2, [11] lambda_1, [12] lambda_2

    prior_bounds = jnp.array([
        [0.0,           2 * jnp.pi],       # ra
        [-jnp.pi / 2,   jnp.pi / 2],      # dec
        [jnp.log(10.0), jnp.log(100.0)],  # logdistance (10–100 Mpc)
        [0.0,           jnp.pi],           # inclination
        [0.0,           2 * jnp.pi],       # phic
        [0.0,           jnp.pi],           # pol
        [1.18,          1.21],             # mc  (chirp mass, M☉)
        [0.5,           1.0],              # q   (mass ratio)
        [-0.1,          0.1],              # tc  (relative to trigger, s)
        [-0.05,         0.05],             # chi1
        [-0.05,         0.05],             # chi2
        [0.0,           5000.0],           # lambda_1
        [0.0,           5000.0],           # lambda_2
    ])

    # 1 = periodic, 0 = reflective
    boundary_conditions = jnp.array([1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0])

    parameter_names = [
        "ra", "dec", "logdistance", "theta_jn", "phiref", "pol",
        "mc", "q", "tc", "chi1", "chi2", "lambda_1", "lambda_2",
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
    print(f"\nStarting SHARPy SMC sampler with {number_of_particles} particles...")
    start = time.time()

    result_dict = run_sharpy(
        log_likelihood,
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
