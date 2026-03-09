"""Full parameter estimation of GW170817 using SHARPy's SMC sampler + mlgw_bns_jax.

This script loads real GW170817 data from local GWOSC files, builds the
detector network using SHARPy's GWNetwork, and runs parameter estimation
with SHARPy's Sequential Monte Carlo (SMC) sampler.

Usage
-----
    python gw170817_pe_sharpy.py

Requirements
------------
    pip install sharpy  (or install from the local sharpy/ folder)
    The mlgw_bns_jax_model.h5 file must be in the repository root.
    GW170817 data files must be in gw170817_data/.
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np

# ── JAX configuration ────────────────────────────────────────────────
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

# ── Ensure sharpy and jax_import_n_predict are importable ────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, os.path.join(SCRIPT_DIR, "sharpy"))

from sharpy.GW_likelihood import load_mlgw_bns_model, GWNetwork, log_likelihood_det
from sharpy.smc_functions import run_sharpy
import sharpy.PSDs

from functools import partial

# ── GW170817 event parameters ────────────────────────────────────────
TRIGGER_TIME = 1187008882.43         # GPS trigger time
DURATION = 32                        # seconds of data
SAMPLING_RATE = 4096                 # Hz
F_LOWER = 20.0                       # Hz
F_UPPER = 2000.0                     # Hz

MODEL_PATH = os.path.join(SCRIPT_DIR, "mlgw_bns_jax_model.h5")
DATA_DIR = os.path.join(SCRIPT_DIR, "gw170817_data")

OUTDIR = "outdir_GW170817_sharpy"
LABEL = "GW170817_sharpy"


def main() -> None:
    os.makedirs(OUTDIR, exist_ok=True)

    # ── 1. Load the waveform model ───────────────────────────────
    print(f"Loading mlgw_bns_jax model from: {MODEL_PATH}")
    load_mlgw_bns_model(MODEL_PATH)
    print("Model loaded.")

    # ── 2. Set up detector network with real GW170817 data ───────
    # Build detector settings pointing to local GWOSC text files.
    # SHARPy's noise.py reads LIGO-convention filenames:
    #   DET-TYPE-STARTTIME-DURATION.txt
    detector_names = ["H1", "L1", "V1"]
    detector_settings = {}

    import glob
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
            "duration": float(DURATION),
            "sampling_rate": SAMPLING_RATE,
            "f_lower": F_LOWER,
            "f_upper": F_UPPER,
            "psd_file": None,           # Estimate PSD from data via Welch
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

    # ── 3. Define prior bounds and boundary conditions ───────────
    # Parameter vector (13 components):
    #   [0] ra, [1] dec, [2] logdistance, [3] inclination,
    #   [4] phic, [5] pol, [6] mc, [7] q, [8] tc,
    #   [9] chi1, [10] chi2, [11] lambda_1, [12] lambda_2

    prior_bounds = jnp.array([
        [0.0,          2 * jnp.pi],     # ra
        [-jnp.pi / 2,  jnp.pi / 2],    # dec
        [jnp.log(10.0), jnp.log(100.0)],  # logdistance (10–100 Mpc)
        [0.0,          jnp.pi],         # inclination
        [0.0,          2 * jnp.pi],     # phic
        [0.0,          jnp.pi],         # pol
        [1.18,         1.21],           # mc  (chirp mass, M☉)
        [0.5,          1.0],            # q   (mass ratio)
        [-0.1,         0.1],            # tc  (relative to trigger, s)
        [-0.05,        0.05],           # chi1
        [-0.05,        0.05],           # chi2
        [0.0,          5000.0],         # lambda_1
        [0.0,          5000.0],         # lambda_2
    ])

    # 1 = periodic, 0 = reflective
    boundary_conditions = jnp.array([
        1,  # ra       (periodic)
        0,  # dec      (reflective)
        0,  # logdist  (reflective)
        0,  # incl     (reflective)
        1,  # phic     (periodic)
        1,  # pol      (periodic)
        0,  # mc       (reflective)
        0,  # q        (reflective)
        0,  # tc       (reflective)
        0,  # chi1     (reflective)
        0,  # chi2     (reflective)
        0,  # lambda_1 (reflective)
        0,  # lambda_2 (reflective)
    ])

    parameter_names = [
        "ra", "dec", "logdistance", "theta_jn", "phiref", "pol",
        "mc", "q", "tc", "chi1", "chi2", "lambda_1", "lambda_2",
    ]

    def prior(params):
        """Uniform prior (log-prior = 0 inside bounds)."""
        return 0.0

    # ── 4. Sampler settings ──────────────────────────────────────
    number_of_particles = 500
    step_size = 0.3
    alpha = 0.95
    seed = 42

    # ── 5. Run the SMC sampler ───────────────────────────────────
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

    # ── 6. Corner plot ───────────────────────────────────────────
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
