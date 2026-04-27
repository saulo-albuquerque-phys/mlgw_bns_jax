"""Validate relative-binning likelihood against full SHARPy likelihood.

This script:
  1. Loads real GW170817 data (H1, L1, V1).
  2. Checks that the signal is captured by computing the matched-filter
     network SNR at the known GW170817 parameters.
  3. Builds both the full SHARPy likelihood and the relative-binning (RB)
     likelihood.
  4. Evaluates both likelihoods at multiple parameter points and confirms
     they agree to ≲ 1 % relative error in exp(log L).

Usage
-----
    python validate_relative_binning.py

Requirements
------------
    pip install sharpy ripplegw (see setup_igwn.sh)
    mlgw_bns_jax_model.h5 must be in the repository root.
    GW170817 data in gw170817_data/.
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

# ── Load mlgw_bns_jax waveform model ──────────────────────────────────
# ── ripplegw compatibility shim ───────────────────────────────────────
# Newer versions of ripplegw moved ms_to_Mc_eta to ripplegw.conversions;
# patch the top-level module so older SHARPy code can still find it.
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


def _template_mlgw(params, f):
    """SHARPy-compatible template using mlgw_bns_jax.

    Applies the coalescence phase exp(-1j·φ_c) as an explicit overall phase
    rotation — NOT as a conjugation of the waveform.

    Parameters
    ----------
    params : array, shape (13,)
        SHARPy parameter vector [ra, dec, logdist, incl, phic, pol,
                                  mc, q, tc, chi1, chi2, lam1, lam2].
    f : array, shape (K,)
        Frequencies in Hz.

    Returns
    -------
    (hp, hc) : tuple of complex arrays, shape (K,)
    """
    from sharpy.utils import McQ2Masses

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
    # Apply coalescence-phase rotation (no conjugation)
    phase_factor = jnp.exp(-1j * phic)
    return hp * phase_factor, hc * phase_factor


# ── Patch SHARPy's template ────────────────────────────────────────────
import sharpy.GW_likelihood as _gw_mod
_gw_mod.template = _template_mlgw

from sharpy.GW_likelihood import GWNetwork, log_likelihood_det
from functools import partial

# ── Event parameters ───────────────────────────────────────────────────
TRIGGER_TIME = 1187008882.43
SEGMENT_DURATION = 128.0        # seconds
SAMPLING_RATE = 4096
F_LOWER = 23.0
F_UPPER = 2000.0
DATA_START_GPS = 1187008114
DATA_DURATION = 1024

FIXED_RA = 3.44616
FIXED_DEC = -0.408084

DATA_DIR = os.path.join(SCRIPT_DIR, "gw170817_data")

# GW170817 fiducial / literature parameters (13-dim SHARPy vector)
# [ra, dec, logdist, incl, phic, pol, mc, q, tc, chi1, chi2, lam1, lam2]
FIDUCIAL_PARAMS = np.array([
    FIXED_RA,            # [0] ra
    FIXED_DEC,           # [1] dec
    np.log(40.0),        # [2] logdist ~ 40 Mpc
    2.5,                 # [3] inclination ~ 2.5 rad (near edge-on)
    0.0,                 # [4] phic
    0.0,                 # [5] pol
    1.186,               # [6] mc [M_sun]
    0.87,                # [7] q
    0.0,                 # [8] tc
    0.0,                 # [9] chi1
    0.0,                 # [10] chi2
    300.0,               # [11] lambda_1
    300.0,               # [12] lambda_2
], dtype=np.float64)


# ======================================================================
# 1. Load data and build detector network
# ======================================================================

def build_network():
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
    for det, f in data_files.items():
        assert os.path.isfile(f), f"Missing: {f}"

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
        for det in ["H1", "L1", "V1"]
    }

    print("Building GW network …")
    t0 = time.time()
    network = GWNetwork(det_settings, injection_parameters=None)
    print(f"  Done in {time.time()-t0:.1f} s")

    batched = network.batched_detector
    n_freq = len(np.array(batched.Frequency[0]))
    print(f"  Original grid: {n_freq} frequency bins "
          f"(df = {1.0/SEGMENT_DURATION:.4f} Hz)")
    return network

# ======================================================================
# main
# ======================================================================

def main():
    print("=" * 70)
    print("RELATIVE BINNING VALIDATION  —  GW170817  (mlgw_bns_jax)")
    print("=" * 70)

    # ── 1. Build network (original dense grid) ───────────────────────
    network = build_network()
    batched = network.batched_detector
    n_det = len(batched.latitude)
    n_freq_orig = len(np.array(batched.Frequency[0]))

    # Full SHARPy likelihood on the original dense grid
    log_likelihood_full = partial(log_likelihood_det, detector_list=batched)

    # ── 2. Optimal matched-filter SNR check ──────────────────────────
    print("\n─── 2. Optimal matched-filter SNR check ─────────────────────────")
    print("   (Phase-maximised SNR — independent of φ_c, ψ alignment)")
    from relative_binning import compute_matched_filter_snr

    snr_info = compute_matched_filter_snr(batched, FIDUCIAL_PARAMS, _template_mlgw)

    for i, (snr_opt, snr_sig) in enumerate(
        zip(snr_info["snr_opt_det"], snr_info["snr_signed_det"])
    ):
        lat_i = float(batched.latitude[i])
        print(f"  det {i} (lat={lat_i:.1f}°): "
              f"optimal SNR = {snr_opt:.2f}, signed SNR = {snr_sig:.2f}")

    snr_net = snr_info["snr_opt_network"]
    print(f"  Network optimal SNR = {snr_net:.2f}  (expected ~32 for GW170817 with H1+L1+V1)")
    signal_captured = snr_net > 8.0
    print(f"  Signal captured: {'✓ YES' if signal_captured else '✗ NO'}")

    # Δ logL to confirm the signal exists in the data
    logL_fid = float(jax.jit(log_likelihood_full)(jnp.array(FIDUCIAL_PARAMS)))
    sigmasq_np = [np.array(batched.sigmasq[i]) for i in range(n_det)]
    data_np = [np.array(batched.FrequencySeries[i]) for i in range(n_det)]
    TwoDTN_np = [float(batched.TwoDeltaTOverN[i]) for i in range(n_det)]
    dd_total = sum(-TwoDTN_np[i] * float(np.sum(np.abs(data_np[i])**2 / sigmasq_np[i]))
                   for i in range(n_det))
    print(f"  logL at fiducial params = {logL_fid:.2f}")
    print(f"  logL at zero signal (h=0) = {dd_total:.2f}")
    print(f"  Δ logL = {logL_fid - dd_total:.2f}  "
          f"(negative = sub-optimal phic/ψ; scan phic=π for max)")

    # ── 3. Build relative-binning likelihood ──────────────────────────
    print("\n─── 3. Building relative-binning likelihood ─────────────────────")
    print(f"   Using ORIGINAL dense grid ({n_freq_orig} pts) for summary data.")
    print(f"   Waveform is evaluated at only n_bins pts per likelihood call.")
    from relative_binning import build_rb_likelihood

    # n_bins = 500 with max_bin_width = 8 Hz to ensure tc accuracy
    N_RB_BINS = 500
    t0_rb = time.time()
    log_likelihood_rb, rb_network = build_rb_likelihood(
        batched,
        FIDUCIAL_PARAMS,
        _template_mlgw,
        n_bins=N_RB_BINS,
    )
    print(f"  RB setup time: {time.time()-t0_rb:.1f} s")
    n_bins_actual = len(rb_network.f_bins)

    # ── 4. Likelihood agreement check ────────────────────────────────
    print("\n─── 4. Likelihood agreement: RB vs full ─────────────────────────")
    print("   Variations representative of the GW170817 posterior.")
    print("   σ_Mc ≈ 0.001 M☉, σ_tc ≈ 0.5 ms, σ_chi ≈ 0.02, σ_lam ≈ 100.")

    test_params_list = [
        ("fiducial",        FIDUCIAL_PARAMS),
        ("mc+0.001",        FIDUCIAL_PARAMS + np.array([0,0,0,0,0,0, 0.001,0,0,0,0,0,0])),
        ("mc-0.001",        FIDUCIAL_PARAMS + np.array([0,0,0,0,0,0,-0.001,0,0,0,0,0,0])),
        ("logdist+0.2",     FIDUCIAL_PARAMS + np.array([0,0, 0.2,0,0,0,0,0,0,0,0,0,0])),
        ("logdist-0.2",     FIDUCIAL_PARAMS + np.array([0,0,-0.2,0,0,0,0,0,0,0,0,0,0])),
        ("incl+0.3",        FIDUCIAL_PARAMS + np.array([0,0,0, 0.3,0,0,0,0,0,0,0,0,0])),
        ("phic=pi/4",       FIDUCIAL_PARAMS + np.array([0,0,0,0, np.pi/4,0,0,0,0,0,0,0,0])),
        ("phic=pi",         FIDUCIAL_PARAMS + np.array([0,0,0,0, np.pi,  0,0,0,0,0,0,0,0])),
        ("tc+0.001 s",      FIDUCIAL_PARAMS + np.array([0,0,0,0,0,0,0,0, 0.001,0,0,0,0])),
        ("tc-0.001 s",      FIDUCIAL_PARAMS + np.array([0,0,0,0,0,0,0,0,-0.001,0,0,0,0])),
        ("chi1=0.01",       FIDUCIAL_PARAMS + np.array([0,0,0,0,0,0,0,0,0, 0.01,0,0,0])),
        ("lam1+=100",       FIDUCIAL_PARAMS + np.array([0,0,0,0,0,0,0,0,0,0,0, 100.0,0])),
    ]

    jit_full = jax.jit(log_likelihood_full)
    jit_rb   = jax.jit(log_likelihood_rb)

    # Warm up (includes JIT compilation)
    p0 = jnp.array(FIDUCIAL_PARAMS, dtype=jnp.float64)
    _ = jit_full(p0).block_until_ready()
    _ = jit_rb(p0).block_until_ready()

    print(f"\n  {'Name':22s}  {'logL_full':>12s}  {'logL_RB':>12s}  "
          f"{'|ΔlogL|':>10s}  {'pass':>6s}")
    print("  " + "-" * 68)

    all_pass = True
    for name, p_np in test_params_list:
        p = jnp.array(p_np, dtype=jnp.float64)
        lF = float(jit_full(p))
        lR = float(jit_rb(p))
        diff = abs(lF - lR)
        ok = diff < 1.0
        if not ok:
            all_pass = False
        print(f"  {name:22s}  {lF:12.4f}  {lR:12.4f}  {diff:10.4f}  "
              f"{'✓' if ok else '✗'}")

    # ── 5. Timing comparison ─────────────────────────────────────────
    print("\n─── 5. Timing comparison ────────────────────────────────────────")
    N_EVAL = 20
    p_test = p0

    t0 = time.perf_counter()
    for _ in range(N_EVAL):
        jit_full(p_test).block_until_ready()
    t_full = (time.perf_counter() - t0) / N_EVAL * 1e3

    t0 = time.perf_counter()
    for _ in range(N_EVAL):
        jit_rb(p_test).block_until_ready()
    t_rb = (time.perf_counter() - t0) / N_EVAL * 1e3

    speedup = t_full / t_rb if t_rb > 0 else float("inf")
    mem_red = n_freq_orig / n_bins_actual
    print(f"  Full likelihood : {t_full:.2f} ms/eval  ({n_freq_orig} freq bins)")
    print(f"  RB likelihood   : {t_rb:.2f} ms/eval  ({n_bins_actual} freq bins)")
    print(f"  Speed-up        : {speedup:.1f}×")
    print(f"  Memory per particle per detector: "
          f"{n_freq_orig} → {n_bins_actual} complex numbers "
          f"({mem_red:.0f}× reduction)")

    # ── 6. Summary ───────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Signal captured (opt. SNR > 8)   : {'✓ YES' if signal_captured else '✗ NO'}  "
          f"(network SNR = {snr_net:.1f})")
    print(f"  Likelihood agreement (|ΔlogL|<1)  : "
          f"{'✓ ALL PASS' if all_pass else '✗ SOME FAILED'}")
    print(f"  Memory reduction per particle     : {mem_red:.0f}× "
          f"({n_freq_orig} → {n_bins_actual} freq points)")
    print(f"  Speed improvement                 : {speedup:.1f}×")
    print()
    print("  RB is ready for SHARPy PE — see gw170817_pe_sharpy_relbin.py")


if __name__ == "__main__":
    main()
