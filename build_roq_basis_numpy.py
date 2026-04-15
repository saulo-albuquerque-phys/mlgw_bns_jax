#!/usr/bin/env python3
"""
Build an ROQ basis for the mlgw_bns waveform model using JenpyROQ.

This script uses the **original NumPy-based mlgw_bns** model (no JAX),
avoiding LLVM JIT compilation out-of-memory issues that occur with the
JAX wrapper on machines with limited RAM (e.g., Google Colab, IGWN
JupyterHub).

The resulting ROQ interpolants are saved under
``roq_basis_mlgw_bns_jax/ROQ_data/`` and can be loaded for ROQ-accelerated
parameter estimation (see ``roq_pe_mlgw_bns_jax.ipynb``).

Usage
-----
    python build_roq_basis_numpy.py

Prerequisites
-------------
    pip install h5py scikit-learn poetry-core
    pip install "JenpyROQ @ git+https://github.com/GCArullo/JenpyROQ.git"
    pip install -e .          # installs mlgw_bns package

References
----------
    * mlgw_bns paper: https://arxiv.org/abs/2210.15684
    * JenpyROQ:       https://github.com/GCArullo/JenpyROQ
"""

from __future__ import annotations

import gc
import logging
import os
import sys
import time

import numpy as np

# ── Register the NumPy wrapper (no JAX) ─────────────────────────────
import mlgw_bns_roq_wrapper  # noqa: F401 — side-effect: registers wrapper

from JenpyROQ.initialise import read_config
from JenpyROQ.jenpyroq import JenpyROQ
from JenpyROQ.parallel import initialize_serial_pool

# ── Logging ─────────────────────────────────────────────────────────
logger = logging.getLogger("JenpyROQ")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s  %(message)s"))
logger.addHandler(handler)

# ── Configuration ───────────────────────────────────────────────────
CONFIG_FILE = "config_roq_mlgw_bns_jax_gw170817.ini"
OUT_DIR = "./roq_basis_mlgw_bns_jax/"
os.makedirs(OUT_DIR, exist_ok=True)

config_pars, params_ranges, test_values = read_config(CONFIG_FILE, OUT_DIR, logger)

wf_cfg = config_pars["Waveform_and_parametrisation"]
fmin = wf_cfg["f-min"]
fmax = wf_cfg["f-max"]
seglen = wf_cfg["seglen"]

print(f"\n{'=' * 60}")
print("Building ROQ basis for mlgw-bns (NumPy — no JAX)")
print(f"  f = [{fmin}, {fmax}] Hz, seglen = {seglen} s")
print(f"  df = {1.0 / seglen:.4f} Hz")
print(f"  Training ranges: {params_ranges}")
print(f"  Output: {OUT_DIR}")
print(f"{'=' * 60}\n")

# ── Smoke test ──────────────────────────────────────────────────────
from mlgw_bns_roq_wrapper import WfMLGWBNS

wf = WfMLGWBNS("mlgw-bns-jax")
p_test = {
    "m1": 1.365,
    "m2": 1.365,
    "s1z": 0.0,
    "s2z": 0.0,
    "lambda1": 300.0,
    "lambda2": 300.0,
    "iota": 2.5,
    "phiref": 0.6,
}
hp_test, hc_test = wf.generate_waveform(p_test, 1.0 / seglen, fmin, fmax, 10.0)
print(f"Smoke test OK: hp shape={hp_test.shape}, max|hp|={np.max(np.abs(hp_test)):.3e}")
del hp_test, hc_test, wf
gc.collect()

# ── Build ROQ basis ─────────────────────────────────────────────────
t0 = time.time()

pool = initialize_serial_pool()
with pool as p:
    roq = JenpyROQ(config_pars, params_ranges, distance=10.0, pool=p)

    # Phase 1: LINEAR basis
    print("\n--- Building LINEAR basis ---")
    data_lin = roq.run("lin")
    n_lin = len(data_lin["lin_emp_nodes"])
    print(f"Linear basis: {n_lin} elements")

    # Free linear arrays before quadratic build.
    # All results are already saved to disk by JenpyROQ.
    del data_lin
    gc.collect()
    print("(linear data freed from RAM — saved on disk)")

    # Phase 2: QUADRATIC basis
    print("\n--- Building QUADRATIC basis ---")
    data_qua = roq.run("qua")
    n_qua = len(data_qua["qua_emp_nodes"])
    print(f"Quadratic basis: {n_qua} elements")

elapsed = time.time() - t0

# ── Summary ─────────────────────────────────────────────────────────
f_full = np.arange(fmin, fmax + 1.0 / seglen, 1.0 / seglen)
n_full = len(f_full)

print(f"\n{'=' * 60}")
print("ROQ basis construction complete!")
print(f"  Full frequency grid : {n_full} points")
print(f"  Linear basis        : {n_lin} elements  ({n_full / n_lin:.0f}x reduction)")
print(f"  Quadratic basis     : {n_qua} elements  ({n_full / n_qua:.0f}x reduction)")
print(f"  Elapsed time        : {elapsed / 3600:.1f} hours")
print(f"  Output directory    : {OUT_DIR}")
print(f"{'=' * 60}")
