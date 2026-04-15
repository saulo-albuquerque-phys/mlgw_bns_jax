#!/usr/bin/env python3
"""
Build an ROQ basis for the mlgw_bns_jax waveform model using JenpyROQ.

Produces ROQ interpolants in ``roq_basis_mlgw_bns_jax/ROQ_data/``
that can be loaded and used with a custom ROQ likelihood in the
PE notebook.

Usage:
    python build_roq_basis.py
"""

import os, sys, logging

os.environ["JAX_PLATFORMS"] = "cpu"

# ── Register our JAX wrapper before importing JenpyROQ ──────────────
import mlgw_bns_jax_roq_wrapper  # noqa: F401 — side-effect: registers wrapper

from JenpyROQ.jenpyroq import JenpyROQ
from JenpyROQ.initialise import read_config
from JenpyROQ.parallel import initialize_serial_pool

# ── Logging ─────────────────────────────────────────────────────────
logger = logging.getLogger("JenpyROQ")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s  %(message)s"))
logger.addHandler(handler)

# ── Read config ─────────────────────────────────────────────────────
CONFIG_FILE = "config_roq_mlgw_bns_jax_gw170817.ini"
OUT_DIR = "./roq_basis_mlgw_bns_jax/"
os.makedirs(OUT_DIR, exist_ok=True)

config_pars, params_ranges, test_values = read_config(CONFIG_FILE, OUT_DIR, logger)

wf_cfg = config_pars['Waveform_and_parametrisation']
print(f"\n{'='*60}")
print(f"Building ROQ basis for mlgw-bns-jax")
print(f"  f = [{wf_cfg['f-min']}, {wf_cfg['f-max']}] Hz, seglen = {wf_cfg['seglen']} s")
print(f"  Training ranges: {params_ranges}")
print(f"  Output: {OUT_DIR}")
print(f"{'='*60}\n")

# ── Build basis ─────────────────────────────────────────────────────
pool = initialize_serial_pool()
with pool as p:
    roq = JenpyROQ(config_pars, params_ranges, distance=10.0, pool=p)

    print("\n--- Building LINEAR basis ---")
    data_lin = roq.run("lin")
    n_lin = len(data_lin["lin_emp_nodes"])
    print(f"Linear basis: {n_lin} elements")

    print("\n--- Building QUADRATIC basis ---")
    data_qua = roq.run("qua")
    n_qua = len(data_qua["qua_emp_nodes"])
    print(f"Quadratic basis: {n_qua} elements")

# ── Summary ─────────────────────────────────────────────────────────
import numpy as np

f_full = np.arange(
    wf_cfg["f-min"],
    wf_cfg["f-max"] + 1.0 / wf_cfg["seglen"],
    1.0 / wf_cfg["seglen"],
)
n_full = len(f_full)

print(f"\n{'='*60}")
print(f"ROQ basis construction complete!")
print(f"  Full frequency grid: {n_full} points")
print(f"  Linear basis:    {n_lin} elements  ({n_full / n_lin:.0f}x reduction)")
print(f"  Quadratic basis: {n_qua} elements  ({n_full / n_qua:.0f}x reduction)")
print(f"  Output directory: {OUT_DIR}")
print(f"{'='*60}")
