#!/usr/bin/env python3
"""
Build ROQ_basis_2 for the mlgw_bns_jax waveform model.

Parameters reproduce the basis described in the paper:

  "A pre-selected basis is constructed using Npre = 200 (10) elements
   for the linear (quadratic) case, and Nstep = 1000 points at each
   step.  We set three enrichment cycles each composed of
   [1e4, 1e5, 1e5] datapoints, Ni_out = 0 and a respective relative
   tolerance of [0.1, 1.0, 1.0].  The resulting bases are composed of
   267 (10) linear (quadratic) elements, achieving a linear (quadratic)
   frequency axis reduction factor of 950 (25300)."

All configuration lives in config_roq_basis_2.ini.  The results are
saved under ./ROQ_basis_2/ROQ_data/{linear,quadratic}/.

Resume support is ON by default: if a previous run was interrupted the
build picks up from the last completed phase (pre-selection, enrichment,
or EIM) without repeating work.

Usage
-----
    python build_roq_basis_2.py

Requirements
------------
    pip install jax jaxlib numpy h5py
    The mlgw_bns_jax_model.h5 file must be present in the working directory.
"""

from __future__ import annotations

import gc
import os
import sys
import time

# ── CPU + float64 must be set before any JAX import ─────────────────
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
jax.config.update("jax_enable_x64", True)

# ── Make sure the local modules are importable ───────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from roq_builder_jax import ROQConfig, build_roq_basis, warmup_jit, _load_predictor

# ── Configuration ───────────────────────────────────────────────────
CONFIG_FILE = os.path.join(SCRIPT_DIR, "config_roq_basis_2.ini")
OUT_DIR = os.path.join(SCRIPT_DIR, "ROQ_basis_2")

cfg = ROQConfig.from_ini(CONFIG_FILE)
cfg.output_dir = OUT_DIR
os.makedirs(OUT_DIR, exist_ok=True)

print(f"\n{'='*65}")
print("Building ROQ_basis_2 for mlgw_bns_jax")
print(f"  Config           : {CONFIG_FILE}")
print(f"  Frequency range  : [{cfg.f_min}, {cfg.f_max}] Hz")
print(f"  Segment length   : {cfg.seglen} s  (df = {cfg.delta_f:.4f} Hz)")
print(f"  Frequency points : {cfg.n_freq}")
print(f"  Pre-basis (lin)  : {cfg.n_pre_basis_lin}  (Npre)")
print(f"  Pre-basis (qua)  : {cfg.n_pre_basis_qua}  (Npre)")
print(f"  Search iter/step : {cfg.n_pre_basis_search_iter}  (Nstep)")
print(f"  Enrichment cycles: {len(cfg.training_set_sizes)}")
for i, (n, rt) in enumerate(zip(cfg.training_set_sizes,
                                cfg.training_set_rel_tol), 1):
    print(f"    Cycle {i}: {n:>7,} waveforms, rel_tol={rt:.1f}")
print(f"  Tolerance (lin)  : {cfg.tolerance_lin:.1e}")
print(f"  Tolerance (qua)  : {cfg.tolerance_qua:.1e}")
print(f"  Output directory : {OUT_DIR}")
print(f"  Resume mode      : ON")
print(f"{'='*65}\n")

t_global = time.time()

# ── Phase 1: LINEAR basis ────────────────────────────────────────────
print("=" * 65)
print("  Building LINEAR ROQ basis")
print("=" * 65)

results_lin = build_roq_basis(cfg, kind="linear", resume=True)
n_lin = len(results_lin["nodes"])
print(f"\n✓ Linear basis: {n_lin} empirical nodes  "
      f"({cfg.n_freq / n_lin:.0f}x compression)")

# Free large arrays before building the quadratic basis
del results_lin["basis"]
gc.collect()

# ── Phase 2: QUADRATIC basis ─────────────────────────────────────────
print("\n" + "=" * 65)
print("  Building QUADRATIC ROQ basis")
print("=" * 65)

results_qua = build_roq_basis(cfg, kind="quadratic", resume=True)
n_qua = len(results_qua["nodes"])
print(f"\n✓ Quadratic basis: {n_qua} empirical nodes  "
      f"({cfg.n_freq / n_qua:.0f}x compression)")

# ── Summary ─────────────────────────────────────────────────────────
t_total = time.time() - t_global

print(f"\n{'='*65}")
print("ROQ_basis_2 construction complete!")
print(f"  Full frequency grid : {cfg.n_freq} points")
print(f"  Linear basis        : {n_lin} nodes  "
      f"({cfg.n_freq / n_lin:.0f}x reduction)")
print(f"  Quadratic basis     : {n_qua} nodes  "
      f"({cfg.n_freq / n_qua:.0f}x reduction)")
print(f"  Total wall time     : {t_total:.1f}s  ({t_total / 3600:.2f}h)")
print(f"  Results saved in    : {OUT_DIR}/ROQ_data/")
print(f"{'='*65}")
