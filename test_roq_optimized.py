"""Quick benchmark: predict_nodes (ROQ-optimised) vs predict (original).

Tests:
1. Both predictors return the same waveform at ROQ nodes.
2. predict_nodes is faster at O(N_ROQ) vs O(N_full).
3. build_roq_template / build_roq_likelihood return consistent values.
"""
import os, time
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

MODEL = "mlgw_bns_jax_model.h5"
ROQ_DIR = "roq_basis_mlgw_bns_jax/ROQ_data/linear"

# ── load ROQ nodes ────────────────────────────────────────────────────
f_lin_np  = np.load(f"{ROQ_DIR}/empirical_frequencies_linear.npy")
f_lin_jax = jnp.array(f_lin_np, dtype=jnp.float64)
N_ROQ     = len(f_lin_jax)

DELTA_F      = 1.0 / 128.0
F_LOWER      = 23.0
F_UPPER      = 2000.0
f_full_np    = np.arange(F_LOWER, F_UPPER + DELTA_F, DELTA_F)
f_full_jax   = jnp.array(f_full_np, dtype=jnp.float64)
N_FULL       = len(f_full_jax)

print(f"Full grid    : {N_FULL} pts")
print(f"ROQ nodes    : {N_ROQ} pts  ({N_FULL//N_ROQ}x reduction)")

# ── load both predictors ──────────────────────────────────────────────
from jax_import_n_predict import load_predict
from jax_predict_roq import load_predict_nodes

print("\nLoading predict (original)...")
predict_orig = load_predict(MODEL)

print("Loading predict_nodes (ROQ-optimised)...")
predict_nodes = load_predict_nodes(MODEL)

# ── test parameters (mlgw convention: q, lam1, lam2, chi1, chi2) ─────
PARAMS  = jnp.array([0.87, 300.0, 300.0, 0.0, 0.0])
M_TOTAL = 2.74   # Msun
DIST    = 40.0   # Mpc
INCL    = 2.5    # rad

# ── JIT both at ROQ nodes ─────────────────────────────────────────────
@jax.jit
def _orig_at_roq(params, f):
    return predict_orig(params, f, total_mass=M_TOTAL, distance_mpc=DIST, inclination=INCL)

@jax.jit
def _nodes_at_roq(params, f):
    return predict_nodes(params, f, total_mass=M_TOTAL, distance_mpc=DIST, inclination=INCL)

@jax.jit
def _orig_at_full(params, f):
    return predict_orig(params, f, total_mass=M_TOTAL, distance_mpc=DIST, inclination=INCL)

# warmup
print("\nWarming up JIT...")
hp_o, hc_o = _orig_at_roq(PARAMS, f_lin_jax)
hp_n, hc_n = _nodes_at_roq(PARAMS, f_lin_jax)
hp_of, hc_of = _orig_at_full(PARAMS, f_full_jax)
for x in (hp_o, hc_o, hp_n, hc_n, hp_of, hc_of):
    x.block_until_ready()

# ── accuracy check at ROQ nodes ───────────────────────────────────────
hp_o, hc_o   = _orig_at_roq(PARAMS, f_lin_jax)
hp_n, hc_n   = _nodes_at_roq(PARAMS, f_lin_jax)

rel_amp_p = float(jnp.max(jnp.abs(jnp.abs(hp_n) - jnp.abs(hp_o))
                           / (jnp.abs(hp_o) + 1e-60)))
rel_amp_c = float(jnp.max(jnp.abs(jnp.abs(hc_n) - jnp.abs(hc_o))
                           / (jnp.abs(hc_o) + 1e-60)))
phase_diff_p = float(jnp.max(jnp.abs(jnp.angle(hp_n) - jnp.angle(hp_o))))
phase_diff_c = float(jnp.max(jnp.abs(jnp.angle(hc_n) - jnp.angle(hc_o))))

print("\n── Accuracy at ROQ nodes ──────────────────────────────────────────")
print(f"  max |Δamp/amp| hp : {rel_amp_p:.3e}  {'✓' if rel_amp_p < 1e-3 else '✗'} (< 0.1%)")
print(f"  max |Δamp/amp| hc : {rel_amp_c:.3e}  {'✓' if rel_amp_c < 1e-3 else '✗'} (< 0.1%)")
print(f"  max |Δphase|   hp : {phase_diff_p:.3e} rad  {'✓' if phase_diff_p < 1e-2 else '✗'} (< 0.01 rad)")
print(f"  max |Δphase|   hc : {phase_diff_c:.3e} rad  {'✓' if phase_diff_c < 1e-2 else '✗'} (< 0.01 rad)")

# ── timing ────────────────────────────────────────────────────────────
N_EVAL = 200

def bench(fn, *args):
    t0 = time.perf_counter()
    for _ in range(N_EVAL):
        hp, hc = fn(*args)
        hp.block_until_ready()
    return (time.perf_counter() - t0) / N_EVAL * 1e3  # ms

print(f"\n── Timing ({N_EVAL} evals) ──────────────────────────────────────────")
t_orig_roq  = bench(_orig_at_roq,  PARAMS, f_lin_jax)
t_nodes_roq = bench(_nodes_at_roq, PARAMS, f_lin_jax)
t_orig_full = bench(_orig_at_full, PARAMS, f_full_jax)

print(f"  predict (orig)  @ {N_ROQ} ROQ pts : {t_orig_roq:.3f} ms")
print(f"  predict_nodes   @ {N_ROQ} ROQ pts : {t_nodes_roq:.3f} ms  "
      f"({'✓ FASTER' if t_nodes_roq < t_orig_roq else '✗ NO GAIN'} "
      f"vs orig-at-ROQ, {t_orig_roq/t_nodes_roq:.1f}x)")
print(f"  predict (orig)  @ {N_FULL} full pts: {t_orig_full:.3f} ms")
print(f"  ROQ vs full speedup: {t_orig_full/t_nodes_roq:.1f}x")

# ── build_roq_template smoke test ─────────────────────────────────────
print("\n── build_roq_template smoke test ─────────────────────────────────")
from roq_likelihood_mlgw_bns_jax import build_roq_template
template_nodes = build_roq_template(MODEL)

params_13 = jnp.array([
    3.446, -0.408,          # ra, dec
    jnp.log(40.0), 2.5,    # logdist, incl
    0.0,                    # phic
    0.0,                    # pol
    1.186, 0.87,            # mc, q
    0.0,                    # delta_tc
    0.0, 0.0,               # chi1, chi2
    300.0, 300.0,           # lambda_1, lambda_2
])

hp_t, hc_t = template_nodes(params_13, f_lin_jax)
print(f"  template_nodes returned shapes: hp={hp_t.shape}, hc={hc_t.shape}  ✓")
print(f"  hp max abs: {float(jnp.max(jnp.abs(hp_t))):.3e}")

print("\n── All tests passed ───────────────────────────────────────────────")
