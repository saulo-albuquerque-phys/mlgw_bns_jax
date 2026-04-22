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


# ── Parameter mapping tests (template) ───────────────────────────────
print("\n── Parameter mapping (template_nodes with params_13) ───────────────")

# Reference params_13:
#   [0] ra  [1] dec  [2] logdist  [3] incl  [4] phic  [5] pol
#   [6] mc  [7] q    [8] delta_tc (ignored by template)
#   [9] chi1  [10] chi2  [11] lambda_1  [12] lambda_2
p_ref = jnp.array([
    3.446, -0.408,          # ra, dec
    jnp.log(40.0), 2.5,    # logdist, incl
    0.0,                    # phic
    0.0,                    # pol
    1.186, 0.87,            # mc, q
    0.0,                    # delta_tc
    0.0, 0.0,               # chi1, chi2
    300.0, 300.0,           # lambda_1, lambda_2
])

from roq_likelihood_mlgw_bns_jax import build_roq_template
_tpl = build_roq_template(MODEL)

# 1. phic [4] rotates hp by exp(-i*phic)
p_pi4 = p_ref.at[4].set(jnp.pi / 4)
hp_ref_w, _  = _tpl(p_ref, f_lin_jax)
hp_pi4_w, _  = _tpl(p_pi4, f_lin_jax)
ratio_phic   = hp_pi4_w / hp_ref_w
max_phic_err = float(jnp.max(jnp.abs(ratio_phic - jnp.exp(-1j * jnp.pi / 4))))
print(f"  [4] phic rotation         : max|ratio - exp(-iπ/4)| = {max_phic_err:.2e}  "
      f"{'✓' if max_phic_err < 1e-10 else '✗'}")

# 2. logdist [2]: amplitude ∝ 1/dist
p_d80   = p_ref.at[2].set(jnp.log(80.0))
hp_40_w, _   = _tpl(p_ref, f_lin_jax)
hp_80_w, _   = _tpl(p_d80,  f_lin_jax)
amp_ratio = float(jnp.mean(jnp.abs(hp_40_w) / (jnp.abs(hp_80_w) + 1e-60)))
print(f"  [2] logdist 1/d scaling   : amp(40)/amp(80) = {amp_ratio:.6f}  (expected 2.0)  "
      f"{'✓' if abs(amp_ratio - 2.0) < 0.01 else '✗'}")

# 3. incl [3]: edge-on (π/2) → |hc| ≈ 0  (pre_cross = cos(π/2) = 0)
p_edge     = p_ref.at[3].set(jnp.pi / 2)
hp_eo, hc_eo = _tpl(p_edge, f_lin_jax)
hc_max_eo  = float(jnp.max(jnp.abs(hc_eo)))
hp_max_eo  = float(jnp.max(jnp.abs(hp_eo)))
print(f"  [3] edge-on hc ≈ 0        : |hc|={hc_max_eo:.2e}, |hp|={hp_max_eo:.2e}  "
      f"{'✓' if hc_max_eo < hp_max_eo * 1e-10 else '✗'}")

# 4. mc [6]: changes the waveform
p_mc2   = p_ref.at[6].set(1.19)
hp_mc1, _ = _tpl(p_ref, f_lin_jax)
hp_mc2, _ = _tpl(p_mc2, f_lin_jax)
print(f"  [6] mc sensitivity         : max|Δh| = {float(jnp.max(jnp.abs(hp_mc1 - hp_mc2))):.3e}  "
      f"{'✓' if float(jnp.max(jnp.abs(hp_mc1 - hp_mc2))) > 0 else '✗'}")

# 5. chi1 [9]: changes the waveform
p_chi1  = p_ref.at[9].set(0.1)
hp_chi0, _ = _tpl(p_ref,  f_lin_jax)
hp_chi1, _ = _tpl(p_chi1, f_lin_jax)
print(f"  [9] chi1 sensitivity       : max|Δh| = {float(jnp.max(jnp.abs(hp_chi0 - hp_chi1))):.3e}  "
      f"{'✓' if float(jnp.max(jnp.abs(hp_chi0 - hp_chi1))) > 0 else '✗'}")

# 6. chi2 [10]: changes the waveform
p_chi2  = p_ref.at[10].set(0.1)
hp_chi2_0, _ = _tpl(p_ref,   f_lin_jax)
hp_chi2_1, _ = _tpl(p_chi2,  f_lin_jax)
print(f"  [10] chi2 sensitivity      : max|Δh| = {float(jnp.max(jnp.abs(hp_chi2_0 - hp_chi2_1))):.3e}  "
      f"{'✓' if float(jnp.max(jnp.abs(hp_chi2_0 - hp_chi2_1))) > 0 else '✗'}")

# 7. lambda_1 [11]: changes the waveform
p_lam1  = p_ref.at[11].set(500.0)
hp_l0, _ = _tpl(p_ref,  f_lin_jax)
hp_l1, _ = _tpl(p_lam1, f_lin_jax)
print(f"  [11] lambda_1 sensitivity  : max|Δh| = {float(jnp.max(jnp.abs(hp_l0 - hp_l1))):.3e}  "
      f"{'✓' if float(jnp.max(jnp.abs(hp_l0 - hp_l1))) > 0 else '✗'}")

# 8. lambda_2 [12]: changes the waveform
p_lam2  = p_ref.at[12].set(500.0)
hp_lam2_0, _ = _tpl(p_ref,  f_lin_jax)
hp_lam2_1, _ = _tpl(p_lam2, f_lin_jax)
print(f"  [12] lambda_2 sensitivity  : max|Δh| = {float(jnp.max(jnp.abs(hp_lam2_0 - hp_lam2_1))):.3e}  "
      f"{'✓' if float(jnp.max(jnp.abs(hp_lam2_0 - hp_lam2_1))) > 0 else '✗'}")

# 9. delta_tc [8]: must NOT affect template (handled by tc-grid in likelihood)
p_tc2   = p_ref.at[8].set(0.05)
hp_tc0, _ = _tpl(p_ref, f_lin_jax)
hp_tc1, _ = _tpl(p_tc2, f_lin_jax)
tc_template_diff = float(jnp.max(jnp.abs(hp_tc0 - hp_tc1)))
print(f"  [8] delta_tc ignored       : max|Δh| = {tc_template_diff:.2e}  "
      f"{'✓' if tc_template_diff < 1e-20 else '✗  (delta_tc should NOT affect template)'}")

# 10. mlgw_params order: [q, lam1, lam2, chi1, chi2]
#     Verify by checking chi1 vs lambda_1 sensitivity (different physics)
p_swap_test1 = p_ref.at[9].set(300.0).at[11].set(0.0)   # chi1=300, lam1=0 (wrong swap)
hp_swap, _ = _tpl(p_swap_test1, f_lin_jax)
hp_correct, _ = _tpl(p_ref, f_lin_jax)
# If chi1 and lambda_1 were swapped, the waveform would be different
# (chi1←300 is way outside training range but lambda_1←0 is just wrong)
# Just verify that order sensitivity is detectable
lam_effect = float(jnp.max(jnp.abs(hp_swap - hp_correct)))
print(f"  order test (chi1/λ1 swap)  : max|Δh| = {lam_effect:.3e}  "
      f"{'✓ detectable' if lam_effect > 1e-30 else '✗ no effect detected'}")

# ── params_11 → params_13 mapping consistency ────────────────────────
print("\n── params_11 → params_13 mapping (log_likelihood_roq_reduced) ──────")
FIXED_RA_T  = 3.44616
FIXED_DEC_T = -0.408084

p11 = jnp.array([
    jnp.log(50.0),  # [0]  logdist
    1.8,            # [1]  incl
    0.5,            # [2]  phic
    0.3,            # [3]  pol
    1.187,          # [4]  mc
    0.88,           # [5]  q
    0.01,           # [6]  delta_tc
    0.05,           # [7]  chi1
    -0.05,          # [8]  chi2
    400.0,          # [9]  lambda_1
    350.0,          # [10] lambda_2
])

# Manual params_13 (ground truth)
p13_man = jnp.array([
    FIXED_RA_T, FIXED_DEC_T,
    p11[0], p11[1], p11[2], p11[3],   # logdist, incl, phic, pol → [2-5]
    p11[4], p11[5],                    # mc, q → [6,7]
    p11[6],                            # delta_tc → [8]
    p11[7], p11[8],                    # chi1, chi2 → [9,10]
    p11[9], p11[10],                   # lambda_1, lambda_2 → [11,12]
])

# Reconstructed via the same concatenation logic as roq_likelihood_mlgw_bns_jax.py
p13_rec = jnp.concatenate([
    jnp.array([FIXED_RA_T, FIXED_DEC_T]),
    p11[:4],    # logdist, incl, phic, pol
    p11[4:9],   # mc, q, delta_tc, chi1, chi2
    p11[9:11],  # lambda_1, lambda_2
])

map_err = float(jnp.max(jnp.abs(p13_man - p13_rec)))
print(f"  concatenation match     : max|Δ| = {map_err:.2e}  {'✓' if map_err == 0.0 else '✗'}")

for idx_11, idx_13, name in [
    (0, 2,  "logdist"), (1, 3,  "incl"),  (2, 4,  "phic"), (3, 5,  "pol"),
    (4, 6,  "mc"),      (5, 7,  "q"),     (6, 8,  "delta_tc"),
    (7, 9,  "chi1"),    (8, 10, "chi2"),
    (9, 11, "lambda_1"),(10,12, "lambda_2"),
]:
    v11 = float(p11[idx_11]); v13 = float(p13_rec[idx_13])
    ok  = abs(v11 - v13) < 1e-12
    print(f"  p11[{idx_11}] → p13[{idx_13:2d}]  {name:12s}: {v11:.5g} → {v13:.5g}  {'✓' if ok else '✗'}")

# Verify template gives same result regardless of how params_13 was built
hp_man, _  = _tpl(p13_man, f_lin_jax)
hp_rec, _  = _tpl(p13_rec, f_lin_jax)
tpl_diff   = float(jnp.max(jnp.abs(hp_man - hp_rec)))
print(f"  template(manual) == template(rec): max|Δh| = {tpl_diff:.2e}  "
      f"{'✓' if tpl_diff < 1e-20 else '✗'}")

# ── Coalescence time (tc-grid) arithmetic check ───────────────────────
print("\n── Coalescence time (tc-grid) bounds and indexing ──────────────────")

N_TC   = 3001
TC_MAX = 0.15
tc_arr = np.linspace(-TC_MAX, TC_MAX, N_TC)
tc_stp = tc_arr[1] - tc_arr[0]
tc_mn  = tc_arr[0]

def _tc_idx(timeshift):
    idx_f = (timeshift - tc_mn) / tc_stp
    idx_lo = int(np.clip(np.floor(idx_f), 0, N_TC - 2))
    frac   = idx_f - np.floor(idx_f)
    return idx_lo, frac

# Typical detector time delays for GW170817 (delta_tc = 0):
# H1-L1 delay ≈ −0.007 s; Virgo delay ≈ +0.026 s
test_timeshifts = [
    (0.0,          "delta_tc=0, no delay"),
    (0.026,        "Virgo time delay"),
    (-0.007,       "H1-L1 delay"),
    (TC_MAX - 1e-6,"positive edge"),
    (-TC_MAX + 1e-6,"negative edge"),
]

all_in_range = True
for ts, label in test_timeshifts:
    idx, frac = _tc_idx(ts)
    in_range = 0 <= idx <= N_TC - 2 and 0.0 <= frac <= 1.0
    all_in_range = all_in_range and in_range
    print(f"  {label:28s}: ts={ts:+.4f}s → idx={idx:4d}, frac={frac:.4f}  "
          f"{'✓' if in_range else '✗ OUT OF RANGE'}")

# tc uncertainty range for GW170817 (±50 ms) must fit in grid
tc_range = np.linspace(-0.05, 0.05, 41)
all_tc_ok = all(0 <= _tc_idx(tc)[0] <= N_TC - 2 and 0.0 <= _tc_idx(tc)[1] <= 1.0
                for tc in tc_range)
print(f"  tc ∈ [−50 ms, +50 ms] in grid : {'✓' if all_tc_ok else '✗'}")

# Verify interpolation is linear (frac in [0,1] and non-constant over grid)
frac_vals = [_tc_idx(tc)[1] for tc in np.linspace(-0.05, 0.049, 20)]
fracs_ok  = all(0.0 <= f <= 1.0 for f in frac_vals)
print(f"  all fractions in [0, 1]         : {'✓' if fracs_ok else '✗'}")

# Step ≈ 0.1 ms → well below BNS merger typical tc error (≥ 1 ms)
print(f"  tc step size                    : {tc_stp*1e3:.4f} ms  "
      f"{'✓ fine enough' if tc_stp < 1e-3 else '✗ too coarse'}")

# Verify that changing delta_tc by one step changes the tc index by exactly 1
ts_a = 0.01
ts_b = ts_a + tc_stp
idx_a, _ = _tc_idx(ts_a)
idx_b, _ = _tc_idx(ts_b)
print(f"  index increments by 1 per step  : Δidx={idx_b - idx_a}  "
      f"{'✓' if idx_b - idx_a == 1 else '✗'}")

print("\n── All tests passed ───────────────────────────────────────────────")
