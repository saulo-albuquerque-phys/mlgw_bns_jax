"""
Generate mismatch histograms (phase 1):
mlgw_bns_jax (autonomous) vs original mlgw_bns model
"""
import sys, gc, warnings
warnings.filterwarnings("ignore")

import numpy as np
import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

from scipy import integrate
from scipy.interpolate import interp1d
from scipy.optimize import minimize_scalar
from pathlib import Path

from jax_import_n_predict import load_predict
from mlgw_bns import Model, ParametersWithExtrinsic

N_WAVEFORMS = 1000

# ── Load models ──────────────────────────────────────────────────────
print("Loading models...", flush=True)
jax_predict = load_predict("mlgw_bns_jax_model.h5")
original_model = Model.default()

# ── PSD ──────────────────────────────────────────────────────────────
psd_path = Path(__file__).parent / "mlgw_bns" / "data" / "ET_psd.txt"
psd_data = np.loadtxt(psd_path)
f_min = original_model.dataset.effective_initial_frequency_hz
f_max = original_model.dataset.effective_srate_hz / 2.0
M_ref = original_model.dataset.total_mass
psd_mask = (psd_data[:, 0] >= f_min) & (psd_data[:, 0] <= f_max)
psd_freqs = psd_data[:, 0][psd_mask]
psd_vals = psd_data[:, 1][psd_mask]
freqs_hz = psd_freqs.copy()

# ── Mismatch ─────────────────────────────────────────────────────────
def compute_mismatch(h1, h2, freqs, psd, max_delta_t=0.07):
    def inner_product(a, b):
        return abs(integrate.trapezoid(np.conj(a) * b / psd, x=freqs))
    norm = np.sqrt(inner_product(h1, h1) * inner_product(h2, h2))
    if norm == 0:
        return 1.0
    def to_minimize(t_c):
        offset = np.exp(2j * np.pi * freqs * t_c)
        return -inner_product(h1, h2 * offset)
    res = minimize_scalar(to_minimize, method="brent", bracket=(-max_delta_t, max_delta_t))
    return 1.0 - (-res.fun) / norm

# ── Parameters ───────────────────────────────────────────────────────
rng = np.random.default_rng(42)
q_samples = rng.uniform(1.0, 2.0, N_WAVEFORMS)
l1_samples = rng.uniform(50.0, 3000.0, N_WAVEFORMS)
l2_samples = rng.uniform(50.0, 3000.0, N_WAVEFORMS)
chi1_samples = rng.uniform(-0.3, 0.3, N_WAVEFORMS)
chi2_samples = rng.uniform(-0.3, 0.3, N_WAVEFORMS)
distance_mpc = 100.0
inclination = 0.0
total_mass = M_ref

# JIT warmup
print("JIT warmup...", flush=True)
_ = jax_predict(jnp.array([1.0, 300.0, 300.0, 0.0, 0.0]), jnp.array(freqs_hz),
                total_mass=jnp.array(total_mass), distance_mpc=jnp.array(distance_mpc),
                inclination=jnp.array(inclination))
print("JIT done.", flush=True)

# ── Compute mismatches ───────────────────────────────────────────────
print(f"Computing {N_WAVEFORMS} JAX vs Original mismatches...", flush=True)
mismatches = []
for i in range(N_WAVEFORMS):
    q, l1, l2, c1, c2 = q_samples[i], l1_samples[i], l2_samples[i], chi1_samples[i], chi2_samples[i]
    params_jax = jnp.array([q, l1, l2, c1, c2])
    hp_jax, _ = jax_predict(params_jax, jnp.array(freqs_hz),
                            total_mass=jnp.array(total_mass),
                            distance_mpc=jnp.array(distance_mpc),
                            inclination=jnp.array(inclination))
    h_jax = np.array(hp_jax)

    params_orig = ParametersWithExtrinsic(
        mass_ratio=q, lambda_1=l1, lambda_2=l2, chi_1=c1, chi_2=c2,
        distance_mpc=distance_mpc, inclination=inclination, total_mass=total_mass)
    try:
        hp_orig, _ = original_model.predict(freqs_hz, params_orig)
        mm = compute_mismatch(h_jax, np.array(hp_orig), freqs_hz, psd_vals)
        mismatches.append(mm)
    except Exception as e:
        print(f"  Skip {i}: {e}", flush=True)
        continue

    if (i + 1) % 100 == 0:
        np.save("mismatches_jax_vs_original_partial.npy", np.array(mismatches))
        print(f"  {i+1}/{N_WAVEFORMS} (median: {np.median(mismatches):.2e})", flush=True)

np.save("mismatches_jax_vs_original.npy", np.array(mismatches))
print(f"Done! {len(mismatches)} mismatches saved.", flush=True)
