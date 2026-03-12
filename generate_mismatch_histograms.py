"""
Generate mismatch histograms:
1. mlgw_bns_jax (autonomous) vs original mlgw_bns model
2. mlgw_bns_jax (autonomous) vs TEOBResumS
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

from scipy import integrate
from scipy.interpolate import interp1d
from scipy.optimize import minimize_scalar
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Load models ──────────────────────────────────────────────────────
from jax_import_n_predict import load_predict
from mlgw_bns import Model, ParametersWithExtrinsic
from mlgw_bns.dataset_generation import (
    TEOBResumSGenerator, WaveformParameters, Dataset
)
from EOBRun_module import EOBRunPy

print("Loading models...")
jax_predict = load_predict("mlgw_bns_jax_model.h5")
original_model = Model.default()
teob_gen = TEOBResumSGenerator(EOBRunPy)

# ── PSD for mismatch computation ─────────────────────────────────────
psd_path = Path(__file__).parent / "mlgw_bns" / "data" / "ET_psd.txt"
psd_data = np.loadtxt(psd_path)

# Frequency range: use the common range of the model
f_min = original_model.dataset.effective_initial_frequency_hz
f_max = original_model.dataset.effective_srate_hz / 2.0
M_ref = original_model.dataset.total_mass  # 2.8 solar masses

psd_mask = (psd_data[:, 0] >= f_min) & (psd_data[:, 0] <= f_max)
psd_freqs = psd_data[:, 0][psd_mask]
psd_vals = psd_data[:, 1][psd_mask]
psd_interp = interp1d(psd_freqs, psd_vals)


# ── Mismatch computation ─────────────────────────────────────────────
def compute_mismatch(h1, h2, freqs, psd, max_delta_t=0.07):
    """Compute mismatch between two complex waveforms, optimizing over time shift."""
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


# ── Parameter sampling ───────────────────────────────────────────────
N_WAVEFORMS = 1000
rng = np.random.default_rng(42)

q_range = (1.0, 2.0)
lambda_range = (50.0, 3000.0)
chi_range = (-0.3, 0.3)

q_samples = rng.uniform(*q_range, N_WAVEFORMS)
l1_samples = rng.uniform(*lambda_range, N_WAVEFORMS)
l2_samples = rng.uniform(*lambda_range, N_WAVEFORMS)
chi1_samples = rng.uniform(*chi_range, N_WAVEFORMS)
chi2_samples = rng.uniform(*chi_range, N_WAVEFORMS)

# Fixed extrinsic parameters
distance_mpc = 100.0
inclination = 0.0
total_mass = M_ref  # 2.8 solar masses

# Use the PSD frequency grid as the evaluation frequencies
freqs_hz = psd_freqs.copy()

# ── Generate waveforms and compute mismatches ────────────────────────

print(f"Computing {N_WAVEFORMS} waveform pairs...")
print("Frequencies: {:.1f} Hz to {:.1f} Hz ({} points)".format(
    freqs_hz[0], freqs_hz[-1], len(freqs_hz)))

# Pre-compile JAX predict
print("JIT-compiling JAX model...")
_params_test = jnp.array([1.0, 300.0, 300.0, 0.0, 0.0])
_freqs_test = jnp.array(freqs_hz)
_ = jax_predict(_params_test, _freqs_test,
                total_mass=jnp.array(total_mass),
                distance_mpc=jnp.array(distance_mpc),
                inclination=jnp.array(inclination))
print("JIT compilation done.")

# 1. JAX autonomous vs Original model
print("\n=== Histogram 1: mlgw_bns_jax vs Original mlgw_bns ===")
mismatches_jax_vs_original = []
for i in range(N_WAVEFORMS):
    q, l1, l2, c1, c2 = q_samples[i], l1_samples[i], l2_samples[i], chi1_samples[i], chi2_samples[i]

    # JAX autonomous
    params_jax = jnp.array([q, l1, l2, c1, c2])
    hp_jax, hc_jax = jax_predict(
        params_jax, jnp.array(freqs_hz),
        total_mass=jnp.array(total_mass),
        distance_mpc=jnp.array(distance_mpc),
        inclination=jnp.array(inclination),
    )
    h_jax = np.array(hp_jax)

    # Original model
    params_orig = ParametersWithExtrinsic(
        mass_ratio=q, lambda_1=l1, lambda_2=l2,
        chi_1=c1, chi_2=c2,
        distance_mpc=distance_mpc, inclination=inclination,
        total_mass=total_mass,
    )
    try:
        hp_orig, hc_orig = original_model.predict(freqs_hz, params_orig)
        h_orig = np.array(hp_orig)

        mm = compute_mismatch(h_jax, h_orig, freqs_hz, psd_vals)
        mismatches_jax_vs_original.append(mm)
    except Exception as e:
        print(f"  Skipping waveform {i}: {e}")
        continue

    if (i + 1) % 50 == 0:
        print(f"  {i+1}/{N_WAVEFORMS} done (median mismatch so far: {np.median(mismatches_jax_vs_original):.2e})")

print(f"Computed {len(mismatches_jax_vs_original)} mismatches (JAX vs Original)")
mismatches_jax_vs_original = np.array(mismatches_jax_vs_original)

# 2. JAX autonomous vs TEOBResumS
print("\n=== Histogram 2: mlgw_bns_jax vs TEOBResumS ===")
mismatches_jax_vs_teob = []
dataset = original_model.dataset

for i in range(N_WAVEFORMS):
    q, l1, l2, c1, c2 = q_samples[i], l1_samples[i], l2_samples[i], chi1_samples[i], chi2_samples[i]

    # JAX autonomous
    params_jax = jnp.array([q, l1, l2, c1, c2])
    hp_jax, hc_jax = jax_predict(
        params_jax, jnp.array(freqs_hz),
        total_mass=jnp.array(total_mass),
        distance_mpc=jnp.array(distance_mpc),
        inclination=jnp.array(inclination),
    )
    h_jax = np.array(hp_jax)

    # TEOBResumS
    wf_params = WaveformParameters(
        mass_ratio=q, lambda_1=l1, lambda_2=l2,
        chi_1=c1, chi_2=c2, dataset=dataset,
    )
    try:
        f_teob, amp_teob, phase_teob = teob_gen.effective_one_body_waveform(
            wf_params, frequencies=freqs_hz * dataset.mass_sum_seconds
        )

        # TEOBResumS returns amplitude and phase in geometric units
        # Reconstruct the complex waveform with same conventions as JAX
        # The EOB waveform is h = A * exp(i * phi) in geometric units
        # We need to convert to physical units for comparison
        h_teob_geo = amp_teob * np.exp(1j * phase_teob)

        # Apply scaling: the JAX model returns hp at given distance
        # TEOBResumS with distance=1 (geometric) needs scaling
        # Scale factor: total_mass^2 * eta / (distance_mpc * AMP_SI_BASE)
        eta = q / (1.0 + q) ** 2
        AMP_SI_BASE = 4.2425873413901263e24
        pre = total_mass**2 / AMP_SI_BASE * eta / distance_mpc

        # Apply inclination factor for plus polarization
        cosi = np.cos(inclination)
        pre_plus = (1.0 + cosi**2) / 2.0

        h_teob_phys = pre_plus * pre * h_teob_geo

        mm = compute_mismatch(h_jax, h_teob_phys, freqs_hz, psd_vals)
        mismatches_jax_vs_teob.append(mm)
    except Exception as e:
        print(f"  Skipping waveform {i}: {e}")
        continue

    if (i + 1) % 50 == 0:
        print(f"  {i+1}/{N_WAVEFORMS} done (median mismatch so far: {np.median(mismatches_jax_vs_teob):.2e})")

print(f"Computed {len(mismatches_jax_vs_teob)} mismatches (JAX vs TEOBResumS)")
mismatches_jax_vs_teob = np.array(mismatches_jax_vs_teob)


# ── Plot histograms ──────────────────────────────────────────────────
print("\nGenerating histogram plots...")

# Histogram 1: JAX vs Original
fig1, ax1 = plt.subplots(figsize=(10, 6))
ax1.hist(np.log10(mismatches_jax_vs_original), bins=40, color="steelblue",
         edgecolor="black", alpha=0.8)
ax1.set_xlabel(r"$\log_{10}(\mathrm{Mismatch})$", fontsize=14)
ax1.set_ylabel("Count", fontsize=14)
ax1.set_title("Mismatch: mlgw_bns_jax (autonomous) vs Original mlgw_bns", fontsize=14)
ax1.axvline(np.log10(np.median(mismatches_jax_vs_original)), color="red",
            linestyle="--", linewidth=2,
            label=f"Median: {np.median(mismatches_jax_vs_original):.2e}")
ax1.legend(fontsize=12)
ax1.tick_params(labelsize=12)
fig1.tight_layout()
fig1.savefig("mismatch_jax_vs_original.png", dpi=150)
print("Saved: mismatch_jax_vs_original.png")

# Histogram 2: JAX vs TEOBResumS
fig2, ax2 = plt.subplots(figsize=(10, 6))
ax2.hist(np.log10(mismatches_jax_vs_teob), bins=40, color="darkorange",
         edgecolor="black", alpha=0.8)
ax2.set_xlabel(r"$\log_{10}(\mathrm{Mismatch})$", fontsize=14)
ax2.set_ylabel("Count", fontsize=14)
ax2.set_title("Mismatch: mlgw_bns_jax (autonomous) vs TEOBResumS", fontsize=14)
ax2.axvline(np.log10(np.median(mismatches_jax_vs_teob)), color="red",
            linestyle="--", linewidth=2,
            label=f"Median: {np.median(mismatches_jax_vs_teob):.2e}")
ax2.legend(fontsize=12)
ax2.tick_params(labelsize=12)
fig2.tight_layout()
fig2.savefig("mismatch_jax_vs_teobresums.png", dpi=150)
print("Saved: mismatch_jax_vs_teobresums.png")

# Print summary statistics
print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"\nJAX vs Original mlgw_bns ({len(mismatches_jax_vs_original)} waveforms):")
print(f"  Median mismatch:  {np.median(mismatches_jax_vs_original):.4e}")
print(f"  Mean mismatch:    {np.mean(mismatches_jax_vs_original):.4e}")
print(f"  Max mismatch:     {np.max(mismatches_jax_vs_original):.4e}")
print(f"  Min mismatch:     {np.min(mismatches_jax_vs_original):.4e}")

print(f"\nJAX vs TEOBResumS ({len(mismatches_jax_vs_teob)} waveforms):")
print(f"  Median mismatch:  {np.median(mismatches_jax_vs_teob):.4e}")
print(f"  Mean mismatch:    {np.mean(mismatches_jax_vs_teob):.4e}")
print(f"  Max mismatch:     {np.max(mismatches_jax_vs_teob):.4e}")
print(f"  Min mismatch:     {np.min(mismatches_jax_vs_teob):.4e}")

plt.close("all")
print("\nDone!")
