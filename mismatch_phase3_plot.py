"""
Generate mismatch histograms (phase 3): Plot the results.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

m1 = np.load("mismatches_jax_vs_original.npy")
m2 = np.load("mismatches_jax_vs_teob.npy")

# Histogram 1: JAX vs Original
fig1, ax1 = plt.subplots(figsize=(10, 6))
ax1.hist(np.log10(m1), bins=40, color="steelblue", edgecolor="black", alpha=0.8)
ax1.set_xlabel(r"$\log_{10}(\mathrm{Mismatch})$", fontsize=14)
ax1.set_ylabel("Count", fontsize=14)
ax1.set_title("Mismatch: mlgw_bns_jax (autonomous) vs Original mlgw_bns", fontsize=14)
ax1.axvline(np.log10(np.median(m1)), color="red", linestyle="--", linewidth=2,
            label=f"Median: {np.median(m1):.2e}")
ax1.legend(fontsize=12)
ax1.tick_params(labelsize=12)
fig1.tight_layout()
fig1.savefig("mismatch_jax_vs_original.png", dpi=150)
print("Saved: mismatch_jax_vs_original.png")

# Histogram 2: JAX vs TEOBResumS
fig2, ax2 = plt.subplots(figsize=(10, 6))
ax2.hist(np.log10(m2), bins=40, color="darkorange", edgecolor="black", alpha=0.8)
ax2.set_xlabel(r"$\log_{10}(\mathrm{Mismatch})$", fontsize=14)
ax2.set_ylabel("Count", fontsize=14)
ax2.set_title("Mismatch: mlgw_bns_jax (autonomous) vs TEOBResumS", fontsize=14)
ax2.axvline(np.log10(np.median(m2)), color="red", linestyle="--", linewidth=2,
            label=f"Median: {np.median(m2):.2e}")
ax2.legend(fontsize=12)
ax2.tick_params(labelsize=12)
fig2.tight_layout()
fig2.savefig("mismatch_jax_vs_teobresums.png", dpi=150)
print("Saved: mismatch_jax_vs_teobresums.png")

# Summary
print(f"\nJAX vs Original ({len(m1)} waveforms):")
print(f"  Median: {np.median(m1):.4e}, Mean: {np.mean(m1):.4e}, Max: {np.max(m1):.4e}, Min: {np.min(m1):.4e}")
print(f"\nJAX vs TEOBResumS ({len(m2)} waveforms):")
print(f"  Median: {np.median(m2):.4e}, Mean: {np.mean(m2):.4e}, Max: {np.max(m2):.4e}, Min: {np.min(m2):.4e}")

plt.close("all")
print("\nDone!")
