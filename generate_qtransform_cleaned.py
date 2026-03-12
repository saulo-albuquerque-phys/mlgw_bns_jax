"""
Generate Q-transform spectrograms comparing original (glitched)
vs cleaned (deglitched) GW170817 data for H1, L1, V1.

Produces:
- qtransform_cleaned_H1.png, qtransform_cleaned_L1.png, qtransform_cleaned_V1.png
- qtransform_cleaned_all_detectors.png
- qtransform_L1_comparison.png  (glitched vs deglitched side-by-side)
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from gwpy.timeseries import TimeSeries

# ── Metadata ─────────────────────────────────────────────────────────
DATA_DIR = "gw170817_data"
SAMPLE_RATE = 4096
GPS_START = 1187008867
MERGER_GPS = 1187008882.43

WINDOW_SECONDS = 4.0
T_START_PLOT = MERGER_GPS - WINDOW_SECONDS / 2
T_END_PLOT = MERGER_GPS + WINDOW_SECONDS / 2
F_MIN, F_MAX = 20.0, 800.0
Q_RANGE = (4, 64)

ORIGINAL_FILES = {
    "H1": f"{DATA_DIR}/H-H1_GWOSC_4KHZ_R1-1187008867-32.txt",
    "L1": f"{DATA_DIR}/L-L1_GWOSC_4KHZ_R1-1187008867-32.txt",
    "V1": f"{DATA_DIR}/V-V1_GWOSC_4KHZ_R1-1187008867-32.txt",
}
CLEANED_FILES = {
    "H1": f"{DATA_DIR}/H1_cleaned.txt",
    "L1": f"{DATA_DIR}/L1_cleaned.txt",
    "V1": f"{DATA_DIR}/V1_cleaned.txt",
}

DET_COLORS = {"H1": "Reds", "L1": "Blues", "V1": "Purples"}
DET_LABELS = {"H1": "LIGO Hanford (H1)", "L1": "LIGO Livingston (L1)", "V1": "Virgo (V1)"}


def load_and_qtransform(filepath):
    strain = np.loadtxt(filepath, comments="#")
    ts = TimeSeries(strain, sample_rate=SAMPLE_RATE, t0=GPS_START)
    ts_white = ts.whiten(4, 2)
    ts_crop = ts_white.crop(T_START_PLOT - 1, T_END_PLOT + 1)
    return ts_crop.q_transform(
        frange=(F_MIN, F_MAX), qrange=Q_RANGE,
        outseg=(T_START_PLOT, T_END_PLOT), logf=True,
    )


# ── Compute Q-transforms for cleaned data ────────────────────────────
print("Computing Q-transforms for cleaned data...", flush=True)
qt_cleaned = {}
for det, fpath in CLEANED_FILES.items():
    print(f"  {det}...", flush=True)
    qt_cleaned[det] = load_and_qtransform(fpath)

# ── Individual cleaned spectrograms ──────────────────────────────────
print("Generating cleaned spectrogram plots...", flush=True)
for det, qt in qt_cleaned.items():
    fig, ax = plt.subplots(figsize=(12, 6))
    pcm = ax.pcolormesh(
        qt.times.value - MERGER_GPS, qt.frequencies.value, qt.value.T,
        cmap=DET_COLORS[det], vmin=0, vmax=25,
    )
    ax.set_yscale("log"); ax.set_ylim(F_MIN, F_MAX)
    ax.set_ylabel("Frequency [Hz]", fontsize=14)
    ax.set_xlabel("Time relative to merger [s]", fontsize=14)
    ax.set_title(f"GW170817 — Q-transform (cleaned): {DET_LABELS[det]}", fontsize=14)
    ax.tick_params(labelsize=12)
    cbar = fig.colorbar(pcm, ax=ax); cbar.set_label("Normalized energy", fontsize=12)
    ax.axvline(0, color="white", linestyle="--", linewidth=1, alpha=0.7, label="Merger")
    ax.legend(loc="upper left", fontsize=11)
    fig.tight_layout()
    fig.savefig(f"qtransform_cleaned_{det}.png", dpi=150)
    print(f"  Saved: qtransform_cleaned_{det}.png")
    plt.close(fig)

# ── Combined 3-panel cleaned ─────────────────────────────────────────
fig, axes = plt.subplots(3, 1, figsize=(14, 14), sharex=True)
for ax, (det, qt) in zip(axes, qt_cleaned.items()):
    pcm = ax.pcolormesh(
        qt.times.value - MERGER_GPS, qt.frequencies.value, qt.value.T,
        cmap=DET_COLORS[det], vmin=0, vmax=25,
    )
    ax.set_yscale("log"); ax.set_ylim(F_MIN, F_MAX)
    ax.set_ylabel("Frequency [Hz]", fontsize=13)
    ax.set_title(f"{DET_LABELS[det]} (cleaned)", fontsize=13)
    ax.tick_params(labelsize=11)
    ax.axvline(0, color="white", linestyle="--", linewidth=1, alpha=0.7)
    cbar = fig.colorbar(pcm, ax=ax); cbar.set_label("Normalized energy", fontsize=11)
axes[-1].set_xlabel("Time relative to merger [s]", fontsize=13)
fig.suptitle("GW170817 — Q-transform spectrograms (cleaned data)", fontsize=15, y=0.995)
fig.tight_layout()
fig.savefig("qtransform_cleaned_all_detectors.png", dpi=150)
print("  Saved: qtransform_cleaned_all_detectors.png")
plt.close(fig)

# ── L1 comparison: glitched vs deglitched ─────────────────────────────
print("Computing L1 original (glitched) Q-transform for comparison...", flush=True)
qt_L1_orig = load_and_qtransform(ORIGINAL_FILES["L1"])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 6), sharey=True)

# Glitched
pcm1 = ax1.pcolormesh(
    qt_L1_orig.times.value - MERGER_GPS, qt_L1_orig.frequencies.value, qt_L1_orig.value.T,
    cmap="Blues", vmin=0, vmax=25,
)
ax1.set_yscale("log"); ax1.set_ylim(F_MIN, F_MAX)
ax1.set_ylabel("Frequency [Hz]", fontsize=14)
ax1.set_xlabel("Time relative to merger [s]", fontsize=14)
ax1.set_title("L1 — Original (with glitch)", fontsize=14)
ax1.axvline(0, color="white", linestyle="--", linewidth=1, alpha=0.7, label="Merger")
ax1.legend(loc="upper left", fontsize=11)
ax1.tick_params(labelsize=12)
fig.colorbar(pcm1, ax=ax1).set_label("Normalized energy", fontsize=12)

# Cleaned
pcm2 = ax2.pcolormesh(
    qt_cleaned["L1"].times.value - MERGER_GPS, qt_cleaned["L1"].frequencies.value,
    qt_cleaned["L1"].value.T, cmap="Blues", vmin=0, vmax=25,
)
ax2.set_yscale("log"); ax2.set_ylim(F_MIN, F_MAX)
ax2.set_xlabel("Time relative to merger [s]", fontsize=14)
ax2.set_title("L1 — Cleaned (deglitched)", fontsize=14)
ax2.axvline(0, color="white", linestyle="--", linewidth=1, alpha=0.7, label="Merger")
ax2.legend(loc="upper left", fontsize=11)
ax2.tick_params(labelsize=12)
fig.colorbar(pcm2, ax=ax2).set_label("Normalized energy", fontsize=12)

fig.suptitle("GW170817 — LIGO Livingston (L1): Glitch comparison", fontsize=15)
fig.tight_layout()
fig.savefig("qtransform_L1_comparison.png", dpi=150)
print("  Saved: qtransform_L1_comparison.png")
plt.close(fig)

print("\nDone!")
