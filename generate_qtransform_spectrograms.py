"""
Generate Q-transform spectrograms of GW170817 data
from the three detectors: H1, L1, V1.

Uses gwpy for the Q-transform computation and matplotlib for plotting.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from gwpy.timeseries import TimeSeries

# ── Data files and metadata ──────────────────────────────────────────
DATA_DIR = "gw170817_data"
DETECTORS = {
    "H1": f"{DATA_DIR}/H-H1_GWOSC_4KHZ_R1-1187008867-32.txt",
    "L1": f"{DATA_DIR}/L-L1_GWOSC_4KHZ_R1-1187008867-32.txt",
    "V1": f"{DATA_DIR}/V-V1_GWOSC_4KHZ_R1-1187008867-32.txt",
}

SAMPLE_RATE = 4096  # Hz
GPS_START = 1187008867
DURATION = 32  # seconds

# GW170817 merger time (approximate)
MERGER_GPS = 1187008882.43

# Q-transform display window: center around merger
WINDOW_SECONDS = 4.0  # seconds around merger to display
T_CENTER = MERGER_GPS
T_START_PLOT = T_CENTER - WINDOW_SECONDS / 2
T_END_PLOT = T_CENTER + WINDOW_SECONDS / 2

# Frequency range for Q-transform
F_MIN = 20.0
F_MAX = 800.0

# Q range
Q_RANGE = (4, 64)

# ── Load data and compute Q-transforms ───────────────────────────────
print("Loading detector data and computing Q-transforms...")

qtransforms = {}
for det, filepath in DETECTORS.items():
    print(f"  Processing {det}...", flush=True)

    # Load strain data (skip 3 comment lines)
    strain = np.loadtxt(filepath, comments="#")
    ts = TimeSeries(strain, sample_rate=SAMPLE_RATE, t0=GPS_START)

    # Whiten the data to remove colored noise
    ts_white = ts.whiten(4, 2)

    # Crop to region of interest (with padding for edge effects)
    ts_crop = ts_white.crop(T_START_PLOT - 1, T_END_PLOT + 1)

    # Compute Q-transform
    qt = ts_crop.q_transform(
        frange=(F_MIN, F_MAX),
        qrange=Q_RANGE,
        outseg=(T_START_PLOT, T_END_PLOT),
        logf=True,
    )

    qtransforms[det] = qt
    print(f"    {det} done.", flush=True)

# ── Plot: Individual spectrograms ────────────────────────────────────
print("Generating plots...", flush=True)

det_colors = {"H1": "Reds", "L1": "Blues", "V1": "Purples"}
det_labels = {
    "H1": "LIGO Hanford (H1)",
    "L1": "LIGO Livingston (L1)",
    "V1": "Virgo (V1)",
}

# Individual plots
for det, qt in qtransforms.items():
    fig, ax = plt.subplots(figsize=(12, 6))
    pcm = ax.pcolormesh(
        qt.times.value - MERGER_GPS,
        qt.frequencies.value,
        qt.value.T,
        cmap=det_colors[det],
        vmin=0,
        vmax=25,
    )
    ax.set_yscale("log")
    ax.set_ylabel("Frequency [Hz]", fontsize=14)
    ax.set_xlabel("Time relative to merger [s]", fontsize=14)
    ax.set_title(f"GW170817 — Q-transform: {det_labels[det]}", fontsize=14)
    ax.set_ylim(F_MIN, F_MAX)
    ax.tick_params(labelsize=12)
    cbar = fig.colorbar(pcm, ax=ax)
    cbar.set_label("Normalized energy", fontsize=12)
    ax.axvline(0, color="white", linestyle="--", linewidth=1, alpha=0.7, label="Merger")
    ax.legend(loc="upper left", fontsize=11)
    fig.tight_layout()
    fig.savefig(f"qtransform_{det}.png", dpi=150)
    print(f"  Saved: qtransform_{det}.png")
    plt.close(fig)

# Combined 3-panel plot
fig, axes = plt.subplots(3, 1, figsize=(14, 14), sharex=True)
for ax, (det, qt) in zip(axes, qtransforms.items()):
    pcm = ax.pcolormesh(
        qt.times.value - MERGER_GPS,
        qt.frequencies.value,
        qt.value.T,
        cmap=det_colors[det],
        vmin=0,
        vmax=25,
    )
    ax.set_yscale("log")
    ax.set_ylabel("Frequency [Hz]", fontsize=13)
    ax.set_title(det_labels[det], fontsize=13)
    ax.set_ylim(F_MIN, F_MAX)
    ax.tick_params(labelsize=11)
    ax.axvline(0, color="white", linestyle="--", linewidth=1, alpha=0.7)
    cbar = fig.colorbar(pcm, ax=ax)
    cbar.set_label("Normalized energy", fontsize=11)

axes[-1].set_xlabel("Time relative to merger [s]", fontsize=13)
fig.suptitle("GW170817 — Q-transform spectrograms (H1, L1, V1)", fontsize=15, y=0.995)
fig.tight_layout()
fig.savefig("qtransform_all_detectors.png", dpi=150)
print("  Saved: qtransform_all_detectors.png")
plt.close(fig)

print("\nDone!")
