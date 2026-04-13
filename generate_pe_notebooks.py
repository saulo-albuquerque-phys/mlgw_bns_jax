"""Generate the two PE notebooks: TaylorF2 and mlgw_bns_jax, both using SHARPy on a low-res grid."""
import json, os

def make_cell(cell_type, source, **kwargs):
    cell = {
        "cell_type": cell_type,
        "metadata": {},
        "source": source if isinstance(source, list) else source.split("\n"),
    }
    # Fix: split produces single strings, we need lines with \n
    if isinstance(source, str):
        lines = source.split("\n")
        cell["source"] = [l + "\n" for l in lines[:-1]] + [lines[-1]]
    if cell_type == "code":
        cell["execution_count"] = None
        cell["outputs"] = []
    return cell

def make_notebook(cells):
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10.0"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }

# ═══════════════════════════════════════════════════════════════════════
# SHARED CELLS (used in both notebooks)
# ═══════════════════════════════════════════════════════════════════════

SETUP_MD = """\
## Environment setup (Colab / LIGO JupyterHub / fresh environment)

This cell installs all required packages and clones the repositories.

- **Google Colab**: Installs JAX (CUDA 12), clones `mlgw_bns_jax` and SHARPy, installs dependencies.
- **LIGO JupyterHub**: Ensures `gwpy`, `corner`, and `h5py` are available.
- **Local**: Skip if everything is already installed."""

SETUP_CODE = '''\
import os, subprocess, sys, shutil

COLAB = "google.colab" in sys.modules
LIGO  = os.path.isdir("/cvmfs/oasis.opensciencegrid.org")
REPO_DIR = "/content/mlgw_bns_jax" if COLAB else os.getcwd()

if COLAB:
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "-q", "--upgrade",
        "jax[cuda12]",
        "-f", "https://storage.googleapis.com/jax-releases/jax_cuda_releases.html",
    ])
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "-q",
        "corner", "gwpy", "h5py", "blackjax", "tqdm", "netket",
    ])

    if not os.path.isdir(REPO_DIR):
        subprocess.check_call([
            "git", "clone", "--branch", "jax_mlgw_bns", "--depth", "1",
            "https://github.com/saulo-albuquerque-phys/mlgw_bns_jax.git",
            REPO_DIR,
        ])

    sharpy_repo = os.path.join(REPO_DIR, "_sharpy_repo")
    sharpy_pkg  = os.path.join(sharpy_repo, "sharpy")
    sharpy_link = os.path.join(REPO_DIR, "sharpy")
    if not os.path.isdir(sharpy_repo):
        subprocess.check_call([
            "git", "clone", "--depth", "1",
            "https://github.com/saulo-albuquerque-phys/sharpy.git",
            sharpy_repo,
        ])
    if not os.path.exists(sharpy_link):
        os.symlink(sharpy_pkg, sharpy_link)

    os.chdir(REPO_DIR)
    print(f"Working directory: {os.getcwd()}")

elif LIGO:
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "-q",
        "corner", "gwpy", "h5py",
    ])
    os.chdir(os.path.dirname(os.path.abspath("__file__")))
    print(f"LIGO JupyterHub — Working directory: {os.getcwd()}")

else:
    print("Local environment — skipping setup.")'''

# ── Download BayesWave-cleaned data ──────────────────────────────────
DOWNLOAD_DATA_MD = """\
## Download GWOSC data (BayesWave-cleaned L1)

Downloads 1024 s of 4 kHz strain from GWOSC for H1, L1 and V1.

For **L1**, the scatter-light glitch near the merger is removed using the
official **BayesWave glitch subtraction** from
[DCC LIGO-T1700406-v3](https://dcc.ligo.org/LIGO-T1700406/public) —
the same cleaned data used for the GWTC-1 parameter estimation
(Abbott+ 2019, PRX 9, 011001).

The cleaned GWF covers GPS ≥ 1187008667 (553 s into our 1024 s window).
We splice: raw L1 for the earlier portion + BayesWave-cleaned for
the rest.

**Skip if the cleaned files already exist.**"""

DOWNLOAD_DATA_CODE = '''\
import os, sys, time, shutil
import numpy as np

_GPS_START = 1187008114
_DURATION  = 1024
_SRATE     = 4096
_DATA_DIR  = "gw170817_data"
os.makedirs(_DATA_DIR, exist_ok=True)

_DETECTORS = ["H1", "L1", "V1"]

# ── BayesWave-subtracted L1 data from DCC LIGO-T1700406-v3 ──────────
_DCC_GWF_URL = (
    "https://dcc.ligo.org/public/0144/T1700406/003/"
    "L-L1_CLEANED_HOFT_C02_T1700406_v3-1187008667-4096.gwf"
)
_DCC_CHANNEL = "L1:DCH-CLEAN_STRAIN_C02_T1700406_v3"
_DCC_GPS0    = 1187008667
_DCC_SRATE   = 16384


def _ensure_gwf_backend():
    """Make sure at least one GWF reader is importable."""
    for mod in ("frameCPP", "lalframe", "framel"):
        try:
            __import__(mod)
            return
        except ImportError:
            pass
    conda = shutil.which("conda") or shutil.which("mamba")
    if conda:
        import subprocess
        for pkg in ("framel", "python-lalframe"):
            print(f"  Trying: {conda} install -c conda-forge {pkg}", flush=True)
            ret = subprocess.call([conda, "install", "-c", "conda-forge", "-y", "-q", pkg])
            if ret == 0:
                return
    import subprocess
    for pkg in ("framel",):
        ret = subprocess.call([sys.executable, "-m", "pip", "install", "-q", pkg])
        if ret == 0:
            return
    raise ImportError(
        "Cannot read GWF files. Install a backend manually:\\n"
        "  conda install -c conda-forge framel\\n"
    )


def _read_gwf_channel(path, channel, start, end):
    """Read a single channel from a GWF file (try gwpy then framel)."""
    try:
        from gwpy.timeseries import TimeSeries
        ts = TimeSeries.read(path, channel, start=start, end=end)
        return np.asarray(ts.value, dtype=np.float64), float(ts.sample_rate.value)
    except Exception:
        pass
    import framel
    vec = framel.frgetvect1d(path, channel, start, end - start, 0)
    data = np.asarray(vec[0], dtype=np.float64)
    sr   = 1.0 / vec[3]
    return data, sr


_all_exist = all(
    os.path.isfile(os.path.join(_DATA_DIR,
        f"{d[0]}-{d}_BWCLEANED_4KHZ-{_GPS_START}-{_DURATION}.txt"))
    for d in _DETECTORS
)

if _all_exist:
    print("Cleaned data files already exist — skipping download.")
else:
    from gwpy.timeseries import TimeSeries
    from scipy.signal import decimate as _decimate

    for det in _DETECTORS:
        out_file = os.path.join(_DATA_DIR,
            f"{det[0]}-{det}_BWCLEANED_4KHZ-{_GPS_START}-{_DURATION}.txt")

        if det == "L1":
            print("L1: building cleaned timeseries")
            t0 = time.time()

            print("  Downloading raw L1 from GWOSC...", flush=True)
            ts_raw = TimeSeries.fetch_open_data(
                "L1", _GPS_START, _GPS_START + _DURATION, sample_rate=_SRATE)

            gwf_local = os.path.join(_DATA_DIR, "L1_cleaned_bw_T1700406.gwf")
            if not os.path.isfile(gwf_local):
                import requests
                print("  Downloading BayesWave GWF from DCC (~1 GB)…", flush=True)
                resp = requests.get(_DCC_GWF_URL, stream=True)
                resp.raise_for_status()
                with open(gwf_local, "wb") as fout:
                    for chunk in resp.iter_content(chunk_size=1 << 20):
                        fout.write(chunk)
                print(f"  Saved GWF ({os.path.getsize(gwf_local)/1e6:.0f} MB)")
            else:
                print("  BayesWave GWF already cached.")

            _ensure_gwf_backend()

            _need_end = _GPS_START + _DURATION
            print("  Reading cleaned segment from GWF...", flush=True)
            bw_data, bw_sr = _read_gwf_channel(
                gwf_local, _DCC_CHANNEL, _DCC_GPS0, _need_end)

            if int(round(bw_sr)) != _SRATE:
                factor = int(round(bw_sr)) // _SRATE
                print(f"  Resampling {int(bw_sr)} -> {_SRATE} Hz (factor {factor})")
                bw_data = _decimate(bw_data, factor, ftype="iir", zero_phase=True)

            n_raw = int((_DCC_GPS0 - _GPS_START) * _SRATE)
            strain = np.concatenate([ts_raw.value[:n_raw], bw_data])
            n_expected = _DURATION * _SRATE
            assert len(strain) == n_expected, (
                f"L1 splice length mismatch: {len(strain)} vs {n_expected}")

            with open(out_file, "w") as fw:
                fw.write("# BayesWave-cleaned L1 strain for GW170817\\n")
                fw.write(f"# GPS [{_GPS_START}, {_GPS_START+_DURATION}], "
                         f"splice at GPS {_DCC_GPS0}\\n")
                fw.write("# Before splice: raw GWOSC.  After: DCC T1700406-v3 (BayesWave)\\n")
                fw.write(f"# {_SRATE} samples per second\\n")
                for val in strain:
                    fw.write(f"{val:.16e}\\n")
            print(f"  -> L1 done in {time.time()-t0:.1f}s")

        else:
            print(f"{det}: downloading {_DURATION}s from GWOSC…", flush=True)
            t0 = time.time()
            ts = TimeSeries.fetch_open_data(
                det, _GPS_START, _GPS_START + _DURATION, sample_rate=_SRATE)
            with open(out_file, "w") as fw:
                fw.write(f"# {det}: raw GWOSC strain for GW170817\\n")
                fw.write(f"# {_SRATE} samples per second\\n")
                fw.write(f"# starting GPS {_GPS_START} duration {_DURATION}\\n")
                for val in ts.value:
                    fw.write(f"{val:.16e}\\n")
            print(f"  -> saved in {time.time()-t0:.1f}s")

    print("All detectors ready.")'''

# ── JAX setup ────────────────────────────────────────────────────────
JAX_SETUP_CODE = '''\
from __future__ import annotations

import os, sys, time
from functools import partial

import numpy as np
from scipy.signal import welch as scipy_welch
from scipy.signal.windows import tukey
from scipy.interpolate import interp1d

if "google.colab" not in sys.modules:
    os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

print("JAX devices:", jax.devices())'''

# ── Event parameters ─────────────────────────────────────────────────
EVENT_PARAMS_MD = "## Event parameters and analysis settings"

EVENT_PARAMS_CODE = '''\
TRIGGER_TIME = 1187008882.43
SEGMENT_DURATION = 128.0
SAMPLING_RATE = 4096
F_LOWER = 23.0
F_UPPER = 2000.0
N_FREQ_POINTS = 3000
DATA_START_GPS = 1187008114
DATA_DURATION = 1024

FIXED_RA  = 3.44616     # rad
FIXED_DEC = -0.408084   # rad

DATA_DIR = "gw170817_data"
{outdir_line}
{label_line}
os.makedirs(OUTDIR, exist_ok=True)

print(f"Segment duration: {{SEGMENT_DURATION}}s  ->  original df = {{1/SEGMENT_DURATION:.4f}} Hz")
print(f"Low-res grid: {{N_FREQ_POINTS}} points in [{{F_LOWER}}, {{F_UPPER}}] Hz  ->  df = {{(F_UPPER - F_LOWER) / (N_FREQ_POINTS - 1):.4f}} Hz")
print(f"Fixed sky: RA = {{FIXED_RA:.5f}} rad, Dec = {{FIXED_DEC:.6f}} rad  (NGC 4993)")'''

# ── Build detector network (full-resolution) ────────────────────────
BUILD_NETWORK_MD = """\
## Load cleaned data and build detector network (full resolution)

Build the detector network from the 1024 s GWOSC strain files.
With `SEGMENT_DURATION = 128 s`, we analyse a 128-s chunk ending 1 s after the trigger,
and use the remaining ~896 s for Welch PSD estimation (~7 independent segments).

SHARPy's `GWNetwork` handles data loading, windowing, FFT, and PSD estimation automatically."""

BUILD_NETWORK_CODE = '''\
from sharpy.GW_likelihood import GWNetwork, log_likelihood_det
from sharpy.smc_functions import run_sharpy

data_files = {
    "H1": os.path.join(DATA_DIR, f"H-H1_BWCLEANED_4KHZ-{DATA_START_GPS}-{DATA_DURATION}.txt"),
    "L1": os.path.join(DATA_DIR, f"L-L1_BWCLEANED_4KHZ-{DATA_START_GPS}-{DATA_DURATION}.txt"),
    "V1": os.path.join(DATA_DIR, f"V-V1_BWCLEANED_4KHZ-{DATA_START_GPS}-{DATA_DURATION}.txt"),
}
for det, f in data_files.items():
    assert os.path.isfile(f), f"Missing: {f}"
    print(f"{det}: {os.path.basename(f)}")

detector_settings = {}
for det in ["H1", "L1", "V1"]:
    detector_settings[det] = dict(
        data_file=data_files[det], channel="GWOSC",
        trigger_time=TRIGGER_TIME, duration=SEGMENT_DURATION,
        sampling_rate=SAMPLING_RATE,
        f_lower=F_LOWER, f_upper=F_UPPER,
        psd_file=None, psd_method="welch",
        download_data=False, zero_noise=False,
    )

print(f"\\nBuilding GW network (segment={SEGMENT_DURATION}s)...")
t0 = time.time()
gw_network = GWNetwork(detector_settings, injection_parameters=None)
print(f"Network built in {time.time() - t0:.2f} s")'''

# ── Spectrograms ─────────────────────────────────────────────────────
SPECTROGRAM_MD = """\
## Q-transform spectrograms

Verify the data quality: compare raw vs BayesWave-cleaned L1, and show all three cleaned detectors around the merger time."""

SPECTROGRAM_CODE = '''\
from gwpy.timeseries import TimeSeries
import matplotlib.pyplot as plt

MERGER_GPS = TRIGGER_TIME
WINDOW = 6.0
T_START_PLOT = MERGER_GPS - WINDOW / 2
T_END_PLOT   = MERGER_GPS + WINDOW / 2
F_MIN, F_MAX = 20.0, 800.0
Q_RANGE = (4, 64)

DET_COLORS = {"H1": "Reds", "L1": "Blues", "V1": "Purples"}
DET_LABELS = {"H1": "LIGO Hanford (H1)", "L1": "LIGO Livingston (L1)", "V1": "Virgo (V1)"}

def _qtransform(filepath):
    strain = np.loadtxt(filepath, comments="#")
    ts = TimeSeries(strain, sample_rate=SAMPLING_RATE, t0=DATA_START_GPS)
    ts_w = ts.whiten(4, 2)
    ts_c = ts_w.crop(T_START_PLOT - 1, T_END_PLOT + 1)
    return ts_c.q_transform(frange=(F_MIN, F_MAX), qrange=Q_RANGE,
                            outseg=(T_START_PLOT, T_END_PLOT), logf=True)

# ── L1 raw vs cleaned comparison ─────────────────────────────────────
raw_file = os.path.join(DATA_DIR,
    f"L-L1_GWOSC_4KHZ_R1-{DATA_START_GPS}-{DATA_DURATION}.txt")

if os.path.isfile(raw_file):
    qt_raw = _qtransform(raw_file)
    qt_cln = _qtransform(data_files["L1"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 6), sharey=True)
    for ax, qt, title in [(ax1, qt_raw, "L1 — Raw (with glitch)"),
                           (ax2, qt_cln, "L1 — BayesWave cleaned")]:
        pcm = ax.pcolormesh(qt.times.value - MERGER_GPS, qt.frequencies.value,
                            qt.value.T, cmap="Blues", vmin=0, vmax=25)
        ax.set_yscale("log"); ax.set_ylim(F_MIN, F_MAX)
        ax.set_xlabel("Time relative to merger [s]", fontsize=13)
        ax.set_title(title, fontsize=14)
        ax.axvline(0, color="white", ls="--", lw=1, alpha=0.7, label="Merger")
        ax.legend(loc="upper left"); ax.tick_params(labelsize=11)
        fig.colorbar(pcm, ax=ax).set_label("Normalized energy")
    ax1.set_ylabel("Frequency [Hz]", fontsize=13)
    fig.suptitle("GW170817 — L1 glitch comparison (1024 s data)", fontsize=15)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, f"{LABEL}_L1_comparison.png"), dpi=150)
    plt.show()
else:
    print(f"Raw L1 file not found ({raw_file}) — skipping glitch comparison.")

# ── All detectors cleaned ────────────────────────────────────────────
fig, axes = plt.subplots(3, 1, figsize=(14, 14), sharex=True)
for ax, det in zip(axes, ["H1", "L1", "V1"]):
    qt = _qtransform(data_files[det])
    pcm = ax.pcolormesh(qt.times.value - MERGER_GPS, qt.frequencies.value,
                        qt.value.T, cmap=DET_COLORS[det], vmin=0, vmax=25)
    ax.set_yscale("log"); ax.set_ylim(F_MIN, F_MAX)
    ax.set_ylabel("Frequency [Hz]", fontsize=13)
    ax.set_title(f"{DET_LABELS[det]} (BayesWave cleaned)", fontsize=13)
    ax.tick_params(labelsize=11)
    ax.axvline(0, color="white", ls="--", lw=1, alpha=0.7)
    fig.colorbar(pcm, ax=ax).set_label("Normalized energy")
axes[-1].set_xlabel("Time relative to merger [s]", fontsize=13)
fig.suptitle("GW170817 — Q-transform spectrograms (BayesWave-cleaned 1024 s data)",
             fontsize=15, y=0.995)
fig.tight_layout()
fig.savefig(os.path.join(OUTDIR, f"{LABEL}_qtransform_all.png"), dpi=150)
plt.show()

print("Spectrograms saved.")'''

# ── Resample to low-res grid ────────────────────────────────────────
RESAMPLE_MD = """\
## Resample detector data to low-resolution frequency grid

The original FFT grid from the 128 s segment has df ~ 0.0078 Hz (~253k bins in [23, 2000] Hz).
We now interpolate the data strain d(f) and the PSD Sn(f) onto a **uniform grid of 3000 points**,
reducing the cost of each likelihood evaluation by ~84x.

### Phase rotation (critical for coarse grid)

Before interpolating, we rotate the data by $e^{+2\\pi i f(D-1)}$ to absorb the large coalescence-time offset.
This ensures the residual `d_rot - h` only involves the slowly-varying phase $e^{-2\\pi i f(t_d + \\Delta t_c)}$,
which is safely within the Nyquist limit of the coarse grid (~0.76 s).

**The template must correspondingly use `timeshift = td + delta_tc` only.**
We monkey-patch SHARPy's `project_waveform` to remove the `(T-1)` term from the timeshift,
since that phase has been absorbed into the rotated data."""

RESAMPLE_CODE = '''\
from scipy.interpolate import interp1d as _interp1d
import sharpy.GW_likelihood as _gw_mod
from sharpy.GW_likelihood import antenna_pattern_functions
from sharpy.utils import TimeDelayFromEarthCenter

# Access SHARPy's batched detector arrays
batched_det = gw_network.batched_detector

# Extract original frequency grid (same for all detectors after SHARPy build)
f_orig = np.array(batched_det.Frequency[0])
n_det = len(batched_det.latitude)

f_new = np.linspace(F_LOWER, F_UPPER, N_FREQ_POINTS)
df_new = f_new[1] - f_new[0]

print(f"Original FFT grid : {len(f_orig)} bins, df = {1/SEGMENT_DURATION:.4f} Hz")
print(f"New uniform grid  : {N_FREQ_POINTS} bins, df = {df_new:.4f} Hz  ({len(f_orig)/N_FREQ_POINTS:.0f}x reduction)\\n")

print(f"Phase rotation: absorbing exp(+2pi i f x {SEGMENT_DURATION - 1:.0f}s) "
      f"into data before resampling")
print(f"  Nyquist time (coarse grid): {1/(2*df_new):.2f} s  "
      f"->  max |td+dtc| ~ 0.13 s is safe\\n")

# Resample each detector
new_FrequencySeries_list = []
new_PSD_list = []

for i in range(n_det):
    f_det = np.array(batched_det.Frequency[i])
    sf_det = np.array(batched_det.FrequencySeries[i])
    psd_det = np.array(batched_det.PowerSpectralDensity[i])

    # Phase rotation: absorb (duration - 1) phase into data
    phase_corr = 2.0 * np.pi * f_det * (SEGMENT_DURATION - 1.0)
    sf_rotated = sf_det * np.exp(1j * phase_corr)

    # Interpolate rotated frequency series
    sf_new_real = _interp1d(f_det, sf_rotated.real, kind='cubic',
                            bounds_error=False, fill_value=0.0)(f_new)
    sf_new_imag = _interp1d(f_det, sf_rotated.imag, kind='cubic',
                            bounds_error=False, fill_value=0.0)(f_new)
    sf_new = sf_new_real + 1j * sf_new_imag

    # Interpolate PSD in log-space for positivity
    log_psd = np.log(np.where(psd_det > 0, psd_det, 1e-100))
    psd_new = np.exp(
        _interp1d(f_det, log_psd, kind='cubic',
                  bounds_error=False, fill_value=np.log(1e-100))(f_new)
    )

    new_FrequencySeries_list.append(sf_new)
    new_PSD_list.append(psd_new)

# Overwrite the batched detector arrays in-place with low-res versions
# SHARPy uses Frequency, FrequencySeries, PowerSpectralDensity, sigmasq, TwoDeltaTOverN
batched_det = batched_det.replace(
    Frequency=jnp.stack([jnp.array(f_new, dtype=jnp.float64)] * n_det),
    FrequencySeries=jnp.stack([jnp.array(s, dtype=jnp.complex128) for s in new_FrequencySeries_list]),
    PowerSpectralDensity=jnp.stack([jnp.array(p, dtype=jnp.float64) for p in new_PSD_list]),
    sigmasq=jnp.stack([jnp.array(p, dtype=jnp.float64) for p in new_PSD_list]),
    TwoDeltaTOverN=jnp.stack([jnp.float64(2.0 * df_new)] * n_det),
)

# Update the network object
gw_network.batched_detector = batched_det


# ── CRITICAL: Monkey-patch project_waveform to remove (T-1) from timeshift ──
# The data has been pre-rotated by exp(+2πi f (D-1)), absorbing the (D-1) phase.
# SHARPy's original project_waveform uses:
#     timeshift = timedelay + params[8] + (detector_dictionary.T - 1)
# We must remove the (T-1) term to avoid doubling the phase:
#     timeshift = timedelay + params[8]

def project_waveform_lowgrid(params, detector_dictionary):
    """project_waveform for phase-rotated (low-grid) data.

    Identical to SHARPy's project_waveform, except the timeshift does NOT
    include (T - 1) — that phase has been absorbed into the data during
    the resampling step.
    """
    f = detector_dictionary.Frequency
    h_plus, h_cross = _gw_mod.template(params, f)

    fplus, fcross = antenna_pattern_functions(
        params,
        detector_dictionary.latitude, detector_dictionary.longitude,
        detector_dictionary.gamma, detector_dictionary.zeta,
        detector_dictionary.trigtime,
    )

    ra = params[0]
    dec = params[1]
    tc = detector_dictionary.trigtime + params[8]

    timedelay = TimeDelayFromEarthCenter(
        detector_dictionary.latitude, detector_dictionary.longitude,
        detector_dictionary.elevation, ra, dec, tc,
    )

    # Only td + delta_tc — the (T-1) phase is already in the rotated data
    timeshift = timedelay + params[8]
    shift = 2.0 * np.pi * f * timeshift

    h = (fplus * h_plus + fcross * h_cross) * (jnp.cos(shift) - 1j * jnp.sin(shift))
    return h


_gw_mod.project_waveform = project_waveform_lowgrid
print("Monkey-patched project_waveform for low-grid (removed T-1 phase).")

print(f"\\nNetwork resampled to {N_FREQ_POINTS}-point uniform grid (trigger-frame).")
for i, det_name in enumerate(["H1", "L1", "V1"]):
    psd_vals = new_PSD_list[i]
    print(f"  {det_name}: {N_FREQ_POINTS} freq bins, "
          f"PSD range [{psd_vals.min():.2e}, {psd_vals.max():.2e}]")'''

# ── Corner plot ──────────────────────────────────────────────────────
CORNER_MD = "## Corner plot"

CORNER_CODE_TEMPLATE = '''\
from corner import corner

fig = corner(
    np.array(samples), show_titles=True,
    labels=parameter_names, title_kwargs={{"fontsize": 10}},
)
plot_path = os.path.join(OUTDIR, f"{{LABEL}}_corner.png")
fig.savefig(plot_path, dpi=150)
print(f"Saved to {{plot_path}}")
fig'''


# ═══════════════════════════════════════════════════════════════════════
# TaylorF2 NOTEBOOK
# ═══════════════════════════════════════════════════════════════════════

TF2_HEADER_MD = """\
# GW170817 PE — TaylorF2 — Fixed sky — Low-resolution grid — BayesWave-cleaned data

Parameter estimation of GW170817 with **sky location fixed** to the known EM counterpart (NGC 4993):
$$\\alpha = 3.44616\\;\\mathrm{rad}, \\quad \\delta = -0.408084\\;\\mathrm{rad}$$

- **Waveform**: `TaylorF2` — Post-Newtonian 3.5PN point-particle waveform (no tidal terms), implemented in JAX
- **Sampler**: SHARPy SMC (Sequential Monte Carlo with NUTS mutation kernel)
- **Data**: 1024 s of GWOSC strain for H1, L1, V1 — L1 glitch subtracted via **BayesWave** (DCC T1700406-v3)
- **Segment duration**: 128 s (used for FFT and PSD estimation)
- **Likelihood evaluation**: on a **uniform grid of 3000 points** in $[23, 2000]$ Hz ($\\Delta f \\approx 0.66$ Hz)

The TaylorF2 waveform can be evaluated on **any** frequency grid (closed-form PN expansion). We exploit this by resampling the detector data (strain FFT and PSD) onto a coarse uniform grid of 3000 points, drastically reducing the likelihood evaluation cost.

**Compatibility**: This notebook runs on **Google Colab** and **LIGO JupyterHub**.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/saulo-albuquerque-phys/mlgw_bns_jax/blob/jax_mlgw_bns/pe_taylorf2_sharpy_lowgrid.ipynb)

### Sampled parameters (9 — no tidal terms in TaylorF2 point-particle)

| Index | Parameter | Prior range | Boundary |
|:---:|---|---|---|
| 0 | $\\ln d_L$ (luminosity distance) | $[\\ln 1, \\ln 75]$ | reflective |
| 1 | $\\theta_{JN}$ (inclination) | $[0, \\pi]$ | reflective |
| 2 | $\\phi_c$ (phase) | $[0, 2\\pi]$ | periodic |
| 3 | $\\psi$ (polarisation) | $[0, \\pi]$ | periodic |
| 4 | $\\mathcal{M}_c$ (chirp mass) | $[1.18, 1.21]\\,M_\\odot$ | reflective |
| 5 | $q$ (mass ratio) | $[0.5, 1.0]$ | reflective |
| 6 | $t_c$ (coalescence time) | $[-0.1, 0.1]\\,\\mathrm{s}$ | reflective |
| 7 | $\\chi_1$ (spin 1) | $[-0.5, 0.5]$ | reflective |
| 8 | $\\chi_2$ (spin 2) | $[-0.5, 0.5]$ | reflective |

### Phase rotation for low-resolution grid

Before interpolating to the coarse grid, the data is rotated by $e^{+2\\pi i f(D-1)}$ to absorb the large coalescence-time offset. The template then uses `timeshift = td + delta_tc` only, keeping the residual phase safely within the Nyquist limit of the coarse grid (~0.76 s)."""

TF2_WAVEFORM_MD = """\
## TaylorF2 waveform template (JAX)

Pure JAX implementation of the 3.5PN point-particle TaylorF2 frequency-domain waveform.
This is written as a drop-in replacement for SHARPy's `template()` function,
using the same 13-parameter convention."""

TF2_WAVEFORM_CODE = '''\
import sharpy.GW_likelihood as _gw_mod
from sharpy.utils import McQ2Masses

from astropy import constants as const
M_sun = const.M_sun.value
G = const.G.value
c = const.c.value
pc = const.pc.value


def TaylorF2_template(params, frequency_array):
    """TaylorF2 3.5PN point-particle waveform (JAX).

    Uses the SHARPy 13-parameter convention:
        [0] ra, [1] dec, [2] logdist, [3] incl, [4] phic, [5] pol,
        [6] mc, [7] q, [8] tc, [9] chi1, [10] chi2, [11] lambda_1, [12] lambda_2

    NOTE: Tidal parameters (lambda_1, lambda_2) are IGNORED in this
    point-particle approximant.
    """
    Mc       = params[6]           # chirp mass (M_sun)
    q        = params[7]           # mass ratio m2/m1
    phi_c    = params[4]           # coalescence phase
    logdist  = params[2]           # log(distance / Mpc)
    cos_iota = jnp.cos(params[3]) # cos(inclination)

    distance = jnp.exp(logdist)
    nu = q / ((1 + q) ** 2)       # symmetric mass ratio

    Mc_kg = Mc * M_sun
    r = distance * pc * 1e6       # distance in metres

    M = Mc_kg / (nu ** (3.0 / 5.0))

    pi_M = G * jnp.pi * M
    v = jnp.power(pi_M * frequency_array, 1.0 / 3.0) / c
    gamma_e = jnp.float64(0.5772156649015329)

    # Amplitude
    amp = (jnp.power(jnp.pi, -2.0 / 3.0) * jnp.sqrt(5.0 / 24.0)
           * jnp.power(G * Mc_kg / c**3, 5.0 / 6.0)
           * jnp.power(frequency_array, -7.0 / 6.0)
           * (c / r))

    # Phase: 3.5PN expansion
    v2 = v**2;  v3 = v**3;  v4 = v**4
    v5 = v**5;  v6 = v**6;  v7 = v**7
    log_v = jnp.log(v)

    psi = (3.0 / (128.0 * nu * v5)) * (
        1.0
        + v2 * (20.0 / 9.0) * (743.0 / 336.0 + nu * 11.0 / 4.0)
        - v3 * (16.0 * jnp.pi)
        + v4 * 10.0 * (3058673.0 / 1016064.0 + nu * 5429.0 / 1008.0 + nu**2 * 617.0 / 144.0)
        + v5 * jnp.pi * (38645.0 / 756.0 - nu * 65.0 / 9.0) * (1.0 + 3.0 * log_v)
        + v6 * (11583231236531.0 / 4694215680.0 - jnp.pi**2 * 640.0 / 3.0
                - 6848.0 * gamma_e / 21.0 - 6848.0 / 21.0 * log_v
                + nu * (-15737765635.0 / 3048192.0 + 2255.0 * jnp.pi**2 / 12.0)
                + nu**2 * 76055.0 / 1728.0 - nu**3 * 127825.0 / 1296.0)
        + v7 * jnp.pi * (77096675.0 / 254016.0 + nu * 378515.0 / 1512.0 - nu**2 * 74045.0 / 756.0)
    )

    cos_iota_sq = cos_iota**2
    # Convention: exp(-i*psi) matches IMRPhenomD / SHARPy's project_waveform
    h_plus  = jnp.exp(-1j * phi_c) * amp * ((1.0 + cos_iota_sq) / 2.0) * jnp.exp(-1j * psi)
    h_cross = jnp.exp(-1j * phi_c) * amp * cos_iota * jnp.exp(-1j * (psi + jnp.pi / 2.0))

    return h_plus, h_cross


# ── Monkey-patch SHARPy's template ────────────────────────────────────
_gw_mod.template = TaylorF2_template

print("SHARPy template patched with TaylorF2 (3.5PN point-particle).")'''

TF2_LIKELIHOOD_MD = """\
## Define likelihood and priors (9 parameters — no tidal)

Since TaylorF2 is a point-particle waveform (no tidal deformability), we sample only **9 parameters**
(dropping $\\Lambda_1, \\Lambda_2$). The wrapper inserts the fixed RA/Dec and sets $\\Lambda_{1,2} = 0$
for the 13-parameter vector expected by SHARPy.

**Important (v4 fix)**: The data was pre-rotated by $e^{+2\\pi i f(D-1)}$ during resampling.
The `project_waveform` inside SHARPy still applies the full timeshift including $(D-1)$.
We compensate by using the low-res batched detector which has this phase absorbed."""

TF2_LIKELIHOOD_CODE = '''\
batched_detector = gw_network.batched_detector
log_likelihood_full = partial(log_likelihood_det, detector_list=batched_detector)


def log_likelihood_reduced(params_9):
    """Insert fixed RA/Dec and zero tidal params, evaluate full 13-param likelihood.

    params_9 layout:
        [0] logdist, [1] incl, [2] phic, [3] pol,
        [4] mc, [5] q, [6] tc, [7] chi1, [8] chi2
    """
    params_13 = jnp.concatenate([
        jnp.array([FIXED_RA, FIXED_DEC]),     # [0] ra, [1] dec (fixed)
        params_9,                              # [2..10] logdist..chi2
        jnp.array([0.0, 0.0]),                 # [11] lambda_1, [12] lambda_2 (unused)
    ])
    return log_likelihood_full(params_13)


# Prior bounds for the 9 sampled parameters
prior_bounds = jnp.array([
    [jnp.log(1.0),  jnp.log(75.0)],     # [0] logdistance (1-75 Mpc)
    [0.0,           jnp.pi],             # [1] inclination
    [0.0,           2 * jnp.pi],         # [2] phic
    [0.0,           jnp.pi],             # [3] pol
    [1.18,          1.21],               # [4] mc (chirp mass, M_sun)
    [0.5,           1.0],                # [5] q  (mass ratio)
    [-0.1,          0.1],                # [6] tc (relative to trigger, s)
    [-0.5,          0.5],                # [7] chi1
    [-0.5,          0.5],                # [8] chi2
])

# 1 = periodic, 0 = reflective
boundary_conditions = jnp.array([0, 0, 1, 1, 0, 0, 0, 0, 0])

parameter_names = [
    "logdistance", "theta_jn", "phiref", "pol",
    "mc", "q", "tc", "chi1", "chi2",
]


def prior(params):
    """Uniform prior (log-prior = 0 inside bounds)."""
    return 0.0


# ── Sanity check ──────────────────────────────────────────────────────
test_params = jnp.array([
    jnp.log(40.0),         # logdist
    jnp.pi / 6,            # inclination
    1.0,                   # phic
    0.5,                   # pol
    1.1976,                # mc
    0.9,                   # q
    0.0,                   # tc
    0.0,                   # chi1
    0.0,                   # chi2
])

logL_check = log_likelihood_reduced(test_params)
print(f"logL at approximate GW170817 params = {float(logL_check):.2f}")
print(f"Fixed: RA = {FIXED_RA:.5f}, Dec = {FIXED_DEC:.6f}")
print(f"Sampling {len(parameter_names)} parameters: {parameter_names}")'''

TF2_SAMPLER_MD = """\
## Run the SHARPy SMC sampler

With 9 parameters (no tidal deformability), TaylorF2 PE is relatively lightweight. The
NUTS mutation kernel with Hessian-based mass matrix adaptation ensures efficient exploration."""

TF2_SAMPLER_CODE = '''\
N_PARTICLES = 500
STEP_SIZE = 0.3
ALPHA = 0.95
SEED = 42

print(f"Starting SHARPy SMC with {N_PARTICLES} particles over {len(parameter_names)} parameters...")
start = time.time()

result_dict = run_sharpy(
    log_likelihood_reduced, prior,
    prior_bounds, boundary_conditions,
    ALPHA, N_PARTICLES, STEP_SIZE,
    jax.random.PRNGKey(SEED),
    folder=OUTDIR, label=LABEL,
)

dt_run = time.time() - start
samples = np.array(result_dict["posterior_samples"])
logZ, dlogZ = result_dict["logZ"], result_dict["dlogZ"]
print(f"\\nDone in {dt_run:.1f} s — log Z = {logZ:.2f} +/- {dlogZ:.2f}")
print(f"Posterior samples: {len(samples)}")'''

TF2_CORNER_CODE = '''\
from corner import corner

fig = corner(
    np.array(samples), show_titles=True,
    labels=parameter_names, title_kwargs={"fontsize": 10},
)
plot_path = os.path.join(OUTDIR, f"{LABEL}_corner.png")
fig.savefig(plot_path, dpi=150)
print(f"Saved to {plot_path}")
fig'''

TF2_PAPER_MD = """\
## Paper-style corner plot

Compute derived parameters from the posterior samples and produce a corner plot showing:
- $\\mathcal{M}_c$ — chirp mass
- $q$ — mass ratio
- $\\chi_\\text{eff}$ — effective spin parameter
- $D_L$ — luminosity distance [Mpc]

Column indices for the 9-parameter samples:
`[0] logdist, [1] incl, [2] phic, [3] pol, [4] mc, [5] q, [6] tc, [7] chi1, [8] chi2`"""

TF2_PAPER_CODE = '''\
from sharpy.utils import McQ2Masses

mc_samples   = np.array(samples[:, 4])
q_samples    = np.array(samples[:, 5])
chi1_samples = np.array(samples[:, 7])
chi2_samples = np.array(samples[:, 8])
logd_samples = np.array(samples[:, 0])

m1_samples = np.zeros(len(mc_samples))
m2_samples = np.zeros(len(mc_samples))
for i in range(len(mc_samples)):
    m1_samples[i], m2_samples[i] = McQ2Masses(mc_samples[i], q_samples[i])

chi_eff_samples = (m1_samples * chi1_samples + m2_samples * chi2_samples) / (m1_samples + m2_samples)
dL_samples = np.exp(logd_samples)

paper_samples = np.column_stack([mc_samples, q_samples, chi_eff_samples, dL_samples])

paper_labels = [
    r"$\\mathcal{M}_c$ $[M_\\odot]$",
    r"$q$",
    r"$\\chi_{\\rm eff}$",
    r"$D_L$ [Mpc]",
]

fig_paper = corner(
    paper_samples, show_titles=True,
    labels=paper_labels,
    title_kwargs={"fontsize": 12},
    quantiles=[0.05, 0.5, 0.95],
    levels=(0.5, 0.9),
    fill_contours=True,
    color="tab:blue",
)
plot_path_paper = os.path.join(OUTDIR, f"{LABEL}_corner_paper.png")
fig_paper.savefig(plot_path_paper, dpi=150)
print(f"Saved to {plot_path_paper}")
fig_paper'''


# ═══════════════════════════════════════════════════════════════════════
# MLGW_BNS_JAX NOTEBOOK
# ═══════════════════════════════════════════════════════════════════════

MLGW_HEADER_MD = """\
# GW170817 PE — mlgw_bns_jax — Fixed sky — Low-resolution grid — BayesWave-cleaned data

Parameter estimation of GW170817 with **sky location fixed** to the known EM counterpart (NGC 4993):
$$\\alpha = 3.44616\\;\\mathrm{rad}, \\quad \\delta = -0.408084\\;\\mathrm{rad}$$

- **Waveform**: `mlgw_bns_jax` — JAX-based BNS approximant (neural-network surrogate of `mlgw_bns`)
- **Sampler**: SHARPy SMC (Sequential Monte Carlo with NUTS mutation kernel)
- **Data**: 1024 s of GWOSC strain for H1, L1, V1 — L1 glitch subtracted via **BayesWave** (DCC T1700406-v3)
- **Segment duration**: 128 s (used for FFT and PSD estimation)
- **Likelihood evaluation**: on a **uniform grid of 3000 points** in $[23, 2000]$ Hz ($\\Delta f \\approx 0.66$ Hz)

The key advantage of `mlgw_bns_jax` over FFT-based waveforms is that it can be evaluated on **any** frequency grid. We exploit this by resampling the detector data (strain FFT and PSD) onto a coarse uniform grid of 3000 points, drastically reducing the number of likelihood evaluations per sample (from ~253k to 3k frequency bins).

**Compatibility**: This notebook runs on **Google Colab** and **LIGO JupyterHub**.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/saulo-albuquerque-phys/mlgw_bns_jax/blob/jax_mlgw_bns/pe_mlgw_bns_jax_sharpy_lowgrid.ipynb)

### Sampled parameters (11 — including tidal deformabilities)

| Index | Parameter | Prior range | Boundary |
|:---:|---|---|---|
| 0 | $\\ln d_L$ (luminosity distance) | $[\\ln 1, \\ln 75]$ | reflective |
| 1 | $\\theta_{JN}$ (inclination) | $[0, \\pi]$ | reflective |
| 2 | $\\phi_c$ (phase) | $[0, 2\\pi]$ | periodic |
| 3 | $\\psi$ (polarisation) | $[0, \\pi]$ | periodic |
| 4 | $\\mathcal{M}_c$ (chirp mass) | $[1.18, 1.21]\\,M_\\odot$ | reflective |
| 5 | $q$ (mass ratio) | $[0.5, 1.0]$ | reflective |
| 6 | $t_c$ (coalescence time) | $[-0.1, 0.1]\\,\\mathrm{s}$ | reflective |
| 7 | $\\chi_1$ (spin 1) | $[-0.5, 0.5]$ | reflective |
| 8 | $\\chi_2$ (spin 2) | $[-0.5, 0.5]$ | reflective |
| 9 | $\\Lambda_1$ (tidal 1) | $[5, 5000]$ | reflective |
| 10 | $\\Lambda_2$ (tidal 2) | $[5, 5000]$ | reflective |

### Phase rotation for low-resolution grid

Before interpolating to the coarse grid, the data is rotated by $e^{+2\\pi i f(D-1)}$ to absorb the large coalescence-time offset. The template then uses `timeshift = td + delta_tc` only."""

MLGW_WAVEFORM_MD = """\
## Load the mlgw_bns_jax waveform model and monkey-patch SHARPy

Load the JAX-based BNS waveform surrogate from its HDF5 file and inject it into SHARPy's
`template()` function via monkey-patching. No SHARPy source files are modified."""

MLGW_WAVEFORM_CODE = '''\
from jax_import_n_predict import load_predict

MODEL_PATH = "mlgw_bns_jax_model.h5"
_mlgw_predict = load_predict(MODEL_PATH)

# ── Monkey-patch SHARPy's template ────────────────────────────────────
import sharpy.GW_likelihood as _gw_mod
from sharpy.utils import McQ2Masses


def _template_mlgw_bns(params, frequency_array):
    """mlgw_bns_jax waveform, drop-in replacement for SHARPy's template."""
    mc, q = params[6], params[7]
    m1, m2 = McQ2Masses(mc, q)
    total_mass = m1 + m2
    chi1, chi2 = params[9], params[10]
    lambda_1, lambda_2 = params[11], params[12]
    phic = params[4]
    dist_mpc = jnp.exp(params[2])
    inclination = params[3]

    mlgw_params = jnp.array([q, lambda_1, lambda_2, chi1, chi2])
    hp, hc = _mlgw_predict(
        mlgw_params, frequency_array,
        total_mass=total_mass,
        distance_mpc=dist_mpc,
        inclination=inclination,
    )
    phase_factor = jnp.exp(-1j * phic)
    return hp * phase_factor, hc * phase_factor


_gw_mod.template = _template_mlgw_bns

# Quick test
_test_params = jnp.array([1.0, 300.0, 300.0, 0.0, 0.0])
_test_freqs  = jnp.linspace(23.0, 2000.0, 100)
_hp_test, _ = _mlgw_predict(
    _test_params, _test_freqs,
    total_mass=jnp.array(2.8),
    distance_mpc=jnp.array(40.0),
    inclination=jnp.array(0.3),
)
print(f"Model loaded — test hp shape: {_hp_test.shape}, max|hp|: {float(jnp.max(jnp.abs(_hp_test))):.3e}")
print("SHARPy template patched with mlgw_bns_jax.")'''

MLGW_LIKELIHOOD_MD = """\
## Define likelihood and priors (11 parameters — with tidal)

11 parameters sampled (RA and Dec fixed to NGC 4993), with uniform priors.
The wrapper inserts the fixed RA/Dec into the 13-parameter vector expected by SHARPy."""

MLGW_LIKELIHOOD_CODE = '''\
batched_detector = gw_network.batched_detector
log_likelihood_full = partial(log_likelihood_det, detector_list=batched_detector)


def log_likelihood_reduced(params_11):
    """Insert fixed RA/Dec and evaluate the full 13-param likelihood.

    params_11 layout:
        [0] logdist, [1] incl, [2] phic, [3] pol,
        [4] mc, [5] q, [6] tc, [7] chi1, [8] chi2,
        [9] lambda_1, [10] lambda_2
    """
    params_13 = jnp.concatenate([
        jnp.array([FIXED_RA, FIXED_DEC]),
        params_11,
    ])
    return log_likelihood_full(params_13)


# Prior bounds for the 11 sampled parameters
prior_bounds = jnp.array([
    [jnp.log(1.0),  jnp.log(75.0)],     # [0]  logdistance (1-75 Mpc)
    [0.0,           jnp.pi],             # [1]  inclination
    [0.0,           2 * jnp.pi],         # [2]  phic
    [0.0,           jnp.pi],             # [3]  pol
    [1.18,          1.21],               # [4]  mc (chirp mass, M_sun)
    [0.5,           1.0],                # [5]  q  (mass ratio)
    [-0.1,          0.1],                # [6]  tc (relative to trigger, s)
    [-0.5,          0.5],                # [7]  chi1
    [-0.5,          0.5],                # [8]  chi2
    [5.0,           5000.0],             # [9]  lambda_1
    [5.0,           5000.0],             # [10] lambda_2
])

# 1 = periodic, 0 = reflective
boundary_conditions = jnp.array([0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0])

parameter_names = [
    "logdistance", "theta_jn", "phiref", "pol",
    "mc", "q", "tc", "chi1", "chi2", "lambda_1", "lambda_2",
]


def prior(params):
    """Uniform prior (log-prior = 0 inside bounds)."""
    return 0.0


# ── Sanity check ──────────────────────────────────────────────────────
test_params = jnp.array([
    jnp.log(40.0),         # logdist
    jnp.pi / 6,            # inclination
    1.0,                   # phic
    0.5,                   # pol
    1.1976,                # mc
    0.9,                   # q
    0.0,                   # tc
    0.0,                   # chi1
    0.0,                   # chi2
    300.0,                 # lambda_1
    300.0,                 # lambda_2
])

logL_check = log_likelihood_reduced(test_params)
print(f"logL at approximate GW170817 params = {float(logL_check):.2f}")
print(f"Fixed: RA = {FIXED_RA:.5f}, Dec = {FIXED_DEC:.6f}")
print(f"Sampling {len(parameter_names)} parameters: {parameter_names}")'''

MLGW_SAMPLER_MD = """\
## Run the SHARPy SMC sampler

With 11 parameters (RA/Dec fixed), the sampler explores the BNS parameter space including
tidal deformabilities $\\Lambda_{1,2}$."""

MLGW_SAMPLER_CODE = '''\
N_PARTICLES = 500
STEP_SIZE = 0.3
ALPHA = 0.95
SEED = 42

print(f"Starting SHARPy SMC with {N_PARTICLES} particles over {len(parameter_names)} parameters...")
start = time.time()

result_dict = run_sharpy(
    log_likelihood_reduced, prior,
    prior_bounds, boundary_conditions,
    ALPHA, N_PARTICLES, STEP_SIZE,
    jax.random.PRNGKey(SEED),
    folder=OUTDIR, label=LABEL,
)

dt_run = time.time() - start
samples = np.array(result_dict["posterior_samples"])
logZ, dlogZ = result_dict["logZ"], result_dict["dlogZ"]
print(f"\\nDone in {dt_run:.1f} s — log Z = {logZ:.2f} +/- {dlogZ:.2f}")
print(f"Posterior samples: {len(samples)}")'''

MLGW_CORNER_CODE = '''\
from corner import corner

fig = corner(
    np.array(samples), show_titles=True,
    labels=parameter_names, title_kwargs={"fontsize": 10},
)
plot_path = os.path.join(OUTDIR, f"{LABEL}_corner.png")
fig.savefig(plot_path, dpi=150)
print(f"Saved to {plot_path}")
fig'''

MLGW_PAPER_MD = """\
## Paper-style corner plot (Figure 9)

Compute derived parameters from the posterior samples and produce a corner plot showing:
- $\\mathcal{M}_c$ — chirp mass
- $q$ — mass ratio
- $\\chi_\\text{eff}$ — effective spin parameter
- $\\tilde{\\Lambda}$ — reduced tidal deformability
- $D_L$ — luminosity distance [Mpc]

Column indices for the 11-parameter samples:
`[0] logdist, [1] incl, [2] phic, [3] pol, [4] mc, [5] q, [6] tc, [7] chi1, [8] chi2, [9] lambda_1, [10] lambda_2`"""

MLGW_PAPER_CODE = '''\
from sharpy.utils import McQ2Masses

mc_samples   = np.array(samples[:, 4])
q_samples    = np.array(samples[:, 5])
chi1_samples = np.array(samples[:, 7])
chi2_samples = np.array(samples[:, 8])
lam1_samples = np.array(samples[:, 9])
lam2_samples = np.array(samples[:, 10])
logd_samples = np.array(samples[:, 0])

m1_samples = np.zeros(len(mc_samples))
m2_samples = np.zeros(len(mc_samples))
for i in range(len(mc_samples)):
    m1_samples[i], m2_samples[i] = McQ2Masses(mc_samples[i], q_samples[i])

chi_eff_samples = (m1_samples * chi1_samples + m2_samples * chi2_samples) / (m1_samples + m2_samples)

M_samples = m1_samples + m2_samples
lambda_tilde_samples = (16.0 / 13.0) * (
    (m1_samples + 12.0 * m2_samples) * m1_samples**4 * lam1_samples
    + (m2_samples + 12.0 * m1_samples) * m2_samples**4 * lam2_samples
) / M_samples**5

dL_samples = np.exp(logd_samples)

paper_samples = np.column_stack([
    mc_samples, q_samples, chi_eff_samples, lambda_tilde_samples, dL_samples,
])

paper_labels = [
    r"$\\mathcal{M}_c$ $[M_\\odot]$",
    r"$q$",
    r"$\\chi_{\\rm eff}$",
    r"$\\tilde{\\Lambda}$",
    r"$D_L$ [Mpc]",
]

fig_paper = corner(
    paper_samples, show_titles=True,
    labels=paper_labels,
    title_kwargs={"fontsize": 12},
    quantiles=[0.05, 0.5, 0.95],
    levels=(0.5, 0.9),
    fill_contours=True,
    color="tab:orange",
)
plot_path_paper = os.path.join(OUTDIR, f"{LABEL}_corner_paper.png")
fig_paper.savefig(plot_path_paper, dpi=150)
print(f"Saved to {plot_path_paper}")
fig_paper'''


# ═══════════════════════════════════════════════════════════════════════
# ASSEMBLE NOTEBOOKS
# ═══════════════════════════════════════════════════════════════════════

def build_taylorf2_notebook():
    cells = [
        make_cell("markdown", TF2_HEADER_MD),
        make_cell("markdown", SETUP_MD),
        make_cell("code", SETUP_CODE),
        make_cell("markdown", DOWNLOAD_DATA_MD),
        make_cell("code", DOWNLOAD_DATA_CODE),
        make_cell("code", JAX_SETUP_CODE),
        make_cell("markdown", TF2_WAVEFORM_MD),
        make_cell("code", TF2_WAVEFORM_CODE),
        make_cell("markdown", EVENT_PARAMS_MD),
        make_cell("code", EVENT_PARAMS_CODE.format(
            outdir_line='OUTDIR = "results_taylorf2_sharpy_lowgrid"',
            label_line='LABEL = "GW170817_TaylorF2_sharpy_lowgrid"',
        )),
        make_cell("markdown", BUILD_NETWORK_MD),
        make_cell("code", BUILD_NETWORK_CODE),
        make_cell("markdown", SPECTROGRAM_MD),
        make_cell("code", SPECTROGRAM_CODE),
        make_cell("markdown", RESAMPLE_MD),
        make_cell("code", RESAMPLE_CODE),
        make_cell("markdown", TF2_LIKELIHOOD_MD),
        make_cell("code", TF2_LIKELIHOOD_CODE),
        make_cell("markdown", TF2_SAMPLER_MD),
        make_cell("code", TF2_SAMPLER_CODE),
        make_cell("markdown", CORNER_MD),
        make_cell("code", TF2_CORNER_CODE),
        make_cell("markdown", TF2_PAPER_MD),
        make_cell("code", TF2_PAPER_CODE),
    ]
    return make_notebook(cells)


def build_mlgw_notebook():
    cells = [
        make_cell("markdown", MLGW_HEADER_MD),
        make_cell("markdown", SETUP_MD),
        make_cell("code", SETUP_CODE),
        make_cell("markdown", DOWNLOAD_DATA_MD),
        make_cell("code", DOWNLOAD_DATA_CODE),
        make_cell("code", JAX_SETUP_CODE),
        make_cell("markdown", MLGW_WAVEFORM_MD),
        make_cell("code", MLGW_WAVEFORM_CODE),
        make_cell("markdown", EVENT_PARAMS_MD),
        make_cell("code", EVENT_PARAMS_CODE.format(
            outdir_line='OUTDIR = "results_mlgw_bns_jax_sharpy_lowgrid"',
            label_line='LABEL = "GW170817_mlgw_bns_jax_sharpy_lowgrid"',
        )),
        make_cell("markdown", BUILD_NETWORK_MD),
        make_cell("code", BUILD_NETWORK_CODE),
        make_cell("markdown", SPECTROGRAM_MD),
        make_cell("code", SPECTROGRAM_CODE),
        make_cell("markdown", RESAMPLE_MD),
        make_cell("code", RESAMPLE_CODE),
        make_cell("markdown", MLGW_LIKELIHOOD_MD),
        make_cell("code", MLGW_LIKELIHOOD_CODE),
        make_cell("markdown", MLGW_SAMPLER_MD),
        make_cell("code", MLGW_SAMPLER_CODE),
        make_cell("markdown", CORNER_MD),
        make_cell("code", MLGW_CORNER_CODE),
        make_cell("markdown", MLGW_PAPER_MD),
        make_cell("code", MLGW_PAPER_CODE),
    ]
    return make_notebook(cells)


if __name__ == "__main__":
    out_dir = "/workspaces/mlgw_bns_jax"

    tf2_path = os.path.join(out_dir, "pe_taylorf2_sharpy_lowgrid.ipynb")
    with open(tf2_path, "w") as f:
        json.dump(build_taylorf2_notebook(), f, indent=1)
    print(f"Created: {tf2_path}")

    mlgw_path = os.path.join(out_dir, "pe_mlgw_bns_jax_sharpy_lowgrid.ipynb")
    with open(mlgw_path, "w") as f:
        json.dump(build_mlgw_notebook(), f, indent=1)
    print(f"Created: {mlgw_path}")
