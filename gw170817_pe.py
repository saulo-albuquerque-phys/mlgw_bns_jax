"""Full parameter estimation of GW170817 using bilby + mlgw_bns_jax.

This script fetches open data from GWOSC for GW170817, builds a
frequency-domain source model backed by the JAX-based mlgw_bns waveform
approximant, and runs a Bayesian PE campaign with dynesty.

Usage
-----
    python gw170817_pe.py                  # single-core
    python gw170817_pe.py --npool 4        # parallel with 4 workers

Requirements
------------
    pip install bilby gwpy
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

# ── JAX configuration (must come before any JAX import) ───────────────
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

import bilby
import logging
logging.getLogger("bilby").setLevel(logging.INFO)
# Suppress the repetitive zenith/azimuth conversion warnings
logging.getLogger("bilby").addFilter(
    lambda record: "zenith/azimuth" not in record.getMessage()
)
from bilby.gw.conversion import (
    generate_all_bns_parameters,
    convert_to_lal_binary_neutron_star_parameters,
)

# ── Load the JAX waveform model ──────────────────────────────────────
# Ensure the standalone loader is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from jax_import_n_predict import load_predict

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "mlgw_bns_jax_model.h5")

_predict_fn = jax.jit(load_predict(MODEL_PATH))

# ── GW170817 event parameters ────────────────────────────────────────
TRIGGER_TIME = 1187008882.43          # GPS trigger time
DURATION = 32                         # seconds of data to analyse
SAMPLING_FREQUENCY = 4096             # Hz
MAXIMUM_FREQUENCY = 2000              # Hz
MINIMUM_FREQUENCY = 20.0              # Hz  (low-frequency cut-off)
POST_TRIGGER_DURATION = 2             # seconds after trigger to keep
DATA_START_GPS = 1187008867            # GPS start of the downloaded data files
LABEL = "GW170817"
OUTDIR = "outdir_GW170817"


# =====================================================================
# Source model: bridge between bilby parameters and mlgw_bns_jax
# =====================================================================

def mlgw_bns_jax_frequency_domain_source_model(
    frequency_array: np.ndarray,
    mass_1: float,
    mass_2: float,
    luminosity_distance: float,
    theta_jn: float,
    phase: float,
    chi_1: float,
    chi_2: float,
    lambda_1: float,
    lambda_2: float,
    **kwargs,
) -> dict[str, np.ndarray]:
    """Frequency-domain BNS waveform using the mlgw_bns JAX model.

    Maps standard bilby CBC parameters to the mlgw_bns_jax ``predict``
    interface and returns ``{'plus': hp, 'cross': hc}`` as complex
    numpy arrays.
    """
    # Mask out the DC component (f=0) — model is undefined there.
    freq_mask = frequency_array > 0
    freqs = frequency_array[freq_mask]

    # ── Convert bilby masses to mlgw_bns parameterisation ────────
    # mlgw_bns expects q = m1/m2 >= 1 and total_mass = m1 + m2
    m1 = max(mass_1, mass_2)
    m2 = min(mass_1, mass_2)
    mass_ratio = m1 / m2
    total_mass = m1 + m2

    params = jnp.array([mass_ratio, lambda_1, lambda_2, chi_1, chi_2])

    hp_jax, hc_jax = _predict_fn(
        params,
        jnp.array(freqs),
        jnp.array(total_mass),
        jnp.array(luminosity_distance),
        jnp.array(theta_jn),
    )

    hp_out = np.array(hp_jax)
    hc_out = np.array(hc_jax)

    # Apply reference phase rotation: h → h * e^{-i * 2 * phase}
    # (factor-of-2 convention matches LAL / bilby standard)
    phase_shift = np.exp(-2j * phase)
    hp_out *= phase_shift
    hc_out *= phase_shift

    # Build full-length arrays (zero where f <= 0)
    hp_full = np.zeros(len(frequency_array), dtype=complex)
    hc_full = np.zeros(len(frequency_array), dtype=complex)
    hp_full[freq_mask] = hp_out
    hc_full[freq_mask] = hc_out

    return {"plus": hp_full, "cross": hc_full}


# =====================================================================
# Main
# =====================================================================

def _load_data_from_gwosc(interferometers, start_time):
    """Try to fetch strain data from GWOSC (requires internet)."""
    for ifo in interferometers:
        ifo.set_strain_data_from_channel_name(
            channel=f"{ifo.name}:GWOSC-16KHZ_R1_STRAIN",
            sampling_frequency=SAMPLING_FREQUENCY,
            duration=DURATION,
            start_time=start_time,
        )
        ifo.minimum_frequency = MINIMUM_FREQUENCY
        ifo.maximum_frequency = MAXIMUM_FREQUENCY


def _load_data_from_local_files(interferometers, start_time, data_dir="./gw170817_data"):
    """Load strain from local files previously downloaded from GWOSC.

    Supports .txt, .hdf5 and .gwf formats.
    """
    import glob

    for ifo in interferometers:
        # Try .txt first (ASCII), then .hdf5, then .gwf
        for ext in ("txt", "hdf5", "gwf"):
            pattern = os.path.join(data_dir, f"*{ifo.name}*GWOSC*.{ext}")
            matches = sorted(glob.glob(pattern))
            if matches:
                break
        if not matches:
            raise FileNotFoundError(
                f"No data file found for {ifo.name} in {data_dir}. "
                f"Download from https://gwosc.org/eventapi/html/GWTC-1-confident/GW170817/"
            )

        filepath = matches[0]
        if filepath.endswith(".txt"):
            # GWOSC ASCII: 3-line header then one strain value per line
            strain = np.loadtxt(filepath, comments="#")
            ifo.set_strain_data_from_frequency_domain_strain(
                sampling_frequency=SAMPLING_FREQUENCY,
                duration=DURATION,
                start_time=DATA_START_GPS,
                frequency_domain_strain=np.fft.rfft(strain) / SAMPLING_FREQUENCY,
            )
        else:
            ifo.set_strain_data_from_frame_file(
                frame_file=filepath,
                sampling_frequency=SAMPLING_FREQUENCY,
                duration=DURATION,
                start_time=start_time,
            )
        ifo.minimum_frequency = MINIMUM_FREQUENCY
        ifo.maximum_frequency = MAXIMUM_FREQUENCY


def _use_gaussian_noise(interferometers, start_time):
    """Use simulated Gaussian noise (for testing the pipeline without real data)."""
    for ifo in interferometers:
        ifo.set_strain_data_from_power_spectral_density(
            sampling_frequency=SAMPLING_FREQUENCY,
            duration=DURATION,
            start_time=start_time,
        )
        ifo.minimum_frequency = MINIMUM_FREQUENCY
        ifo.maximum_frequency = MAXIMUM_FREQUENCY


def main(npool: int = 1) -> None:
    bilby.core.utils.setup_logger(outdir=OUTDIR, label=LABEL, log_level="info")

    # ── 1. Set up interferometers and fetch open data ────────────
    interferometers = bilby.gw.detector.InterferometerList(["H1", "L1", "V1"])

    # Analysis segment: [start, start + DURATION]
    start_time = TRIGGER_TIME + POST_TRIGGER_DURATION - DURATION

    # Try data sources in order: local files → GWOSC → Gaussian noise fallback
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gw170817_data")
    if os.path.isdir(data_dir):
        print(f"Loading strain data from local files in {data_dir}")
        _load_data_from_local_files(interferometers, start_time, data_dir)
    else:
        try:
            print("Fetching strain data from GWOSC (requires internet)...")
            _load_data_from_gwosc(interferometers, start_time)
        except Exception as e:
            print(f"GWOSC fetch failed: {e}")
            print("WARNING: Falling back to simulated Gaussian noise — "
                  "results will NOT be physical. Download real data from "
                  "https://gwosc.org/eventapi/html/GWTC-1-confident/GW170817/ "
                  f"into {data_dir}/ to use real strain.")
            _use_gaussian_noise(interferometers, start_time)

    # ── 2. Build the waveform generator ──────────────────────────
    waveform_generator = bilby.gw.WaveformGenerator(
        duration=DURATION,
        sampling_frequency=SAMPLING_FREQUENCY,
        frequency_domain_source_model=mlgw_bns_jax_frequency_domain_source_model,
        parameter_conversion=convert_to_lal_binary_neutron_star_parameters,
        waveform_arguments=dict(
            minimum_frequency=MINIMUM_FREQUENCY,
        ),
    )

    # ── 3. Define priors ─────────────────────────────────────────
    priors = bilby.core.prior.PriorDict()

    # Sample in chirp_mass + mass_ratio (standard bilby parameterisation)
    priors["chirp_mass"] = bilby.core.prior.Uniform(
        minimum=1.18, maximum=1.21, name="chirp_mass",
        latex_label=r"$\mathcal{M}$", unit=r"$M_\odot$",
    )
    priors["mass_ratio"] = bilby.core.prior.Uniform(
        minimum=0.5, maximum=1.0, name="mass_ratio",
        latex_label=r"$q$",
    )

    # Aligned spins only
    priors["chi_1"] = bilby.core.prior.Uniform(
        minimum=-0.05, maximum=0.05, name="chi_1",
        latex_label=r"$\chi_1$",
    )
    priors["chi_2"] = bilby.core.prior.Uniform(
        minimum=-0.05, maximum=0.05, name="chi_2",
        latex_label=r"$\chi_2$",
    )

    # Tidal deformabilities — wide uniform prior
    priors["lambda_1"] = bilby.core.prior.Uniform(
        minimum=0, maximum=5000, name="lambda_1",
        latex_label=r"$\Lambda_1$",
    )
    priors["lambda_2"] = bilby.core.prior.Uniform(
        minimum=0, maximum=5000, name="lambda_2",
        latex_label=r"$\Lambda_2$",
    )

    # Luminosity distance — uniform-in-volume prior, constrained
    priors["luminosity_distance"] = bilby.gw.prior.UniformSourceFrame(
        minimum=10, maximum=100, name="luminosity_distance",
        latex_label=r"$d_L$", unit="Mpc",
    )

    # Inclination
    priors["theta_jn"] = bilby.core.prior.Sine(
        minimum=0, maximum=np.pi, name="theta_jn",
        latex_label=r"$\theta_{JN}$",
    )

    # Sky location and polarisation
    priors["ra"] = bilby.core.prior.Uniform(
        minimum=0, maximum=2 * np.pi, name="ra",
        latex_label=r"$\alpha$", boundary="periodic",
    )
    priors["dec"] = bilby.core.prior.Cosine(
        minimum=-np.pi / 2, maximum=np.pi / 2, name="dec",
        latex_label=r"$\delta$",
    )
    priors["psi"] = bilby.core.prior.Uniform(
        minimum=0, maximum=np.pi, name="psi",
        latex_label=r"$\psi$", boundary="periodic",
    )

    # Reference phase
    priors["phase"] = bilby.core.prior.Uniform(
        minimum=0, maximum=2 * np.pi, name="phase",
        latex_label=r"$\phi$", boundary="periodic",
    )

    # Geocentric time — tight window around trigger
    priors["geocent_time"] = bilby.core.prior.Uniform(
        minimum=TRIGGER_TIME - 0.1,
        maximum=TRIGGER_TIME + 0.1,
        name="geocent_time",
        latex_label=r"$t_c$",
        unit="s",
    )

    # ── 4. Construct the likelihood ──────────────────────────────
    likelihood = bilby.gw.GravitationalWaveTransient(
        interferometers=interferometers,
        waveform_generator=waveform_generator,
        priors=priors,
        time_marginalization=True,
        distance_marginalization=True,
        phase_marginalization=True,
        reference_frame="H1L1V1",
        jitter_time=True,
    )

    # ── 5. Run the sampler ───────────────────────────────────────
    sampler_kwargs = dict(
        likelihood=likelihood,
        priors=priors,
        sampler="dynesty",
        npoints=1024,
        walks=100,
        nact=10,
        maxmcmc=5000,
        injection_parameters=None,
        outdir=OUTDIR,
        label=LABEL,
        conversion_function=generate_all_bns_parameters,
        result_class=bilby.gw.result.CBCResult,
    )
    if npool > 1:
        sampler_kwargs["npool"] = npool
    result = bilby.run_sampler(**sampler_kwargs)

    # ── 6. Plot results ──────────────────────────────────────────
    result.plot_corner()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GW170817 PE with mlgw_bns_jax")
    parser.add_argument(
        "--npool", type=int, default=1,
        help="Number of parallel workers for likelihood evaluation (default: 1)",
    )
    args = parser.parse_args()
    main(npool=args.npool)
