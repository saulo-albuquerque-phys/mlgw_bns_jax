"""Full parameter estimation of GW170817 using bilby + IMRPhenomPv2_NRTidal.

Same analysis setup as gw170817_pe.py but using the LAL waveform
approximant IMRPhenomPv2_NRTidal instead of mlgw_bns_jax.

Usage
-----
    python gw170817_pe_imrphenom.py

Requirements
------------
    pip install bilby gwpy lalsuite
"""

from __future__ import annotations

import os
import logging

import numpy as np

import bilby
logging.getLogger("bilby").setLevel(logging.INFO)
logging.getLogger("bilby").addFilter(
    lambda record: "zenith/azimuth" not in record.getMessage()
)
from bilby.gw.conversion import (
    generate_all_bns_parameters,
    convert_to_lal_binary_neutron_star_parameters,
)

# ── GW170817 event parameters ────────────────────────────────────────
TRIGGER_TIME = 1187008882.43          # GPS trigger time
DURATION = 32                         # seconds of data to analyse
SAMPLING_FREQUENCY = 4096             # Hz
MAXIMUM_FREQUENCY = 2000              # Hz
MINIMUM_FREQUENCY = 20.0              # Hz  (low-frequency cut-off)
POST_TRIGGER_DURATION = 2             # seconds after trigger to keep
DATA_START_GPS = 1187008867            # GPS start of the downloaded data files
LABEL = "GW170817_IMRPhenomPv2_NRTidal"
OUTDIR = "outdir_GW170817_IMRPhenom"


# =====================================================================
# Data loading helpers (shared logic with gw170817_pe.py)
# =====================================================================

def _load_data_from_gwosc(interferometers, start_time):
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
    import glob

    for ifo in interferometers:
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
    for ifo in interferometers:
        ifo.set_strain_data_from_power_spectral_density(
            sampling_frequency=SAMPLING_FREQUENCY,
            duration=DURATION,
            start_time=start_time,
        )
        ifo.minimum_frequency = MINIMUM_FREQUENCY
        ifo.maximum_frequency = MAXIMUM_FREQUENCY


# =====================================================================
# Main
# =====================================================================

def main() -> None:
    bilby.core.utils.setup_logger(outdir=OUTDIR, label=LABEL, log_level="info")

    # ── 1. Set up interferometers and load data ──────────────────
    interferometers = bilby.gw.detector.InterferometerList(["H1", "L1", "V1"])
    start_time = TRIGGER_TIME + POST_TRIGGER_DURATION - DURATION

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
            print("WARNING: Falling back to simulated Gaussian noise.")
            _use_gaussian_noise(interferometers, start_time)

    # ── 2. Build the waveform generator (LAL approximant) ────────
    waveform_generator = bilby.gw.WaveformGenerator(
        duration=DURATION,
        sampling_frequency=SAMPLING_FREQUENCY,
        frequency_domain_source_model=bilby.gw.source.lal_binary_neutron_star,
        parameter_conversion=convert_to_lal_binary_neutron_star_parameters,
        waveform_arguments=dict(
            waveform_approximant="IMRPhenomPv2_NRTidal",
            reference_frequency=20.0,
            minimum_frequency=MINIMUM_FREQUENCY,
        ),
    )

    # ── 3. Define priors ─────────────────────────────────────────
    priors = bilby.core.prior.PriorDict()

    priors["chirp_mass"] = bilby.core.prior.Uniform(
        minimum=1.18, maximum=1.21, name="chirp_mass",
        latex_label=r"$\mathcal{M}$", unit=r"$M_\odot$",
    )
    priors["mass_ratio"] = bilby.core.prior.Uniform(
        minimum=0.5, maximum=1.0, name="mass_ratio",
        latex_label=r"$q$",
    )

    # Aligned spins (IMRPhenomPv2_NRTidal uses a_1/a_2 + tilt)
    priors["a_1"] = bilby.core.prior.Uniform(
        minimum=0, maximum=0.05, name="a_1",
        latex_label=r"$a_1$",
    )
    priors["a_2"] = bilby.core.prior.Uniform(
        minimum=0, maximum=0.05, name="a_2",
        latex_label=r"$a_2$",
    )
    priors["tilt_1"] = bilby.core.prior.Sine(
        minimum=0, maximum=np.pi, name="tilt_1",
        latex_label=r"$\\theta_1$",
    )
    priors["tilt_2"] = bilby.core.prior.Sine(
        minimum=0, maximum=np.pi, name="tilt_2",
        latex_label=r"$\\theta_2$",
    )
    priors["phi_12"] = bilby.core.prior.Uniform(
        minimum=0, maximum=2 * np.pi, name="phi_12",
        latex_label=r"$\phi_{12}$", boundary="periodic",
    )
    priors["phi_jl"] = bilby.core.prior.Uniform(
        minimum=0, maximum=2 * np.pi, name="phi_jl",
        latex_label=r"$\phi_{JL}$", boundary="periodic",
    )

    # Tidal deformabilities
    priors["lambda_1"] = bilby.core.prior.Uniform(
        minimum=0, maximum=5000, name="lambda_1",
        latex_label=r"$\Lambda_1$",
    )
    priors["lambda_2"] = bilby.core.prior.Uniform(
        minimum=0, maximum=5000, name="lambda_2",
        latex_label=r"$\Lambda_2$",
    )

    # Luminosity distance
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

    # Geocentric time
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
    result = bilby.run_sampler(
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

    # ── 6. Plot results ──────────────────────────────────────────
    result.plot_corner()


if __name__ == "__main__":
    main()
