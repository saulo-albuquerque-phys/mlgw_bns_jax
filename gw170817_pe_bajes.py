"""Parameter estimation of GW170817 using bajes + dynesty + mlgw_bns_jax.

Replicates the pipeline from arXiv:2210.15684 (Section IV):
    - bajes framework with nested sampling (dynesty)
    - Analytic marginalisation over reference phase and coalescence time
    - Paper priors: mc ∈ [1.18, 1.21], q ∈ [1, 2] (≡ paper q=m2/m1 ∈ [0.5,1]),
      χ ∈ [-0.5, 0.5], Λ ∈ [5, 5000], D_L ∈ [1, 75] Mpc
    - f ∈ [23, 2000] Hz, 4096 Hz sampling rate, 32 s segment
    - Waveform: mlgw_bns_jax (autonomous JAX-based BNS model)

Usage
-----
    python gw170817_pe_bajes.py

The script uses the ``bajes`` library (v1.2.0+) with its ``dynesty`` sampler
wrapper.  The mlgw_bns_jax waveform is injected by registering a custom
approximant in ``bajes.obs.gw.waveform.__approx_dict__``.
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np

# ── JAX configuration ────────────────────────────────────────────────
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

# ── Make helpers importable ──────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from jax_import_n_predict import load_predict

# ── Load the mlgw_bns_jax waveform model ─────────────────────────────
MODEL_PATH = os.path.join(SCRIPT_DIR, "mlgw_bns_jax_model.h5")
_mlgw_predict = load_predict(MODEL_PATH)

# =====================================================================
# Custom bajes waveform wrapper
# =====================================================================

class mlgw_bns_jax_wrapper:
    """Bajes-compatible wrapper for the mlgw_bns_jax waveform model.

    Conforms to the interface expected by ``bajes.obs.gw.waveform``:

    * ``__init__(self, **kwargs)`` — receives ``seglen``, ``srate``, ``domain``.
    * ``__call__(self, freqs, params)`` — returns ``(hp, hc)`` as numpy arrays.

    The ``params`` dict supplied by bajes contains (among others):
        mchirp, q, mtot, s1z, s2z, lambda1, lambda2,
        distance, iota / cos_iota, phi_ref, …
    """

    def __init__(self, **kwargs):
        pass  # no additional state needed; model loaded at module level

    def __call__(self, freqs, params):
        # Map bajes params → mlgw_bns_jax params
        q  = params["q"]                    # mass ratio  m1/m2 ≥ 1
        l1 = params["lambda1"]
        l2 = params["lambda2"]
        s1 = params.get("s1z", 0.0)
        s2 = params.get("s2z", 0.0)

        total_mass   = params["mtot"]
        distance_mpc = params["distance"]
        inclination  = params["iota"]

        mlgw_params = jnp.array([q, l1, l2, s1, s2])
        freqs_jax   = jnp.asarray(freqs)

        hp, hc = _mlgw_predict(
            mlgw_params,
            freqs_jax,
            total_mass=jnp.asarray(total_mass),
            distance_mpc=jnp.asarray(distance_mpc),
            inclination=jnp.asarray(inclination),
        )

        # bajes expects numpy (not JAX) arrays
        return np.asarray(hp), np.asarray(hc)


# ── Register the custom approximant in bajes ─────────────────────────
# We add a module-level reference so the import machinery can find the
# wrapper class.  Then register it in __approx_dict__.
import bajes.obs.gw.waveform as _wf_mod

# Expose the wrapper at a known import path by monkey-patching the module
_wf_mod.mlgw_bns_jax_wrapper = mlgw_bns_jax_wrapper

_wf_mod.__approx_dict__["MLGW-BNS-JAX"] = {
    "path": "bajes.obs.gw.waveform.mlgw_bns_jax_wrapper",
    "type": "cls",
    "domain": "freq",
}

# =====================================================================
# GW170817 configuration – matching arXiv:2210.15684 Table I
# =====================================================================

TRIGGER_TIME  = 1187008882.43        # GPS trigger time
SRATE         = 4096                 # sampling rate [Hz]
SEGLEN        = 32                   # segment length [s] (limited by data file)
F_MIN         = 23.0                 # Hz  (paper value)
F_MAX         = 2000.0               # Hz  (paper value)
APPROX        = "MLGW-BNS-JAX"

DATA_DIR = os.path.join(SCRIPT_DIR, "gw170817_data")
OUTDIR   = os.path.join(SCRIPT_DIR, "outdir_GW170817_bajes")
LABEL    = "GW170817_bajes"

# Sampler settings (paper: 3000 live points, nact=5, maxmcmc=12000)
NLIVE    = 3000
NACT     = 5
MAXMCMC  = 12000

# =====================================================================
# Helper functions
# =====================================================================

DATA_START_GPS = 1187008867.0   # GPS start of the 32-s data files
DATA_NSAMPLES  = 131072          # 32 s × 4096 Hz


def load_strain(det_name: str) -> np.ndarray:
    """Load a single-column ASCII strain file for *det_name* (H1/L1/V1)."""
    prefix = {"H1": "H-H1", "L1": "L-L1", "V1": "V-V1"}[det_name]
    fname  = f"{prefix}_GWOSC_4KHZ_R1-1187008867-32.txt"
    path   = os.path.join(DATA_DIR, fname)
    return np.loadtxt(path)


# =====================================================================
# Main
# =====================================================================

def main() -> None:
    os.makedirs(OUTDIR, exist_ok=True)

    t_start = time.perf_counter()
    print(f"[bajes PE]  GW170817 – mlgw_bns_jax — {APPROX}")
    print(f"  f ∈ [{F_MIN}, {F_MAX}] Hz,  seglen = {SEGLEN} s,  srate = {SRATE} Hz")
    print(f"  nlive = {NLIVE},  nact = {NACT},  maxmcmc = {MAXMCMC}")

    # ------------------------------------------------------------------
    # 1. Load strain data and create bajes Series / Noise objects
    # ------------------------------------------------------------------
    from bajes.obs.gw.strain import Series
    from bajes.obs.gw.noise  import Noise, evaluate_psd
    from bajes.obs.gw.detector import Detector

    detector_names = ["H1", "L1", "V1"]

    datas  = {}
    noises = {}
    dets   = {}
    freqs  = None

    for ifo in detector_names:
        print(f"  Loading {ifo} …")
        strain = load_strain(ifo)

        # ── Create bajes Series (time-domain) ────────────────────
        # t_gps must equal the reference time used for the detector;
        # bajes centres the analysis on this time.
        series = Series(
            "time", strain,
            srate  = SRATE,
            seglen = SEGLEN,
            f_min  = F_MIN,
            f_max  = F_MAX,
            t_gps  = TRIGGER_TIME,
        )
        datas[ifo] = series

        if freqs is None:
            freqs = series.freqs

        # ── Estimate PSD using Welch's method ────────────────────
        # Use 4-s subsegments with 50 % overlap
        psd_freqs, psd_vals = evaluate_psd(
            strain, dt=1.0 / SRATE, subseglen=4.0, overlap_fraction=0.5
        )
        asd = np.sqrt(psd_vals)
        noise = Noise(psd_freqs, asd, f_min=F_MIN, f_max=F_MAX)
        noises[ifo] = noise

        # ── Detector geometry ────────────────────────────────────
        det = Detector(ifo, t_gps=TRIGGER_TIME)
        dets[ifo] = det

    # ------------------------------------------------------------------
    # 2. Build the GW likelihood
    # ------------------------------------------------------------------
    from bajes.pipe.log_like import GWLikelihood

    like = GWLikelihood(
        ifos   = detector_names,
        datas  = datas,
        dets   = dets,
        noises = noises,
        freqs  = freqs,
        srate  = SRATE,
        seglen = SEGLEN,
        approx = APPROX,
        marg_phi_ref    = True,   # analytic marginalisation (as in paper)
        marg_time_shift = True,   # analytic marginalisation (as in paper)
    )
    print("  Likelihood initialised.")

    # ------------------------------------------------------------------
    # 3. Build the prior  (Table I of arXiv:2210.15684)
    # ------------------------------------------------------------------
    from bajes.inf.prior import Prior, Parameter, Constant

    parameters = [
        # Chirp mass
        Parameter(name="mchirp", min=1.18, max=1.21, prior="uniform"),
        # Mass ratio  q = m1/m2 ≥ 1  (paper uses q = m2/m1 ∈ [0.5,1],
        # which is the same physical space flipped)
        Parameter(name="q", min=1.0, max=2.0, prior="uniform"),
        # Aligned spins
        Parameter(name="s1z", min=-0.5, max=0.5, prior="uniform"),
        Parameter(name="s2z", min=-0.5, max=0.5, prior="uniform"),
        # Tidal deformabilities
        Parameter(name="lambda1", min=5.0, max=5000.0, prior="uniform"),
        Parameter(name="lambda2", min=5.0, max=5000.0, prior="uniform"),
        # Luminosity distance
        Parameter(name="distance", min=1.0, max=75.0, prior="quadratic"),
        # Inclination  (sinusoidal prior → uniform in cos(iota))
        Parameter(name="cos_iota", min=-1.0, max=1.0, prior="uniform"),
        # Sky location
        Parameter(name="ra",  min=0.0, max=2 * np.pi, prior="uniform", periodic=1),
        Parameter(name="dec", min=-np.pi / 2, max=np.pi / 2, prior="cosinusoidal"),
        # Polarisation angle
        Parameter(name="psi", min=0.0, max=np.pi, prior="uniform", periodic=1),
    ]

    # Constants: things the likelihood needs but are not sampled
    # (phi_ref and time_shift are analytically marginalised)
    constants = [
        Constant(name="phi_ref", value=0.0),
        Constant(name="time_shift", value=0.0),
        Constant(name="t_gps", value=TRIGGER_TIME),
    ]

    prior = Prior(parameters, constants=constants)
    print(f"  Prior: {prior.ndim}-dimensional  ({[p.name for p in parameters]})")

    # ------------------------------------------------------------------
    # 4. Build the posterior and sampler
    # ------------------------------------------------------------------
    from bajes.inf import Posterior
    from bajes.inf.sampler.dynesty import SamplerDynesty

    posterior = Posterior(like, prior)

    sampler = SamplerDynesty(
        engine        = "dynesty",
        posterior     = posterior,
        nlive         = NLIVE,
        nact          = NACT,
        maxmcmc       = MAXMCMC,
        ncheckpoint   = 500,
        outdir        = OUTDIR,
    )
    print(f"  Sampler ready  (nlive={NLIVE}, nact={NACT}, maxmcmc={MAXMCMC})")

    # ------------------------------------------------------------------
    # 5. Run
    # ------------------------------------------------------------------
    print("  Running nested sampling …  (this will take a long time)")
    sampler.run()
    sampler.get_posterior()

    elapsed = time.perf_counter() - t_start
    print(f"  Sampling complete in {elapsed/3600:.1f} h")

    # ------------------------------------------------------------------
    # 6. Save results and produce corner plot
    # ------------------------------------------------------------------
    save_results(sampler, prior)


def save_results(sampler, prior):
    """Save posterior samples and generate a paper-style corner plot."""
    import json

    # ── Extract posterior samples ─────────────────────────────────
    names  = prior.names
    ndim   = prior.ndim
    # sampler.posterior_samples is a 2-D array: (Nsamples, ndim + 2)
    #   last two columns are log-likelihood and log-prior
    raw = np.array(sampler.posterior_samples)
    samples_dict = {n: raw[:, i] for i, n in enumerate(names)}

    # ── Derived quantities (matching paper Figure 9) ─────────────
    mc   = samples_dict["mchirp"]
    q    = samples_dict["q"]          # m1/m2 ≥ 1
    s1z  = samples_dict["s1z"]
    s2z  = samples_dict["s2z"]
    l1   = samples_dict["lambda1"]
    l2   = samples_dict["lambda2"]
    dist = samples_dict["distance"]

    eta     = q / (1.0 + q) ** 2
    m_total = mc / eta ** 0.6
    m1      = m_total * q / (1.0 + q)
    m2      = m_total / (1.0 + q)
    chi_eff = (m1 * s1z + m2 * s2z) / m_total
    # Combined tidal deformability
    lambda_tilde = (16.0 / 13.0) * (
        (m1 + 12.0 * m2) * m1 ** 4 * l1
        + (m2 + 12.0 * m1) * m2 ** 4 * l2
    ) / m_total ** 5
    q_paper = 1.0 / q  # paper convention: q = m2/m1

    # ── Save to JSON ─────────────────────────────────────────────
    out = {n: samples_dict[n].tolist() for n in names}
    out["chi_eff"] = chi_eff.tolist()
    out["lambda_tilde"] = lambda_tilde.tolist()
    out["q_paper"] = q_paper.tolist()
    out["distance"] = dist.tolist()
    out_path = os.path.join(OUTDIR, f"{LABEL}_posterior.json")
    with open(out_path, "w") as f:
        json.dump(out, f)
    print(f"  Posterior saved → {out_path}")

    # ── Corner plot ──────────────────────────────────────────────
    try:
        import corner
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plot_samples = np.column_stack([mc, q_paper, chi_eff, lambda_tilde, dist])
        labels = [
            r"$\mathcal{M}_c\;[M_\odot]$",
            r"$q$",
            r"$\chi_{\rm eff}$",
            r"$\tilde\Lambda$",
            r"$D_L\;[\rm Mpc]$",
        ]

        fig = corner.corner(
            plot_samples,
            labels=labels,
            quantiles=[0.05, 0.5, 0.95],
            show_titles=True,
            title_kwargs={"fontsize": 12},
            color="tab:orange",
            levels=(0.5, 0.9),
            plot_datapoints=False,
            fill_contours=True,
        )
        fig.suptitle("GW170817 — bajes + dynesty + mlgw_bns_jax", fontsize=14, y=1.02)
        fig_path = os.path.join(OUTDIR, f"{LABEL}_corner.png")
        fig.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Corner plot saved → {fig_path}")
    except ImportError:
        print("  corner / matplotlib not available — skipping plot")


if __name__ == "__main__":
    main()
