"""Relative binning likelihood for SHARPy GW parameter estimation.

Implements the method of Zackay, Dai & Venumadhav (2018) [arXiv:1806.08792]
to reduce the cost of each likelihood evaluation from O(N_freq) to O(N_bins),
where N_bins << N_freq.  This allows SHARPy's SMC sampler to run far more
particles simultaneously without exhausting GPU/CPU memory.

Memory benefit
--------------
Typical GW170817 analysis (T = 128 s, f ∈ [23, 2000] Hz):
    N_freq ≈ 3 000 points (after resampling) or ≈ 8 192 (full grid)
    N_bins ≈ 100 – 500 points (configurable)
    → 6 – 80× fewer complex numbers to store per particle per detector

Convention (matches SHARPy ``GW_likelihood.py`` exactly)
---------------------------------------------------------
    log L = −TwoDeltaTOverN · Σ_k |d_k − h_k|² / σ²_k

where σ²_k = PSD(f_k) · dt² and TwoDeltaTOverN = 2·dt/N.

Relative-binning summary quantities per detector, per frequency bin j
----------------------------------------------------------------------
    A0_j = Σ_{k∈bin_j} conj(d_k) · h0_k / σ²_k        (complex, NOT conjugated on h0)
    B0_j = Σ_{k∈bin_j} |h0_k|² / σ²_k                  (real, positive)
    dd   = Σ_k |d_k|² / σ²_k                            (real, constant)

Approximate likelihood
----------------------
    log L_RB ≈ −TwoDeltaTOverN · (dd − 2·Re[Σ_j r_j · A0_j] + Σ_j |r_j|² · B0_j)

where  r_j = H_proj(f_j ; θ) / H_proj(f_j ; θ_0)  is the ratio of the
projected detector strain at bin-centre frequency f_j, evaluated for
candidate parameters θ relative to the fiducial parameters θ_0.

The key approximation  h(f) ≈ r_j · h0(f)  for f ∈ bin_j  is accurate
when the bins are chosen fine enough that the waveform ratio varies by
≲ 0.1 rad in phase and ≲ 1 % in amplitude within each bin.

Note on waveform convention
---------------------------
The template function should return (hp, hc) WITHOUT applying an
additional conjugation.  The mlgw_bns_jax predict function returns
    hp(f) = pre_plus · A(f) · exp(+i φ(f))
in the standard positive-frequency Fourier convention.  A coalescence-
phase rotation exp(−i φ_c) is applied inside the template wrapper
(see ``_template_wrapper`` below) as an explicit overall phase, NOT as a
conjugation of the waveform.
"""

from __future__ import annotations

from typing import Callable, NamedTuple

import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

# ── SHARPy imports (needed for antenna patterns and time delay) ────────
from sharpy.GW_likelihood import antenna_pattern_functions
from sharpy.utils import TimeDelayFromEarthCenter, McQ2Masses


# ======================================================================
# Data structures
# ======================================================================

class RBSummaryDet(NamedTuple):
    """Precomputed relative-binning summary data for one detector."""
    A0: jnp.ndarray    # shape (n_bins,) complex — Σ conj(d)·h0/σ²  per bin
    B0: jnp.ndarray    # shape (n_bins,) real   — Σ |h0|²/σ²       per bin
    dd: float          # scalar real             — Σ |d|²/σ²
    TwoDeltaTOverN: float
    # Reference projected waveform at bin centres (for ratio computation)
    h0_bins: jnp.ndarray   # shape (n_bins,) complex


class RBNetwork(NamedTuple):
    """Relative-binning data for the full detector network."""
    summaries: list          # list of RBSummaryDet, one per detector
    f_bins: jnp.ndarray      # shape (n_bins,) — bin centre frequencies
    # Detector geometry (needed at evaluation time)
    latitudes: jnp.ndarray   # shape (n_det,)
    longitudes: jnp.ndarray
    elevations: jnp.ndarray
    gammas: jnp.ndarray
    zetas: jnp.ndarray
    trigtimes: jnp.ndarray
    T_durations: jnp.ndarray


# ======================================================================
# Bin-edge selection
# ======================================================================

def choose_bins_pn(
    f: np.ndarray,
    h0: np.ndarray,
    n_bins: int = 400,
    max_bin_width_hz: float = 8.0,
) -> np.ndarray:
    """Choose frequency bin edges based on phase of a reference waveform.

    Bin boundaries are placed so that:
    1. The accumulated waveform *phase change* within each bin is approximately
       equal (good for chirp-mass / spin accuracy).
    2. No bin is wider than ``max_bin_width_hz`` Hz (good for coalescence-time
       accuracy — prevents large linear-phase errors at high frequencies where
       the PN phase varies slowly).

    Parameters
    ----------
    f : np.ndarray, shape (N,)
        Frequency array in Hz.
    h0 : np.ndarray, shape (N,) complex
        Projected reference waveform at all frequencies.
    n_bins : int
        Target number of bins.
    max_bin_width_hz : float
        Maximum allowed bin width in Hz.  Ensures that a time-of-arrival
        shift δt_c introduces at most ``2π · max_bin_width_hz · δt_c`` rad
        of phase error within a bin.  Default 8 Hz → ≤ 0.05 rad for δt_c
        ≤ 1 ms.

    Returns
    -------
    bin_edges : np.ndarray, shape (≤ n_bins+1,) but at least 2 elements.
        Frequencies of bin boundaries, including f[0] and f[-1].
    """
    # Phase-based edges (equal phase change per bin)
    phase = np.unwrap(np.angle(h0))
    cum_phase = np.abs(phase - phase[0])
    total_phase = cum_phase[-1]
    if total_phase < 1.0:
        edges_phase = np.linspace(f[0], f[-1], n_bins + 1)
    else:
        target_phases = np.linspace(0.0, total_phase, n_bins + 1)
        edges_phase = np.interp(target_phases, cum_phase, f)
        edges_phase[0] = f[0]
        edges_phase[-1] = f[-1]

    # Frequency-based edges (maximum bin width constraint)
    n_freq_bins = max(n_bins, int(np.ceil((f[-1] - f[0]) / max_bin_width_hz)) + 1)
    edges_freq = np.linspace(f[0], f[-1], n_freq_bins + 1)

    # Merge both sets of edges and keep unique values
    edges_merged = np.unique(np.concatenate([edges_phase, edges_freq]))
    # Clip to valid frequency range
    edges_merged = edges_merged[
        (edges_merged >= f[0]) & (edges_merged <= f[-1])
    ]
    edges_merged[0] = f[0]
    edges_merged[-1] = f[-1]
    return edges_merged


def bin_centres(bin_edges: np.ndarray) -> np.ndarray:
    """Return the centre frequency of each bin."""
    return 0.5 * (bin_edges[:-1] + bin_edges[1:])


# ======================================================================
# Projected waveform evaluation
# ======================================================================

def project_waveform_at_freqs(
    params: jnp.ndarray,
    f: jnp.ndarray,
    template_fn: Callable,
    lat: float,
    lon: float,
    elev: float,
    gamma: float,
    zeta: float,
    trigtime: float,
    T_duration: float,
) -> jnp.ndarray:
    """Evaluate the projected detector strain at arbitrary frequencies ``f``.

    This mirrors SHARPy's ``project_waveform`` but accepts a custom frequency
    array and an arbitrary template function, making it usable at both the
    full grid and the sparse bin centres.

    Parameters
    ----------
    params : array, shape (13,)
        SHARPy parameter vector:
        [ra, dec, logdist, incl, phic, pol, mc, q, tc,
         chi1, chi2, lambda1, lambda2]
    f : array, shape (K,)
        Frequencies at which to evaluate the strain, in Hz.
    template_fn : callable
        (params, f) → (hp, hc) — waveform template.
    lat, lon, elev, gamma, zeta : float
        Detector geometry.
    trigtime : float
        GPS trigger time of the detector segment.
    T_duration : float
        Segment duration in seconds.

    Returns
    -------
    h : array, shape (K,) complex
        Projected strain at frequencies ``f``.
    """
    hp, hc = template_fn(params, f)

    fplus, fcross = antenna_pattern_functions(
        params, lat, lon, gamma, zeta, trigtime
    )

    ra = params[0]
    dec = params[1]
    tc = trigtime + params[8]
    timedelay = TimeDelayFromEarthCenter(lat, lon, elev, ra, dec, tc)
    timeshift = timedelay + params[8] + (T_duration - 1.0)
    shift = 2.0 * jnp.pi * f * timeshift

    h = (fplus * hp + fcross * hc) * (jnp.cos(shift) - 1j * jnp.sin(shift))
    return h


# ======================================================================
# Summary-data precomputation  (NumPy — runs once before the sampler)
# ======================================================================

def precompute_rb_summary(
    f_full: np.ndarray,
    data: np.ndarray,
    h0_full: np.ndarray,
    sigmasq: np.ndarray,
    TwoDeltaTOverN: float,
    bin_edges: np.ndarray,
    f_bins: np.ndarray,
    h0_bins: np.ndarray,
) -> RBSummaryDet:
    """Compute per-bin summary quantities for one detector.

    Parameters
    ----------
    f_full : array, shape (N,)
    data : array, shape (N,) complex — d(f), the observed frequency-domain data
    h0_full : array, shape (N,) complex — projected reference waveform H0(f)
    sigmasq : array, shape (N,) — PSD · dt²
    TwoDeltaTOverN : float
    bin_edges : array, shape (n_bins+1,) — bin boundaries
    f_bins : array, shape (n_bins,) — bin centres
    h0_bins : array, shape (n_bins,) complex — H0 at bin centres

    Returns
    -------
    RBSummaryDet
    """
    n_bins = len(f_bins)
    A0 = np.zeros(n_bins, dtype=complex)
    B0 = np.zeros(n_bins, dtype=float)
    dd = float(np.sum(np.abs(data) ** 2 / sigmasq))

    # Assign each frequency point to a bin
    bin_indices = np.searchsorted(bin_edges, f_full, side="right") - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    for j in range(n_bins):
        mask = bin_indices == j
        if not np.any(mask):
            continue
        d_j = data[mask]
        h0_j = h0_full[mask]
        s_j = sigmasq[mask]
        # A0_j = Σ conj(d) · h0 / σ²  — NO conjugation on h0
        A0[j] = np.sum(np.conj(d_j) * h0_j / s_j)
        B0[j] = float(np.sum(np.abs(h0_j) ** 2 / s_j))

    return RBSummaryDet(
        A0=jnp.array(A0, dtype=jnp.complex128),
        B0=jnp.array(B0, dtype=jnp.float64),
        dd=float(dd),
        TwoDeltaTOverN=float(TwoDeltaTOverN),
        h0_bins=jnp.array(h0_bins, dtype=jnp.complex128),
    )


# ======================================================================
# JAX-jittable single-detector RB likelihood
# ======================================================================

def _single_det_rb_logL(
    r: jnp.ndarray,
    summary: RBSummaryDet,
) -> jnp.ndarray:
    """Relative-binning log-likelihood for one detector.

    Parameters
    ----------
    r : array, shape (n_bins,) complex
        Waveform ratio H(f_j; θ) / H0(f_j; θ_0) at bin centres.
    summary : RBSummaryDet
        Precomputed per-bin summary data.

    Returns
    -------
    log_L : scalar
    """
    # cross term:  Re[ Σ_j r_j · A0_j ]
    cross = jnp.real(jnp.sum(r * summary.A0))
    # self term:   Σ_j |r_j|² · B0_j
    self_term = jnp.sum(jnp.abs(r) ** 2 * summary.B0)
    # log L = -TwoDeltaTOverN * (dd - 2·cross + self_term)
    return -summary.TwoDeltaTOverN * (summary.dd - 2.0 * cross + self_term)


# ======================================================================
# Main builder: returns a fast log-likelihood function
# ======================================================================

def build_rb_likelihood(
    batched_detector,
    fiducial_params: np.ndarray,
    template_fn: Callable,
    n_bins: int = 400,
) -> tuple[Callable, RBNetwork]:
    """Precompute relative-binning summary data and return a fast likelihood.

    Parameters
    ----------
    batched_detector : sharpy Detector (stacked)
        Output of ``GWNetwork.batched_detector``.
    fiducial_params : array, shape (13,)
        Reference parameter vector θ_0 (best-guess or approximate MAP).
        Should be close to the true signal parameters for accuracy.
    template_fn : callable
        (params, freq_array) → (hp, hc).
        Must NOT apply any additional conjugation to the polarisations.
    n_bins : int
        Number of frequency bins.  More bins → better accuracy but higher
        cost per evaluation.  400 bins is a good default for BNS PE.

    Returns
    -------
    log_likelihood_rb : callable
        (params_13) → scalar.  Drop-in replacement for SHARPy's
        ``log_likelihood_det``.
    rb_network : RBNetwork
        Struct storing all precomputed data (useful for inspection/testing).
    """
    n_det = len(batched_detector.latitude)

    # ── Extract geometry arrays ──────────────────────────────────────
    lats = np.array(batched_detector.latitude)
    lons = np.array(batched_detector.longitude)
    elevs = np.array(batched_detector.elevation)
    gammas = np.array(batched_detector.gamma)
    zetas = np.array(batched_detector.zeta)
    trigtimes = np.array(batched_detector.trigtime)
    T_durs = np.array(batched_detector.T)

    # ── Use first detector's frequency array for bin selection ───────
    f_full_np = np.array(batched_detector.Frequency[0])

    # ── Compute reference projected waveform for each detector ───────
    print("[RB] Computing reference waveform on full grid …")
    h0_full_list = []
    for i in range(n_det):
        h0_i = np.array(
            project_waveform_at_freqs(
                jnp.array(fiducial_params, dtype=jnp.float64),
                jnp.array(f_full_np, dtype=jnp.float64),
                template_fn,
                float(lats[i]), float(lons[i]), float(elevs[i]),
                float(gammas[i]), float(zetas[i]),
                float(trigtimes[i]), float(T_durs[i]),
            )
        )
        h0_full_list.append(h0_i)

    # Average-amplitude reference for bin selection
    h0_avg = np.mean(np.abs(np.stack(h0_full_list)), axis=0)
    h0_phase_ref = h0_full_list[0]  # use first detector's phase for bins

    # ── Choose bin edges ─────────────────────────────────────────────
    print(f"[RB] Choosing {n_bins} frequency bins …")
    # Use the network-averaged amplitude weighted by the phase of det 0
    h0_for_bins = h0_avg * np.exp(1j * np.angle(h0_phase_ref))
    # Mask out zero-amplitude points (outside signal band)
    amp_mask = h0_avg > 0.0
    if np.sum(amp_mask) < n_bins:
        # Too few non-zero points — fall back to uniform bins
        bin_edges = np.linspace(f_full_np[0], f_full_np[-1], n_bins + 1)
    else:
        bin_edges = choose_bins_pn(f_full_np, h0_for_bins, n_bins=n_bins)

    f_bins_np = bin_centres(bin_edges)
    f_bins_jax = jnp.array(f_bins_np, dtype=jnp.float64)

    # ── Compute reference waveform at bin centres for each detector ──
    h0_bins_list = []
    for i in range(n_det):
        h0_b = np.array(
            project_waveform_at_freqs(
                jnp.array(fiducial_params, dtype=jnp.float64),
                f_bins_jax,
                template_fn,
                float(lats[i]), float(lons[i]), float(elevs[i]),
                float(gammas[i]), float(zetas[i]),
                float(trigtimes[i]), float(T_durs[i]),
            )
        )
        h0_bins_list.append(h0_b)

    # ── Precompute summary data per detector ─────────────────────────
    print("[RB] Precomputing per-detector summary data …")
    summaries = []
    for i in range(n_det):
        data_i = np.array(batched_detector.FrequencySeries[i])
        sigmasq_i = np.array(batched_detector.sigmasq[i])
        TwoDTN_i = float(batched_detector.TwoDeltaTOverN[i])
        summary_i = precompute_rb_summary(
            f_full_np,
            data_i,
            h0_full_list[i],
            sigmasq_i,
            TwoDTN_i,
            bin_edges,
            f_bins_np,
            h0_bins_list[i],
        )
        summaries.append(summary_i)
        print(f"  det {i}: dd={summary_i.dd:.3e}, "
              f"|A0| max={float(jnp.max(jnp.abs(summary_i.A0))):.3e}, "
              f"B0 sum={float(jnp.sum(summary_i.B0)):.3e}")

    n_bins_actual = len(f_bins_np)
    print(f"[RB] Ready.  Full grid: {len(f_full_np)} pts → "
          f"RB bins: {n_bins_actual} pts "
          f"({len(f_full_np)//n_bins_actual}× reduction)")

    rb_network = RBNetwork(
        summaries=summaries,
        f_bins=f_bins_jax,
        latitudes=jnp.array(lats),
        longitudes=jnp.array(lons),
        elevations=jnp.array(elevs),
        gammas=jnp.array(gammas),
        zetas=jnp.array(zetas),
        trigtimes=jnp.array(trigtimes),
        T_durations=jnp.array(T_durs),
    )

    # ── Build JAX-jittable likelihood ────────────────────────────────
    # Freeze all arrays into the closure
    f_bins_frozen = f_bins_jax
    summaries_frozen = summaries
    lats_frozen = rb_network.latitudes
    lons_frozen = rb_network.longitudes
    elevs_frozen = rb_network.elevations
    gammas_frozen = rb_network.gammas
    zetas_frozen = rb_network.zetas
    trigs_frozen = rb_network.trigtimes
    Ts_frozen = rb_network.T_durations

    # Pre-extract Python scalars for the closure (needed for JAX JIT)
    lats_py = [float(rb_network.latitudes[i]) for i in range(n_det)]
    lons_py = [float(rb_network.longitudes[i]) for i in range(n_det)]
    elevs_py = [float(rb_network.elevations[i]) for i in range(n_det)]
    gammas_py = [float(rb_network.gammas[i]) for i in range(n_det)]
    zetas_py = [float(rb_network.zetas[i]) for i in range(n_det)]
    trigs_py = [float(rb_network.trigtimes[i]) for i in range(n_det)]
    Ts_py = [float(rb_network.T_durations[i]) for i in range(n_det)]

    def log_likelihood_rb(params: jnp.ndarray) -> jnp.ndarray:
        """Relative-binning log-likelihood (fast).

        Parameters
        ----------
        params : array, shape (13,)
            SHARPy parameter vector.

        Returns
        -------
        log_L : scalar
        """
        log_L = jnp.float64(0.0)
        for i in range(n_det):
            # Evaluate projected waveform at bin centres only
            # All geometry values are Python scalars (JAX-JIT compatible)
            h_bins_i = project_waveform_at_freqs(
                params,
                f_bins_frozen,
                template_fn,
                lats_py[i],
                lons_py[i],
                elevs_py[i],
                gammas_py[i],
                zetas_py[i],
                trigs_py[i],
                Ts_py[i],
            )
            # Waveform ratio (no conjugation on h)
            h0_b = summaries_frozen[i].h0_bins
            # Avoid division by zero using a small guard
            r_i = jnp.where(
                jnp.abs(h0_b) > 0.0,
                h_bins_i / h0_b,
                jnp.zeros_like(h_bins_i),
            )
            log_L = log_L + _single_det_rb_logL(r_i, summaries_frozen[i])
        return log_L

    return log_likelihood_rb, rb_network


# ======================================================================
# Signal-capture check (matched filter SNR)
# ======================================================================

def compute_matched_filter_snr(
    batched_detector,
    fiducial_params: np.ndarray,
    template_fn: Callable,
) -> dict:
    """Compute the optimal matched-filter network SNR for a given template.

    The *optimal* (phase-maximised) SNR is:
        SNR_opt² = 4 · df · Σ_f |h(f)|² / S_n(f)

    where the sum is over positive frequencies and df is the frequency
    resolution.  This equals the SNR that would be achieved with perfect
    phase alignment.  For GW170817 with the H1+L1+V1 network and a good
    template, the expected network SNR is ~32.

    Note: A negative *signed* SNR  (Σ conj(d)·h / σ²) at a given parameter
    point does not imply the signal is absent — it merely means the template
    phase is not aligned with the data at those specific parameters (e.g.
    wrong φ_c or ψ).  The optimal SNR above is phase-independent.

    Parameters
    ----------
    batched_detector : sharpy Detector
    fiducial_params : array, shape (13,)
    template_fn : callable

    Returns
    -------
    dict with keys:
        'snr_opt_det'     - list of per-detector optimal SNR
        'snr_opt_network' - network SNR (quadrature sum)
        'snr_signed_det'  - list of signed matched-filter SNR (phase dependent)
    """
    n_det = len(batched_detector.latitude)
    lats = np.array(batched_detector.latitude)
    lons = np.array(batched_detector.longitude)
    elevs = np.array(batched_detector.elevation)
    gammas = np.array(batched_detector.gamma)
    zetas = np.array(batched_detector.zeta)
    trigtimes = np.array(batched_detector.trigtime)
    T_durs = np.array(batched_detector.T)

    snr_opt_det = []
    snr_signed_det = []
    for i in range(n_det):
        f_i = np.array(batched_detector.Frequency[i])
        data_i = np.array(batched_detector.FrequencySeries[i])
        sigmasq_i = np.array(batched_detector.sigmasq[i])
        TwoDTN_i = float(batched_detector.TwoDeltaTOverN[i])

        h_i = np.array(
            project_waveform_at_freqs(
                jnp.array(fiducial_params, dtype=jnp.float64),
                jnp.array(f_i, dtype=jnp.float64),
                template_fn,
                float(lats[i]), float(lons[i]), float(elevs[i]),
                float(gammas[i]), float(zetas[i]),
                float(trigtimes[i]), float(T_durs[i]),
            )
        )

        # Optimal (phase-maximised) SNR:
        #   SNR_opt² = 2 · TwoDTN · Σ |h|² / σ²  = (h|h) in SHARPy convention
        hh = float(np.real(np.sum(np.abs(h_i) ** 2 / sigmasq_i)))
        snr_opt_i = float(np.sqrt(max(0.0, 2.0 * TwoDTN_i * hh)))
        snr_opt_det.append(snr_opt_i)

        # Signed cross-correlation (phase dependent — can be negative)
        dh = float(np.real(np.sum(np.conj(data_i) * h_i / sigmasq_i)))
        snr_signed_i = (2.0 * TwoDTN_i * dh) / snr_opt_i if snr_opt_i > 0 else 0.0
        snr_signed_det.append(snr_signed_i)

    snr_opt_network = float(np.sqrt(np.sum(np.array(snr_opt_det) ** 2)))
    return {
        "snr_opt_det": snr_opt_det,
        "snr_opt_network": snr_opt_network,
        "snr_signed_det": snr_signed_det,
    }
