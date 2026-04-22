#!/usr/bin/env python3
"""
NumPy-only ROQ basis builder for mlgw_bns_jax — no JAX dependency.

Designed to run on LIGO Jupyter servers (or any CPU-only environment)
where JAX causes memory / OOM problems.  All waveform evaluation and
linear algebra is done with NumPy and SciPy.

Features
--------
* **No JAX**: pure NumPy + SciPy, so memory usage is predictable.
* **Step-level checkpointing**: the greedy pre-selection saves its state
  after *every single step*, so even a 1900 s/step run can be resumed
  without repeating completed steps.
* **Phase-level checkpointing**: pre-selection, enrichment and EIM each
  checkpoint on completion, matching the original JAX builder.
* **Multiprocessing**: waveform batches can optionally be generated in
  parallel using ``multiprocessing`` (set ``n_workers > 1``).

Usage
-----
    python build_roq_basis_numpy.py                  # default config
    python build_roq_basis_numpy.py my_config.ini    # custom config

Configuration
-------------
Same ``config_roq_basis_2.ini`` used by the JAX builder.  The results
are written to the same directory structure so the outputs are
compatible with the JAX builder checkpoints.

Dependencies
------------
    numpy, scipy, h5py  (no jax / jaxlib required)
    The ``mlgw_bns_jax_model.h5`` file must be present in the working directory.
"""

from __future__ import annotations

import configparser
import gc
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import h5py
import numpy as np

# ─────────────────────────────────────────────────────────────────────
# Physical constants (identical to jax_import_n_predict.py)
# ─────────────────────────────────────────────────────────────────────

_SUN_MASS_SECONDS: float = 4.92549094830932e-6
_EULER_GAMMA: float = 0.57721566490153286060
_TF2_BASE: float = 3.668693487138444e-19
_AMP_SI_BASE: float = 4.2425873413901263e24


# ─────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────

@dataclass
class ROQConfig:
    """All parameters controlling an ROQ basis build."""

    # Waveform / frequency grid
    f_min: float = 23.0
    f_max: float = 2000.0
    seglen: float = 128.0

    # Tolerances
    tolerance_lin: float = 1e-4
    tolerance_qua: float = 1e-6

    # Pre-basis
    n_pre_basis_lin: int = 200
    n_pre_basis_qua: int = 10
    n_pre_basis_search_iter: int = 1000

    # Enrichment
    n_training_set_cycles: int = 3
    training_set_sizes: list = field(
        default_factory=lambda: [10_000, 100_000, 100_000]
    )
    training_set_rel_tol: list = field(
        default_factory=lambda: [0.1, 1.0, 1.0]
    )

    # Training range
    mc_range: tuple = (1.18, 1.21)
    q_range: tuple = (1.0, 2.0)
    s1z_range: tuple = (-0.5, 0.5)
    s2z_range: tuple = (-0.5, 0.5)
    lambda1_range: tuple = (5.0, 5000.0)
    lambda2_range: tuple = (5.0, 5000.0)
    iota_range: tuple = (0.0, np.pi)
    phiref_range: tuple = (0.0, 2 * np.pi)

    # I/O
    output_dir: str = "./roq_basis_mlgw_bns_numpy"
    model_path: str = "mlgw_bns_jax_model.h5"
    random_seed: int = 170817
    verbose: int = 1

    # Batching
    waveform_batch_size: int = 32   # number of waveforms per batch (no GPU limit)
    projection_batch_size: int = 500

    # Parallelism (multiprocessing workers for waveform generation; 1 = serial)
    n_workers: int = 1

    @property
    def delta_f(self) -> float:
        return 1.0 / self.seglen

    @property
    def n_freq(self) -> int:
        return int((self.f_max - self.f_min) / self.delta_f) + 1

    @property
    def frequencies(self) -> np.ndarray:
        return np.arange(self.f_min, self.f_max + self.delta_f / 2, self.delta_f)

    @classmethod
    def from_ini(cls, path: str) -> "ROQConfig":
        """Load from a JenpyROQ-style .ini config file."""
        c = configparser.ConfigParser()
        c.read(path)
        wf = c["Waveform_and_parametrisation"]
        roq = c["ROQ"]
        tr = c["Training_range"]
        io_sec = c["I/O"]

        sizes_str = roq.get("training-set-sizes", "10000,100000,100000")
        sizes = [int(s.strip()) for s in sizes_str.split(",")]

        rel_tol_str = roq.get("training-set-rel-tol", ",".join(["1.0"] * len(sizes)))
        rel_tols = [float(s.strip()) for s in rel_tol_str.split(",")]
        while len(rel_tols) < len(sizes):
            rel_tols.append(1.0)
        rel_tols = rel_tols[: len(sizes)]

        return cls(
            f_min=float(wf.get("f-min", 23.0)),
            f_max=float(wf.get("f-max", 2000.0)),
            seglen=float(wf.get("seglen", 128.0)),
            tolerance_lin=float(roq.get("tolerance-lin", 1e-4)),
            tolerance_qua=float(roq.get("tolerance-qua", 1e-6)),
            n_pre_basis_lin=int(roq.get("n-pre-basis-lin", 200)),
            n_pre_basis_qua=int(roq.get("n-pre-basis-qua", 10)),
            n_pre_basis_search_iter=int(roq.get("n-pre-basis-search-iter", 1000)),
            n_training_set_cycles=int(roq.get("n-training-set-cycles", 3)),
            training_set_sizes=sizes,
            training_set_rel_tol=rel_tols,
            mc_range=(float(tr.get("mc-min", 1.18)), float(tr.get("mc-max", 1.21))),
            q_range=(float(tr.get("q-min", 1.0)), float(tr.get("q-max", 2.0))),
            s1z_range=(float(tr.get("s1z-min", -0.5)), float(tr.get("s1z-max", 0.5))),
            s2z_range=(float(tr.get("s2z-min", -0.5)), float(tr.get("s2z-max", 0.5))),
            lambda1_range=(float(tr.get("lambda1-min", 5.0)), float(tr.get("lambda1-max", 5000.0))),
            lambda2_range=(float(tr.get("lambda2-min", 5.0)), float(tr.get("lambda2-max", 5000.0))),
            iota_range=(float(tr.get("iota-min", 0.0)), float(tr.get("iota-max", str(np.pi)))),
            phiref_range=(float(tr.get("phiref-min", 0.0)), float(tr.get("phiref-max", str(2 * np.pi)))),
            output_dir=io_sec.get("output", "./roq_basis_mlgw_bns_numpy"),
            random_seed=int(io_sec.get("random-seed", 170817)),
            verbose=int(io_sec.get("verbose", 1)),
        )


# ─────────────────────────────────────────────────────────────────────
# Post-Newtonian helper functions — pure NumPy
# ─────────────────────────────────────────────────────────────────────

def _compute_quadrupole_yy(lam: np.ndarray) -> np.ndarray:
    loglam = np.log(np.where(lam > 0.0, lam, 1.0))
    logCQ = (
        0.194
        + 0.0936 * loglam
        + 0.0474 * loglam ** 2
        - 4.21e-3 * loglam ** 3
        + 1.23e-4 * loglam ** 4
    )
    return np.where(lam <= 0.0, 1.0, np.exp(logCQ))


def _compute_lambda_tilde(m1: float, m2: float, l1: float, l2: float) -> float:
    M = m1 + m2
    return (16.0 / 13.0) * (
        (m1 + 12.0 * m2) * m1 ** 4 * l1
        + (m2 + 12.0 * m1) * m2 ** 4 * l2
    ) / M ** 5


def _compute_delta_lambda(m1: float, m2: float, l1: float, l2: float) -> float:
    M = m1 + m2
    eta = (m1 * m2) / M ** 2
    X = np.sqrt(max(1.0 - 4.0 * eta, 0.0))
    comb1 = (1690.0 * eta / 1319.0 - 4843.0 / 1319.0) * (m1 ** 4 * l1 - m2 ** 4 * l2) / M ** 4
    comb2 = (6162.0 * X / 1319.0) * (m1 ** 4 * l1 + m2 ** 4 * l2) / M ** 4
    return comb1 + comb2


def _PhifT7hPNComplete(f: np.ndarray, M: float, eta: float,
                       Lama: float, Lamb: float) -> np.ndarray:
    v = np.abs(np.pi * M * f * _SUN_MASS_SECONDS) ** (1.0 / 3.0)
    delta = np.sqrt(max(1.0 - 4.0 * eta, 0.0))
    Xa = 0.5 * (1.0 + delta)
    Xb = 0.5 * (1.0 - delta)
    Xa2 = Xa * Xa; Xa3 = Xa2 * Xa; Xa4 = Xa3 * Xa; Xa5 = Xa4 * Xa
    Xb2 = Xb * Xb; Xb3 = Xb2 * Xb; Xb4 = Xb3 * Xb; Xb5 = Xb4 * Xb
    v2 = v * v; v3 = v2 * v; v4 = v3 * v; v5 = v4 * v
    kapa = 3.0 * Lama * Xa4 * Xb
    kapb = 3.0 * Lamb * Xb4 * Xa
    pNa = -3.0 / (16.0 * eta) * (12.0 + Xa / Xb) if Xb != 0 else 0.0
    pNb = -3.0 / (16.0 * eta) * (12.0 + Xb / Xa) if Xa != 0 else 0.0
    p1a = 5.0 * (3179.0 - 919.0 * Xa - 2286.0 * Xa2 + 260.0 * Xa3) / (672.0 * (12.0 - 11.0 * Xa))
    p1b = 5.0 * (3179.0 - 919.0 * Xb - 2286.0 * Xb2 + 260.0 * Xb3) / (672.0 * (12.0 - 11.0 * Xb))
    p2a = -np.pi
    p2b = -np.pi
    denom_a = 12.0 - 11.0 * Xa
    denom_b = 12.0 - 11.0 * Xb
    p3a = (
        -5 * (
            -387973870.0 + 43246839.0 * Xa + 174965616.0 * Xa2 + 158378220.0 * Xa3
            - 20427120.0 * Xa4 + 4572288.0 * Xa5
        ) / 27433728.0
    ) / denom_a if denom_a != 0 else 0.0
    p3b = (
        -5 * (
            -387973870.0 + 43246839.0 * Xb + 174965616.0 * Xb2 + 158378220.0 * Xb3
            - 20427120.0 * Xb4 + 4572288.0 * Xb5
        ) / 27433728.0
    ) / denom_b if denom_b != 0 else 0.0
    p4a = -np.pi * (27719.0 - 22415.0 * Xa + 7598.0 * Xa2 - 10520.0 * Xa3) / (672.0 * denom_a) if denom_a != 0 else 0.0
    p4b = -np.pi * (27719.0 - 22127.0 * Xb + 7022.0 * Xb2 - 10232.0 * Xb3) / (672.0 * denom_b) if denom_b != 0 else 0.0
    return v5 * (
        kapa * pNa * (1.0 + p1a * v2 + p2a * v3 + p3a * v4 + p4a * v5)
        + kapb * pNb * (1.0 + p1b * v2 + p2b * v3 + p3b * v4 + p4b * v5)
    )


def _PhifQM3hPN(f: np.ndarray, M: float, eta: float,
                s1z: float, s2z: float, Lam1: float, Lam2: float) -> np.ndarray:
    v = np.abs(np.pi * M * f * _SUN_MASS_SECONDS) ** (1.0 / 3.0)
    v2 = v * v
    delta = np.sqrt(max(1.0 - 4.0 * eta, 0.0))
    X1 = 0.5 * (1.0 + delta)
    X2 = 0.5 * (1.0 - delta)
    at1 = X1 * s1z; at2 = X2 * s2z
    CQ1 = _compute_quadrupole_yy(np.array([Lam1]))[0] - 1.0
    CQ2 = _compute_quadrupole_yy(np.array([Lam2]))[0] - 1.0
    a2CQ_p = at1 ** 2 * CQ1 + at2 ** 2 * CQ2
    a2CQ_m = at1 ** 2 * CQ1 - at2 ** 2 * CQ2
    PhifQM = -75.0 / (64.0 * eta) * a2CQ_p / v
    PhifQM += ((45.0 / 16.0 * eta + 15635.0 / 896.0) * a2CQ_p + 2215.0 / 512.0 * delta * a2CQ_m) * v / eta
    PhifQM += -75.0 / (8.0 * eta) * a2CQ_p * v2 * np.pi
    return PhifQM


def _Phif3hPN(f: np.ndarray, M: float, eta: float,
              s1z: float = 0.0, s2z: float = 0.0,
              Lam: float = 0.0, dLam: float = 0.0) -> np.ndarray:
    vlso = 1.0 / np.sqrt(6.0)
    delta = np.sqrt(max(1.0 - 4.0 * eta, 0.0))
    v = np.abs(np.pi * M * f * _SUN_MASS_SECONDS) ** (1.0 / 3.0)
    v2 = v * v; v3 = v2 * v; v4 = v2 * v2
    v5 = v4 * v; v6 = v3 * v3; v7 = v3 * v4
    v10 = v5 * v5; v12 = v10 * v2
    eta2 = eta ** 2; eta3 = eta ** 3

    m1M = 0.5 * (1.0 + delta)
    m2M = 0.5 * (1.0 - delta)
    chi1L = s1z; chi2L = s2z
    chi1sq = s1z * s1z; chi2sq = s2z * s2z
    chi1dotchi2 = s1z * s2z
    SL = m1M * m1M * chi1L + m2M * m2M * chi2L
    dSigmaL = delta * (m2M * chi2L - m1M * chi1L)

    sigma = eta * (721.0 / 48.0 * chi1L * chi2L - 247.0 / 48.0 * chi1dotchi2)
    sigma += 719.0 / 96.0 * (m1M * m1M * chi1L * chi1L + m2M * m2M * chi2L * chi2L)
    sigma -= 233.0 / 96.0 * (m1M * m1M * chi1sq + m2M * m2M * chi2sq)
    phis_15PN = 188.0 * SL / 3.0 + 25.0 * dSigmaL
    ga = (554345.0 / 1134.0 + 110.0 * eta / 9.0) * SL + (13915.0 / 84.0 - 10.0 * eta / 3.0) * dSigmaL
    pn_ss3 = (326.75 / 1.12 + 557.5 / 1.8 * eta) * eta * chi1L * chi2L
    pn_ss3 += (
        (4703.5 / 8.4 + 2935.0 / 6.0 * m1M - 120.0 * m1M * m1M)
        + (-4108.25 / 6.72 - 108.5 / 1.2 * m1M + 125.5 / 3.6 * m1M * m1M)
    ) * m1M * m1M * chi1sq
    pn_ss3 += (
        (4703.5 / 8.4 + 2935.0 / 6.0 * m2M - 120.0 * m2M * m2M)
        + (-4108.25 / 6.72 - 108.5 / 1.2 * m2M + 125.5 / 3.6 * m2M * m2M)
    ) * m2M * m2M * chi2sq
    phis_3PN = np.pi * (3760.0 * SL + 1490.0 * dSigmaL) / 3.0 + pn_ss3
    phis_35PN = (
        -8980424995.0 / 762048.0 + 6586595.0 * eta / 756.0 - 305.0 * eta2 / 36.0
    ) * SL - (
        170978035.0 / 48384.0 - 2876425.0 * eta / 672.0 - 4735.0 * eta2 / 144.0
    ) * dSigmaL

    LO = 3.0 / 128.0 / eta / v5
    pointmass = (
        1
        + 20.0 / 9.0 * (743.0 / 336.0 + 11.0 / 4.0 * eta) * v2
        + (phis_15PN - 16.0 * np.pi) * v3
        + 10.0 * (3058673.0 / 1016064.0 + 5429.0 / 1008.0 * eta + 617.0 / 144.0 * eta2 - sigma) * v4
        + (38645.0 / 756.0 * np.pi - 65.0 / 9.0 * eta * np.pi - ga) * (1.0 + 3.0 * np.log(v / vlso)) * v5
        + (
            11583231236531.0 / 4694215680.0
            - 640.0 / 3.0 * np.pi ** 2
            - 6848.0 / 21.0 * (_EULER_GAMMA + np.log(4.0 * v))
            + (-15737765635.0 / 3048192.0 + 2255.0 * np.pi ** 2 / 12.0) * eta
            + 76055.0 / 1728.0 * eta2
            - 127825.0 / 1296.0 * eta3
            + phis_3PN
        ) * v6
        + (
            np.pi * (77096675.0 / 254016.0 + 378515.0 / 1512.0 * eta - 74045.0 / 756.0 * eta2)
            + phis_35PN
        ) * v7
    )
    tidal = Lam * v10 * (-39.0 / 2.0 - 3115.0 / 64.0 * v2) + dLam * 6595.0 / 364.0 * v12
    return LO * (pointmass + tidal)


def _Phif5hPN(f: np.ndarray, M: float, eta: float,
              s1z: float = 0.0, s2z: float = 0.0) -> np.ndarray:
    phi_35pn = _Phif3hPN(f, M, eta, s1z, s2z, 0.0, 0.0)

    v = (np.pi * M * f * _SUN_MASS_SECONDS) ** (1.0 / 3.0)
    v2 = v * v; v3 = v2 * v; v4 = v2 * v2
    v5 = v4 * v; v6 = v3 * v3; v7 = v3 * v4
    v8 = v7 * v; v9 = v8 * v; v10 = v5 * v5; v11 = v10 * v
    logv = np.log(np.where(v > 0, v, 1e-300))
    eta2 = eta ** 2; eta3 = eta ** 3
    log2 = 0.69314718055994528623
    log3 = 1.0986122886681097821

    coef_8pn = (
        - 36946947827.5 / 1601901100.8 * eta ** 4
        + 51004148102.5 / 1310646355.2 * eta3
        + (30060067316599.7 / 57668439628.8 - 39954.5 / 2721.6 * np.pi ** 2) * eta2
        + (
            -567987228950352.7 / 128152088064.0
            - 532292.8 / 396.9 * _EULER_GAMMA
            + 930221.5 / 5443.2 * np.pi ** 2
            - 142068.8 / 44.1 * log2
            + 2632.5 / 4.9 * log3
        ) * eta
        - 9049.0 / 56.7 * np.pi ** 2
        - 3681.2 / 18.9 * _EULER_GAMMA
        + 255071384399888515.3 / 83042553065472.0
        - 2632.5 / 19.6 * log3
        - 101102.0 / 396.9 * log2
    )
    coef_log8pn = -3 * (
        - 36946947827.5 / 1601901100.8 * eta ** 4
        + 51004148102.5 / 1310646355.2 * eta3
        + (30060067316599.7 / 57668439628.8 - 39954.5 / 2721.6 * np.pi ** 2) * eta2
        + (
            -567987228950352.7 / 128152088064.0
            - 532292.8 / 396.9 * _EULER_GAMMA
            + 930221.5 / 5443.2 * np.pi ** 2
            - 142068.8 / 44.1 * log2
            + 2632.5 / 4.9 * log3
        ) * eta
        - 9049.0 / 56.7 * np.pi ** 2
        - 3681.2 / 18.9 * _EULER_GAMMA
        + 255071384399888515.3 / 83042553065472.0
        - 2632.5 / 19.6 * log3
        - 101102.0 / 396.9 * log2
    )
    coef_loglog8pn = 9 * (266146.4 / 1190.7 * eta + 1840.6 / 56.7)
    coef_9pn = np.pi * (
        1032375.5 / 19958.4 * eta3
        + 4529333.5 / 12700.8 * eta2
        + (2255.0 / 6.0 * np.pi ** 2 - 149291726073.5 / 13412044.8) * eta
        - 640.0 / 3.0 * np.pi ** 2
        - 1369.6 / 2.1 * _EULER_GAMMA
        + 10534427947316.3 / 1877686272.0
        - 2739.2 / 2.1 * log2
    )
    coef_log9pn = -3 * 1369.6 / 6.3 * np.pi

    return phi_35pn + (3.0 / 128.0 / eta / v5) * (
        (coef_8pn + coef_log8pn * logv + coef_loglog8pn * logv * logv) * v8
        + (coef_9pn + coef_log9pn * logv) * v9
    )


def _Af3hPN(f: np.ndarray, M: float, eta: float,
            s1z: float = 0.0, s2z: float = 0.0,
            Lam: float = 0.0, dLam: float = 0.0,
            Deff: float = 1.0) -> np.ndarray:
    Mchirp = M * abs(eta) ** (3.0 / 5.0)
    delta = np.sqrt(max(1.0 - 4.0 * eta, 0.0))
    v = np.abs(np.pi * M * f * _SUN_MASS_SECONDS) ** (1.0 / 3.0)
    v2 = v * v; v3 = v2 * v; v4 = v2 * v2
    v5 = v4 * v; v6 = v3 * v3; v7 = v3 * v4
    eta2 = eta ** 2; eta3 = eta ** 3

    A0 = (
        abs(Mchirp) ** (5.0 / 6.0)
        / np.abs(f) ** (7.0 / 6.0)
        / Deff
        / abs(np.pi) ** (2.0 / 3.0)
        * np.sqrt(5.0 / 24.0)
    )

    chis = 0.5 * (s1z + s2z)
    chia = 0.5 * (s1z - s2z)
    be = 113.0 / 12.0 * (chis + delta * chia - 76.0 / 113.0 * eta * chis)
    sigma = (
        chia ** 2 * (81.0 / 16.0 - 20.0 * eta)
        + 81.0 / 8.0 * chia * chis * delta
        + chis ** 2 * (81.0 / 16.0 - eta / 4.0)
    )
    eps = delta * chia * (502429.0 / 16128.0 - 907.0 / 192.0 * eta) + chis * (
        5.0 / 48.0 * eta2 - 73921.0 / 2016.0 * eta + 502429.0 / 16128.0
    )

    return A0 * (
        1.0
        + v2 * (11.0 / 8.0 * eta + 743.0 / 672.0)
        + v3 * (be / 2.0 - 2.0 * np.pi)
        + v4 * (
            1379.0 / 1152.0 * eta2
            + 18913.0 / 16128.0 * eta
            + 7266251.0 / 8128512.0
            - sigma / 2.0
        )
        + v5 * (57.0 / 16.0 * np.pi * eta - 4757.0 * np.pi / 1344.0 + eps)
        + v6 * (
            856.0 / 105.0 * _EULER_GAMMA
            + 67999.0 / 82944.0 * eta3
            - 1041557.0 / 258048.0 * eta2
            - 451.0 / 96.0 * np.pi ** 2 * eta
            + 10.0 * np.pi ** 2 / 3.0
            + 3526813753.0 / 27869184.0 * eta
            - 29342493702821.0 / 500716339200.0
            + 856.0 / 105.0 * np.log(4.0 * v)
        )
        + v7 * (-1349.0 / 24192.0 * eta2 - 72221.0 / 24192.0 * eta - 5111593.0 / 2709504.0) * np.pi
    )


def _smoothly_connect_with_zero(f_natural: np.ndarray, pn_amp: np.ndarray,
                                pivot_1: float = 0.01, pivot_2: float = 0.02) -> np.ndarray:
    t = (f_natural - pivot_1) / (pivot_2 - pivot_1)
    smooth = (1.0 - np.cos(t * np.pi)) / 2.0
    blended = pn_amp * (1.0 - smooth) + 20.0 * smooth
    return np.where(
        f_natural < pivot_1,
        pn_amp,
        np.where(f_natural < pivot_2, blended, 20.0),
    )


# ─────────────────────────────────────────────────────────────────────
# MLP forward pass — pure NumPy
# ─────────────────────────────────────────────────────────────────────

def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0)

def _tanh(x: np.ndarray) -> np.ndarray:
    return np.tanh(x)

def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))

def _identity(x: np.ndarray) -> np.ndarray:
    return x

_ACTIVATIONS = {
    "relu": _relu,
    "tanh": _tanh,
    "logistic": _sigmoid,
    "identity": _identity,
}


# ─────────────────────────────────────────────────────────────────────
# Fast vectorized cubic spline evaluator (fixed knots, varying y & query)
# ─────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────
# Waveform predictor — pure NumPy/SciPy (no JAX)
# ─────────────────────────────────────────────────────────────────────

class NumpyPredictor:
    """NumPy re-implementation of the mlgw_bns_jax waveform predictor.

    Equivalent to the ``predict`` closure returned by
    ``jax_import_n_predict.load_predict`` but uses only NumPy and SciPy.

    Performance strategy
    --------------------
    * MLP, PCA and PN functions are **vectorised over all N waveforms at
      once** (cheap matrix operations, no Python loop over N).
    * The spline evaluation at the full ~253 k output grid is the dominant
      cost; ``predict_batch`` parallelises it over N via
      ``concurrent.futures.ThreadPoolExecutor``.  scipy C extensions
      release the GIL so threads provide real speedup on multi-core CPUs.
      Set ``n_workers`` (or ``cfg.n_workers``) to the number of physical
      cores available (e.g. 8–32 on a LIGO node).
    """

    def __init__(self, path: str):
        with h5py.File(path, "r") as f:
            # MLP
            self.activation = _ACTIVATIONS[f["mlp"].attrs["activation"]]
            n_layers = int(f["mlp"].attrs["n_layers"])
            self.coefs = [f[f"mlp/coef_{i}"][...].astype(np.float64) for i in range(n_layers)]
            self.intercepts = [f[f"mlp/intercept_{i}"][...].astype(np.float64) for i in range(n_layers)]

            # Scaler
            self.scaler_mean = f["scaler/mean"][...].astype(np.float64)
            self.scaler_scale = f["scaler/scale"][...].astype(np.float64)

            # PCA
            self.eigenvectors = f["pca/eigenvectors"][...].astype(np.float64)
            self.eigenvalues = f["pca/eigenvalues"][...].astype(np.float64)
            self.pca_mean = f["pca/mean"][...].astype(np.float64)
            self.pca_scaling = f["pca/principal_components_scaling"][...].astype(np.float64)
            pc_exponent = float(f["pca"].attrs["pc_exponent"])
            self.eigenvalue_scaling = self.eigenvalues ** pc_exponent

            # Grid
            frequencies_hz_np = f["grid/frequencies_hz"][...].astype(np.float64)
            frequencies_natural_np = f["grid/frequencies_natural"][...].astype(np.float64)
            amp_idx = f["grid/amplitude_indices"][...]
            phi_idx = f["grid/phase_indices"][...]
            self.M_ref = float(f["grid"].attrs["total_mass"])

        self.n_amp = len(amp_idx)
        self.n_phi = len(phi_idx)

        self.amp_freqs_hz      = frequencies_hz_np[amp_idx].astype(np.float64)
        self.phi_freqs_hz      = frequencies_hz_np[phi_idx].astype(np.float64)
        self.amp_freqs_natural = frequencies_natural_np[amp_idx].astype(np.float64)

    # ------------------------------------------------------------------
    # Vectorised MLP + PCA  (N waveforms in one shot)
    # ------------------------------------------------------------------

    def _mlp_forward_batch(self, x: np.ndarray) -> np.ndarray:
        """MLP forward: (N, n_in) → (N, n_out)."""
        x = (x - self.scaler_mean) / self.scaler_scale
        for W, b in zip(self.coefs[:-1], self.intercepts[:-1]):
            x = self.activation(x @ W + b)
        x = x @ self.coefs[-1] + self.intercepts[-1]
        return x

    def _nn_pca_predict_batch(self, x: np.ndarray) -> np.ndarray:
        """(N, 5) params → (N, n_amp + n_phi) reconstructed residuals."""
        scaled_pca  = self._mlp_forward_batch(x)
        pca_comp    = scaled_pca / self.eigenvalue_scaling
        scaled_data = pca_comp * self.pca_scaling
        zero_mean   = scaled_data @ self.eigenvectors.T
        return zero_mean + self.pca_mean

    # ------------------------------------------------------------------
    # Vectorised PN functions  (N scalars in, N × n_knots out)
    # ------------------------------------------------------------------

    def _pn_amp_batch(
        self,
        eta:         np.ndarray,  # (N,)
        s1z:         np.ndarray,  # (N,)
        s2z:         np.ndarray,  # (N,)
        lambdatilde: np.ndarray,  # (N,)
        dlambda:     np.ndarray,  # (N,)
    ) -> np.ndarray:
        """Return PN amplitude on amp_freqs_hz for all N: shape (N, n_amp)."""
        f  = self.amp_freqs_hz[None, :]   # (1, n_amp) broadcast to (N, n_amp)
        M  = self.M_ref

        Mchirp = M * np.abs(eta[:, None]) ** (3.0 / 5.0)
        delta  = np.sqrt(np.maximum(1.0 - 4.0 * eta[:, None], 0.0))
        v  = np.abs(np.pi * M * f * _SUN_MASS_SECONDS) ** (1.0 / 3.0)
        v2 = v * v; v3 = v2 * v; v4 = v2 * v2
        v5 = v4 * v; v6 = v3 * v3; v7 = v3 * v4
        eta2 = (eta ** 2)[:, None]; eta3 = (eta ** 3)[:, None]

        A0 = (
            np.abs(Mchirp) ** (5.0 / 6.0)
            / np.abs(f) ** (7.0 / 6.0)
            / np.abs(np.pi) ** (2.0 / 3.0)
            * np.sqrt(5.0 / 24.0)
        )

        chis = (0.5 * (s1z + s2z))[:, None]
        chia = (0.5 * (s1z - s2z))[:, None]
        be   = 113.0 / 12.0 * (chis + delta * chia - 76.0 / 113.0 * eta[:, None] * chis)
        sigma = (
            chia ** 2 * (81.0 / 16.0 - 20.0 * eta[:, None])
            + 81.0 / 8.0 * chia * chis * delta
            + chis ** 2 * (81.0 / 16.0 - eta[:, None] / 4.0)
        )
        eps = delta * chia * (502429.0 / 16128.0 - 907.0 / 192.0 * eta[:, None]) + chis * (
            5.0 / 48.0 * eta2 - 73921.0 / 2016.0 * eta[:, None] + 502429.0 / 16128.0
        )

        lam  = lambdatilde[:, None]; dlam = dlambda[:, None]
        pn = A0 * (
            1.0
            + v2 * (11.0 / 8.0 * eta[:, None] + 743.0 / 672.0)
            + v3 * (be / 2.0 - 2.0 * np.pi)
            + v4 * (
                1379.0 / 1152.0 * eta2 + 18913.0 / 16128.0 * eta[:, None]
                + 7266251.0 / 8128512.0 - sigma / 2.0
            )
            + v5 * (57.0 / 16.0 * np.pi * eta[:, None] - 4757.0 * np.pi / 1344.0 + eps)
            + v6 * (
                856.0 / 105.0 * _EULER_GAMMA
                + 67999.0 / 82944.0 * eta3
                - 1041557.0 / 258048.0 * eta2
                - 451.0 / 96.0 * np.pi ** 2 * eta[:, None]
                + 10.0 * np.pi ** 2 / 3.0
                + 3526813753.0 / 27869184.0 * eta[:, None]
                - 29342493702821.0 / 500716339200.0
                + 856.0 / 105.0 * np.log(4.0 * v)
            )
            + v7 * (-1349.0 / 24192.0 * eta2 - 72221.0 / 24192.0 * eta[:, None]
                    - 5111593.0 / 2709504.0) * np.pi
        )
        pn = pn * (_TF2_BASE * _AMP_SI_BASE / eta[:, None] / M ** 2)
        # Smooth connection to zero at low frequencies
        fn = self.amp_freqs_natural[None, :]
        t = (fn - 0.01) / 0.01
        smooth = (1.0 - np.cos(t * np.pi)) / 2.0
        blended = pn * (1.0 - smooth) + 20.0 * smooth
        pn = np.where(fn < 0.01, pn, np.where(fn < 0.02, blended, 20.0))
        return pn   # (N, n_amp)

    def _pn_phase_batch(
        self,
        eta:    np.ndarray,  # (N,)
        s1z:    np.ndarray,  # (N,)
        s2z:    np.ndarray,  # (N,)
        lam1:   np.ndarray,  # (N,)
        lam2:   np.ndarray,  # (N,)
        m1_arr: np.ndarray,  # (N,) component mass 1
        m2_arr: np.ndarray,  # (N,) component mass 2
    ) -> np.ndarray:
        """Return PN phase on phi_freqs_hz for all N: shape (N, n_phi)."""
        f  = self.phi_freqs_hz[None, :]
        M  = self.M_ref

        # Compute lambda_tilde and delta_lambda vectorised
        M_tot = m1_arr + m2_arr
        Lamt = (16.0 / 13.0) * (
            (m1_arr + 12.0 * m2_arr) * m1_arr ** 4 * lam1
            + (m2_arr + 12.0 * m1_arr) * m2_arr ** 4 * lam2
        ) / M_tot ** 5   # (N,)
        etav  = (m1_arr * m2_arr) / M_tot ** 2
        X     = np.sqrt(np.maximum(1.0 - 4.0 * etav, 0.0))
        dLamt = (
            (1690.0 * etav / 1319.0 - 4843.0 / 1319.0)
            * (m1_arr ** 4 * lam1 - m2_arr ** 4 * lam2) / M_tot ** 4
            + (6162.0 * X / 1319.0)
            * (m1_arr ** 4 * lam1 + m2_arr ** 4 * lam2) / M_tot ** 4
        )  # (N,)

        # _Phif5hPN (vectorised)
        vlso = 1.0 / np.sqrt(6.0)
        delta = np.sqrt(np.maximum(1.0 - 4.0 * eta[:, None], 0.0))
        v  = np.abs(np.pi * M * f * _SUN_MASS_SECONDS) ** (1.0 / 3.0)
        v2 = v * v; v3 = v2 * v; v4 = v2 * v2
        v5 = v4 * v; v6 = v3 * v3; v7 = v3 * v4
        v8 = v7 * v; v9 = v8 * v; v10 = v5 * v5; v11 = v10 * v; v12 = v10 * v2
        logv = np.log(np.where(v > 0, v, 1e-300))
        eta2 = (eta ** 2)[:, None]; eta3 = (eta ** 3)[:, None]
        log2 = 0.69314718055994528623; log3 = 1.0986122886681097821

        m1M = 0.5 * (1.0 + delta); m2M = 0.5 * (1.0 - delta)
        chi1L = s1z[:, None]; chi2L = s2z[:, None]
        SL = m1M * m1M * chi1L + m2M * m2M * chi2L
        dSigmaL = delta * (m2M * chi2L - m1M * chi1L)
        sigma = (eta[:, None] * (721.0 / 48.0 * chi1L * chi2L - 247.0 / 48.0 * chi1L * chi2L)
                 + 719.0 / 96.0 * (m1M ** 2 * chi1L ** 2 + m2M ** 2 * chi2L ** 2)
                 - 233.0 / 96.0 * (m1M ** 2 * chi1L ** 2 + m2M ** 2 * chi2L ** 2))
        phis_15PN = 188.0 * SL / 3.0 + 25.0 * dSigmaL
        ga = ((554345.0 / 1134.0 + 110.0 * eta[:, None] / 9.0) * SL
              + (13915.0 / 84.0 - 10.0 * eta[:, None] / 3.0) * dSigmaL)
        pn_ss3 = ((326.75 / 1.12 + 557.5 / 1.8 * eta[:, None]) * eta[:, None] * chi1L * chi2L
                  + (4703.5 / 8.4 + 2935.0 / 6.0 * m1M - 120.0 * m1M ** 2
                     - 4108.25 / 6.72 - 108.5 / 1.2 * m1M + 125.5 / 3.6 * m1M ** 2)
                  * m1M ** 2 * chi1L ** 2
                  + (4703.5 / 8.4 + 2935.0 / 6.0 * m2M - 120.0 * m2M ** 2
                     - 4108.25 / 6.72 - 108.5 / 1.2 * m2M + 125.5 / 3.6 * m2M ** 2)
                  * m2M ** 2 * chi2L ** 2)
        phis_3PN = np.pi * (3760.0 * SL + 1490.0 * dSigmaL) / 3.0 + pn_ss3
        phis_35PN = (
            (-8980424995.0 / 762048.0 + 6586595.0 * eta[:, None] / 756.0
             - 305.0 * eta2 / 36.0) * SL
            - (170978035.0 / 48384.0 - 2876425.0 * eta[:, None] / 672.0
               - 4735.0 * eta2 / 144.0) * dSigmaL
        )
        LO = 3.0 / 128.0 / eta[:, None] / v5
        phi_35pn = LO * (
            1.0
            + 20.0 / 9.0 * (743.0 / 336.0 + 11.0 / 4.0 * eta[:, None]) * v2
            + (phis_15PN - 16.0 * np.pi) * v3
            + 10.0 * (3058673.0 / 1016064.0 + 5429.0 / 1008.0 * eta[:, None]
                      + 617.0 / 144.0 * eta2 - sigma) * v4
            + (38645.0 / 756.0 * np.pi - 65.0 / 9.0 * eta[:, None] * np.pi - ga)
            * (1.0 + 3.0 * np.log(v / vlso)) * v5
            + (11583231236531.0 / 4694215680.0 - 640.0 / 3.0 * np.pi ** 2
               - 6848.0 / 21.0 * (_EULER_GAMMA + np.log(4.0 * v))
               + (-15737765635.0 / 3048192.0 + 2255.0 * np.pi ** 2 / 12.0) * eta[:, None]
               + 76055.0 / 1728.0 * eta2 - 127825.0 / 1296.0 * eta3 + phis_3PN) * v6
            + (np.pi * (77096675.0 / 254016.0 + 378515.0 / 1512.0 * eta[:, None]
                        - 74045.0 / 756.0 * eta2) + phis_35PN) * v7
        )

        # Higher-order terms (8–9 PN)
        coef_8 = (
            -36946947827.5 / 1601901100.8 * (eta ** 4)[:, None]
            + 51004148102.5 / 1310646355.2 * eta3
            + (30060067316599.7 / 57668439628.8 - 39954.5 / 2721.6 * np.pi ** 2) * eta2
            + (-567987228950352.7 / 128152088064.0
               - 532292.8 / 396.9 * _EULER_GAMMA + 930221.5 / 5443.2 * np.pi ** 2
               - 142068.8 / 44.1 * log2 + 2632.5 / 4.9 * log3) * eta[:, None]
            - 9049.0 / 56.7 * np.pi ** 2 - 3681.2 / 18.9 * _EULER_GAMMA
            + 255071384399888515.3 / 83042553065472.0
            - 2632.5 / 19.6 * log3 - 101102.0 / 396.9 * log2
        )
        coef_log8 = -3 * coef_8
        loglog8   = 9 * (266146.4 / 1190.7 * eta[:, None] + 1840.6 / 56.7)
        coef_9 = np.pi * (
            1032375.5 / 19958.4 * eta3 + 4529333.5 / 12700.8 * eta2
            + (2255.0 / 6.0 * np.pi ** 2 - 149291726073.5 / 13412044.8) * eta[:, None]
            - 640.0 / 3.0 * np.pi ** 2 - 1369.6 / 2.1 * _EULER_GAMMA
            + 10534427947316.3 / 1877686272.0 - 2739.2 / 2.1 * log2
        )
        coef_log9 = -3 * 1369.6 / 6.3 * np.pi

        phi_5pn = phi_35pn + LO * (
            (coef_8 + coef_log8 * logv + loglog8 * logv * logv) * v8
            + (coef_9 + coef_log9 * logv) * v9
        )

        # Tidal + QM phase
        phi_tidal = LO / (3.0 / 128.0 / eta[:, None] / v5) * (
            Lamt[:, None] * v10 * (-39.0 / 2.0 - 3115.0 / 64.0 * v2)
            + dLamt[:, None] * 6595.0 / 364.0 * v12
        )
        # QM phase (vectorised _PhifQM3hPN)
        CQ1 = (_compute_quadrupole_yy(lam1) - 1.0)[:, None]  # (N, 1)
        CQ2 = (_compute_quadrupole_yy(lam2) - 1.0)[:, None]
        at1 = (0.5 * (1.0 + np.sqrt(np.maximum(1.0 - 4.0 * eta, 0.0))) * s1z)[:, None]
        at2 = (0.5 * (1.0 - np.sqrt(np.maximum(1.0 - 4.0 * eta, 0.0))) * s2z)[:, None]
        a2CQ_p = at1 ** 2 * CQ1 + at2 ** 2 * CQ2
        a2CQ_m = at1 ** 2 * CQ1 - at2 ** 2 * CQ2
        phi_qm = (
            -75.0 / (64.0 * eta[:, None]) * a2CQ_p / v
            + ((45.0 / 16.0 * eta[:, None] + 15635.0 / 896.0) * a2CQ_p
               + 2215.0 / 512.0 * delta * a2CQ_m) * v / eta[:, None]
            - 75.0 / (8.0 * eta[:, None]) * a2CQ_p * v2 * np.pi
        )

        pn_phase = -phi_5pn - phi_tidal - phi_qm
        pn_phase = pn_phase - pn_phase[:, :1]   # zero-mean each row
        return pn_phase   # (N, n_phi)

    # ------------------------------------------------------------------
    # Single-waveform convenience method
    # ------------------------------------------------------------------

    def predict(
        self,
        mc: float, q: float, s1z: float, s2z: float,
        lambda1: float, lambda2: float, iota: float, phiref: float,
        frequencies_hz: np.ndarray,
    ) -> np.ndarray:
        """Return h_plus for a single waveform (wraps predict_batch)."""
        p = np.array([[mc, q, s1z, s2z, lambda1, lambda2, iota, phiref]])
        return self.predict_batch(p, frequencies_hz)[0]

    # ------------------------------------------------------------------
    # Vectorised batch prediction
    # ------------------------------------------------------------------

    def predict_batch(
        self,
        params_batch: np.ndarray,    # (N, 8)
        frequencies_hz: np.ndarray,  # (K,)
        n_workers: int = 1,
    ) -> np.ndarray:
        """Predict a batch of h_plus waveforms.

        Strategy
        --------
        * MLP + PCA + PN functions are **vectorised** over all N at once.
        * Spline evaluation at the full 253 k-point output grid is the
          dominant cost; it is parallelised over N waveforms using
          ``concurrent.futures.ThreadPoolExecutor``.  scipy C extensions
          release the GIL, so threads give real parallelism on multi-core
          CPUs.

        Parameters
        ----------
        params_batch : (N, 8) — [mc, q, s1z, s2z, lambda1, lambda2, iota, phiref]
        frequencies_hz : (K,) output frequency grid
        n_workers : int
            Number of threads for parallel spline evaluation.
            Use ``1`` (serial) for safety; set to the number of physical
            CPU cores for best throughput (e.g. ``os.cpu_count()``).

        Returns
        -------
        waveforms : (N, K) complex128
        """
        from concurrent.futures import ThreadPoolExecutor
        from scipy.interpolate import CubicSpline as _CSpline

        N = len(params_batch)

        mc      = params_batch[:, 0]
        q       = params_batch[:, 1]
        s1z     = params_batch[:, 2]
        s2z     = params_batch[:, 3]
        lambda1 = params_batch[:, 4]
        lambda2 = params_batch[:, 5]
        iota    = params_batch[:, 6]
        phiref  = params_batch[:, 7]

        # Total mass and eta for each waveform
        factor     = mc * (1.0 + q) ** 0.2
        m1_arr     = factor * q ** (-0.6)
        m2_arr     = factor * q ** 0.4
        total_mass = m1_arr + m2_arr
        eta        = q / (1.0 + q) ** 2
        scales     = total_mass / self.M_ref   # (N,)

        # ── MLP + PCA: vectorised over N ─────────────────────────────
        mlp_in   = np.column_stack([q, lambda1, lambda2, s1z, s2z])  # (N, 5)
        combined = self._nn_pca_predict_batch(mlp_in)                # (N, n_amp+n_phi)
        amp_residuals = combined[:, :self.n_amp]   # (N, n_amp)
        phi_residuals = combined[:, self.n_amp:]   # (N, n_phi)

        # ── PN functions: vectorised over N (on downsampled grids) ───
        m1_ref = self.M_ref / (1.0 + 1.0 / q)
        m2_ref = self.M_ref / (1.0 + q)
        Lamt   = (16.0 / 13.0) * (
            (m1_ref + 12.0 * m2_ref) * m1_ref ** 4 * lambda1
            + (m2_ref + 12.0 * m1_ref) * m2_ref ** 4 * lambda2
        ) / (m1_ref + m2_ref) ** 5
        etav   = (m1_ref * m2_ref) / (m1_ref + m2_ref) ** 2
        X_etav = np.sqrt(np.maximum(1.0 - 4.0 * etav, 0.0))
        dLamt  = (
            (1690.0 * etav / 1319.0 - 4843.0 / 1319.0)
            * (m1_ref ** 4 * lambda1 - m2_ref ** 4 * lambda2) / (m1_ref + m2_ref) ** 4
            + (6162.0 * X_etav / 1319.0)
            * (m1_ref ** 4 * lambda1 + m2_ref ** 4 * lambda2) / (m1_ref + m2_ref) ** 4
        )

        pn_amp   = self._pn_amp_batch(eta, s1z, s2z, Lamt, dLamt)          # (N, n_amp)
        pn_phase = self._pn_phase_batch(eta, s1z, s2z, lambda1, lambda2,
                                        m1_ref, m2_ref)                     # (N, n_phi)

        amp_ds = np.exp(amp_residuals) * pn_amp    # (N, n_amp)
        phi_ds = phi_residuals + pn_phase           # (N, n_phi)

        # Pre-compute per-waveform scalars needed inside the thread
        pre_mass  = total_mass ** 2 / _AMP_SI_BASE * eta          # (N,)
        pre_plus  = (1.0 + np.cos(iota) ** 2) / 2.0               # (N,)
        phase_rot = np.exp(-1j * phiref)                           # (N,) complex128

        # ── Spline evaluation: parallelised over N via threads ───────
        # scipy CubicSpline releases the GIL, so threads give real
        # parallelism.  Each thread evaluates one waveform's splines.
        out = np.empty((N, len(frequencies_hz)), dtype=np.complex128)

        amp_x = self.amp_freqs_hz
        phi_x = self.phi_freqs_hz

        def _eval_one(i: int) -> None:
            q_freqs = frequencies_hz * scales[i]
            amp_f   = _CSpline(amp_x, amp_ds[i], extrapolate=True)(q_freqs)
            phi_f   = _CSpline(phi_x, phi_ds[i], extrapolate=True)(q_freqs)
            amp_f  *= pre_mass[i]
            h_real  = amp_f * np.cos(phi_f)
            h_imag  = amp_f * np.sin(phi_f)
            out[i]  = (pre_plus[i] * h_real + 1j * pre_plus[i] * h_imag) * phase_rot[i]

        if n_workers <= 1 or N == 1:
            for i in range(N):
                _eval_one(i)
        else:
            with ThreadPoolExecutor(max_workers=n_workers) as ex:
                list(ex.map(_eval_one, range(N)))

        return out


def load_numpy_predictor(model_path: str) -> NumpyPredictor:
    """Load the NumPy waveform predictor from the HDF5 model."""
    return NumpyPredictor(model_path)


# ─────────────────────────────────────────────────────────────────────
# Parameter sampling helpers
# ─────────────────────────────────────────────────────────────────────

def sample_parameters(rng: np.random.Generator, n: int, cfg: ROQConfig) -> np.ndarray:
    """Draw n random parameter sets from the training range.

    Returns shape (n, 8): [mc, q, s1z, s2z, lambda1, lambda2, iota, phiref].
    """
    return np.column_stack([
        rng.uniform(cfg.mc_range[0], cfg.mc_range[1], n),
        rng.uniform(cfg.q_range[0], cfg.q_range[1], n),
        rng.uniform(cfg.s1z_range[0], cfg.s1z_range[1], n),
        rng.uniform(cfg.s2z_range[0], cfg.s2z_range[1], n),
        rng.uniform(cfg.lambda1_range[0], cfg.lambda1_range[1], n),
        rng.uniform(cfg.lambda2_range[0], cfg.lambda2_range[1], n),
        rng.uniform(cfg.iota_range[0], cfg.iota_range[1], n),
        rng.uniform(cfg.phiref_range[0], cfg.phiref_range[1], n),
    ])


def corner_parameters(cfg: ROQConfig) -> np.ndarray:
    """Generate parameters at the 2^6 corners of the training range."""
    corners = []
    ranges = [cfg.mc_range, cfg.q_range, cfg.s1z_range, cfg.s2z_range,
              cfg.lambda1_range, cfg.lambda2_range]
    for bits in range(2 ** len(ranges)):
        point = [r[(bits >> j) & 1] for j, r in enumerate(ranges)]
        point.extend([0.0, 0.0])  # iota=0, phiref=0
        corners.append(point)
    return np.array(corners)


# ─────────────────────────────────────────────────────────────────────
# Linear algebra helpers
# ─────────────────────────────────────────────────────────────────────

def normalise(h: np.ndarray, delta_f: float) -> np.ndarray:
    """Normalise waveforms so <h|h> = 1."""
    norm = np.sqrt(delta_f * np.sum(np.abs(h) ** 2, axis=-1, keepdims=True))
    norm = np.where(norm > 0, norm, 1.0)
    return h / norm


def projection_error(h: np.ndarray, basis: np.ndarray, delta_f: float) -> np.ndarray:
    """Compute ||h - P_V h||^2 for each h in the batch.

    Parameters
    ----------
    h : (n, k) normalised waveforms
    basis : (m, k) orthonormal basis
    delta_f : frequency spacing

    Returns
    -------
    errors : (n,) squared projection errors in [0, 1]
    """
    overlaps = delta_f * (h @ np.conj(basis).T)  # (n, m)
    proj_norm_sq = np.sum(np.abs(overlaps) ** 2, axis=-1)
    return np.maximum(1.0 - proj_norm_sq, 0.0)


def gram_schmidt_add(basis: np.ndarray, new_vec: np.ndarray, delta_f: float) -> np.ndarray:
    """Add a vector to an orthonormal basis via modified Gram-Schmidt (two passes)."""
    v = new_vec.copy()
    for e in basis:
        overlap = delta_f * np.sum(np.conj(e) * v)
        v = v - overlap * e
    for e in basis:
        overlap = delta_f * np.sum(np.conj(e) * v)
        v = v - overlap * e
    norm = np.sqrt(delta_f * np.sum(np.abs(v) ** 2))
    if norm < 1e-15:
        raise ValueError("Vector is linearly dependent on the basis.")
    v = v / norm
    if len(basis) == 0:
        return v.reshape(1, -1)
    return np.vstack([basis, v.reshape(1, -1)])


# ─────────────────────────────────────────────────────────────────────
# Checkpoint helpers
# ─────────────────────────────────────────────────────────────────────

def _save_phase_status(out_dir: Path, kind: str, phase: str, info: dict | None = None):
    status_path = out_dir / f"_status_{kind}.json"
    status: dict = {}
    if status_path.exists():
        with open(status_path) as f:
            status = json.load(f)
    status[phase] = {
        "completed": True,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        **(info or {}),
    }
    with open(status_path, "w") as f:
        json.dump(status, f, indent=2)


def _phase_completed(out_dir: Path, kind: str, phase: str) -> bool:
    status_path = out_dir / f"_status_{kind}.json"
    if not status_path.exists():
        return False
    with open(status_path) as f:
        status = json.load(f)
    return status.get(phase, {}).get("completed", False)


def _save_greedy_step_checkpoint(out_dir: Path, kind: str,
                                 basis: np.ndarray, basis_params: np.ndarray,
                                 errors_history: np.ndarray, step: int):
    """Save step-level checkpoint for the greedy pre-selection phase.

    This allows resuming from the last completed greedy step rather than
    restarting the entire pre-selection (which takes ~1900 s/step).
    """
    ckpt_dir = out_dir / f"_greedy_ckpt_{kind}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    np.save(ckpt_dir / "basis.npy", basis)
    np.save(ckpt_dir / "basis_params.npy", basis_params)
    np.save(ckpt_dir / "errors_history.npy", errors_history)
    meta = {"step": step, "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")}
    with open(ckpt_dir / "meta.json", "w") as f:
        json.dump(meta, f)


def _load_greedy_step_checkpoint(out_dir: Path, kind: str):
    """Load a step-level greedy checkpoint, or return None if none exists."""
    ckpt_dir = out_dir / f"_greedy_ckpt_{kind}"
    meta_path = ckpt_dir / "meta.json"
    if not meta_path.exists():
        return None
    try:
        with open(meta_path) as f:
            meta = json.load(f)
        basis = np.load(ckpt_dir / "basis.npy")
        basis_params = np.load(ckpt_dir / "basis_params.npy")
        errors_history = np.load(ckpt_dir / "errors_history.npy")
        return {
            "basis": basis,
            "basis_params": basis_params,
            "errors_history": errors_history,
            "step": meta["step"],
        }
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────
# Streaming greedy pre-selection with step-level checkpointing
# ─────────────────────────────────────────────────────────────────────

def greedy_basis_streaming(
    all_params: np.ndarray,
    predictor: NumpyPredictor,
    frequencies: np.ndarray,
    delta_f: float,
    tolerance: float,
    max_basis: int = 500,
    quadratic: bool = False,
    batch_size: int = 32,
    proj_batch: int = 500,
    n_workers: int = 1,
    verbose: int = 1,
    out_dir: Path | None = None,
    kind: str = "linear",
    resume: bool = True,
) -> tuple:
    """Streaming greedy basis construction with step-level checkpointing.

    On every greedy step a checkpoint is written immediately after the
    step completes, so a restart skips all completed steps.

    Parameters
    ----------
    all_params : (n, 8) full parameter set
    predictor : NumpyPredictor
    frequencies : frequency grid
    delta_f : frequency spacing
    tolerance : greedy stopping tolerance
    max_basis : maximum basis size
    quadratic : operate on |h|² instead of h
    batch_size : kept for API compatibility (ignored; use n_workers instead)
    proj_batch : waveforms projected at once (controls peak RAM)
    n_workers : threads for parallel spline evaluation inside predict_batch
    verbose : verbosity
    out_dir : directory for step-level checkpoints (None → no checkpointing)
    kind : "linear" or "quadratic" (used in checkpoint filenames)
    resume : if True, load existing step checkpoint to continue

    Returns
    -------
    basis, basis_params, errors_history
    """
    n = len(all_params)
    k = len(frequencies)

    # ── Resume from step checkpoint? ──────────────────────────────────
    start_step = 1
    if resume and out_dir is not None:
        ckpt = _load_greedy_step_checkpoint(out_dir, kind)
        if ckpt is not None:
            basis = ckpt["basis"]
            basis_params_list = list(ckpt["basis_params"])
            errors_history = list(ckpt["errors_history"])
            start_step = ckpt["step"] + 1
            if verbose:
                print(f"    Resuming greedy pre-selection from step {start_step} "
                      f"(basis size = {len(basis)})")
        else:
            basis = None
            basis_params_list = []
            errors_history = []
    else:
        basis = None
        basis_params_list = []
        errors_history = []

    # ── Seed with first waveform if no checkpoint ─────────────────────
    if basis is None:
        first_wf = predictor.predict_batch(all_params[:1], frequencies,
                                           n_workers=n_workers)
        if quadratic:
            first_wf = np.abs(first_wf) ** 2
        first_wf = normalise(first_wf, delta_f)
        basis = np.empty((0, k), dtype=first_wf.dtype)
        basis = gram_schmidt_add(basis, first_wf[0], delta_f)
        basis_params_list = [all_params[0]]

    if verbose:
        print(f"  Streaming greedy: {n} params, tolerance={tolerance:.1e}, "
              f"proj_batch={proj_batch}")

    for step in range(start_step, max_basis):
        t_step = time.time()
        global_max_err = 0.0
        global_max_wf = None
        global_max_p = None

        # Scan all training params in batches
        for bstart in range(0, n, proj_batch):
            bend = min(bstart + proj_batch, n)
            chunk_params = all_params[bstart:bend]

            chunk_wf = predictor.predict_batch(chunk_params, frequencies,
                                               n_workers=n_workers)
            if quadratic:
                chunk_wf = np.abs(chunk_wf) ** 2
            chunk_wf = normalise(chunk_wf, delta_f)

            errs = projection_error(chunk_wf, basis, delta_f)
            local_max_idx = int(errs.argmax())
            local_max_err = float(errs[local_max_idx])

            if local_max_err > global_max_err:
                global_max_err = local_max_err
                global_max_wf = chunk_wf[local_max_idx].copy()
                global_max_p = chunk_params[local_max_idx].copy()

            del chunk_wf, errs
            gc.collect()

        errors_history.append(global_max_err)
        elapsed = time.time() - t_step

        if verbose:
            print(f"    Step {step:4d}: basis={len(basis)}, "
                  f"max_err={global_max_err:.6e}  ({elapsed:.1f}s)")

        # Save step checkpoint immediately
        if out_dir is not None:
            basis_params_arr = np.array(basis_params_list)
            _save_greedy_step_checkpoint(
                out_dir, kind, basis, basis_params_arr,
                np.array(errors_history), step,
            )

        if global_max_err < tolerance:
            if verbose:
                print(f"  ✓ Converged: {len(basis)} basis vectors "
                      f"(error {global_max_err:.2e} < {tolerance:.1e})")
            break

        basis = gram_schmidt_add(basis, global_max_wf, delta_f)
        basis_params_list.append(global_max_p)
    else:
        if verbose:
            print(f"  ⚠ Reached max basis {max_basis} "
                  f"(error {errors_history[-1] if errors_history else float('nan'):.2e})")

    return basis, np.array(basis_params_list), np.array(errors_history)


# ─────────────────────────────────────────────────────────────────────
# Enrichment (streaming)
# ─────────────────────────────────────────────────────────────────────

def enrich_basis(
    basis: np.ndarray,
    basis_params: np.ndarray,
    predictor: NumpyPredictor,
    frequencies: np.ndarray,
    cfg: ROQConfig,
    tolerance: float,
    quadratic: bool = False,
    rng: Optional[np.random.Generator] = None,
    n_workers: int = 1,
) -> tuple:
    """Iteratively enrich basis with streamed training waveforms.

    Parameters
    ----------
    basis : (m, k) current orthonormal basis
    basis_params : (m, p) parameters of current basis elements
    predictor : NumpyPredictor
    frequencies : frequency grid
    cfg : ROQConfig
    tolerance : absolute greedy tolerance
    quadratic : build quadratic basis (|h|²)
    rng : random generator

    Returns
    -------
    basis, basis_params
    """
    if rng is None:
        rng = np.random.default_rng(cfg.random_seed + 1000)

    delta_f = cfg.delta_f
    kind = "quadratic" if quadratic else "linear"
    proj_bs = cfg.projection_batch_size

    rel_tols = list(cfg.training_set_rel_tol)
    while len(rel_tols) < len(cfg.training_set_sizes):
        rel_tols.append(1.0)

    for cycle_idx, n_train in enumerate(cfg.training_set_sizes):
        t0 = time.time()
        cycle_tol = tolerance * rel_tols[cycle_idx]
        if cfg.verbose:
            print(f"\n  Enrichment cycle {cycle_idx + 1}/"
                  f"{len(cfg.training_set_sizes)}: "
                  f"{n_train} waveforms ({kind}), "
                  f"effective_tol={cycle_tol:.2e} "
                  f"(rel={rel_tols[cycle_idx]:.1f})")

        train_params = sample_parameters(rng, n_train, cfg)
        n_scanned = 0
        n_added = 0
        max_err_cycle = 0.0

        for bstart in range(0, n_train, proj_bs):
            bend = min(bstart + proj_bs, n_train)
            chunk_params = train_params[bstart:bend]

            chunk_wf = predictor.predict_batch(chunk_params, frequencies,
                                               n_workers=n_workers)
            if quadratic:
                chunk_wf = np.abs(chunk_wf) ** 2
            chunk_wf = normalise(chunk_wf, delta_f)

            errs = projection_error(chunk_wf, basis, delta_f)
            local_max = float(errs.max())
            if local_max > max_err_cycle:
                max_err_cycle = local_max

            outlier_idx = np.where(errs > cycle_tol)[0]
            if len(outlier_idx) > 0:
                order = outlier_idx[np.argsort(errs[outlier_idx])[::-1]]
                for idx in order:
                    h = chunk_wf[idx]
                    err = projection_error(h.reshape(1, -1), basis, delta_f)[0]
                    if err < cycle_tol:
                        continue
                    basis = gram_schmidt_add(basis, h, delta_f)
                    basis_params = np.vstack([basis_params, chunk_params[idx]])
                    n_added += 1

            n_scanned += len(chunk_params)
            del chunk_wf, errs
            gc.collect()

            if cfg.verbose and n_scanned % (proj_bs * 10) == 0:
                print(f"      scanned {n_scanned}/{n_train}, "
                      f"added {n_added}, basis={len(basis)}")

        elapsed = time.time() - t0
        if cfg.verbose:
            print(f"    Cycle done: scanned={n_scanned}, added={n_added}, "
                  f"basis={len(basis)}, max_err={max_err_cycle:.2e}, "
                  f"time={elapsed:.1f}s")

    return basis, basis_params


# ─────────────────────────────────────────────────────────────────────
# Empirical Interpolation Method (EIM)
# ─────────────────────────────────────────────────────────────────────

def empirical_interpolation(
    basis: np.ndarray,
    delta_f: float,
    verbose: int = 1,
) -> tuple:
    """Compute empirical interpolation nodes and interpolant matrix.

    Parameters
    ----------
    basis : (m, k) orthonormal reduced basis
    delta_f : frequency spacing
    verbose : verbosity

    Returns
    -------
    nodes : (m,) int — indices of the empirical interpolation nodes
    interpolant : (k, m) — matrix B such that h_approx = B @ h[nodes]
    """
    m, k = basis.shape
    if m == 0:
        raise ValueError("Basis is empty.")

    nodes = np.empty(m, dtype=int)

    if verbose:
        print(f"\n  EIM: computing empirical nodes for {m} basis vectors "
              f"on {k} frequency points")

    nodes[0] = np.argmax(np.abs(basis[0]))

    for j in range(1, m):
        V_mat = basis[:j, :][:, nodes[:j]].T
        rhs = basis[j, nodes[:j]]
        c = np.linalg.solve(V_mat, rhs)
        residual = basis[j] - c @ basis[:j]
        nodes[j] = np.argmax(np.abs(residual))

        if verbose and (j + 1) % 50 == 0:
            print(f"    EIM step {j+1}/{m}: node at freq index {nodes[j]}")

    V_full = basis[:, nodes].T
    V_inv = np.linalg.inv(V_full)
    interpolant = basis.T @ V_inv

    if verbose:
        print(f"  ✓ EIM complete: {m} nodes selected")
        print(f"    Node frequency range: [{nodes.min()}, {nodes.max()}] "
              f"(out of {k} points)")

    return nodes, interpolant


# ─────────────────────────────────────────────────────────────────────
# Full build pipeline
# ─────────────────────────────────────────────────────────────────────

def build_roq_basis(
    cfg: ROQConfig,
    kind: str = "linear",
    resume: bool = True,
) -> dict:
    """Full pipeline: pre-selection → enrichment → EIM.

    Every phase and every greedy step is checkpointed.  Restart the
    script at any time and it will pick up where it left off.

    Parameters
    ----------
    cfg : ROQConfig
    kind : "linear" or "quadratic"
    resume : skip completed phases/steps

    Returns
    -------
    dict with keys: basis, basis_params, nodes, interpolant,
                    frequencies, empirical_frequencies, errors_history
    """
    quadratic = kind == "quadratic"
    tolerance = cfg.tolerance_qua if quadratic else cfg.tolerance_lin
    n_pre = cfg.n_pre_basis_qua if quadratic else cfg.n_pre_basis_lin

    frequencies = cfg.frequencies
    delta_f = cfg.delta_f

    out_dir = Path(cfg.output_dir) / "ROQ_data" / kind
    out_dir.mkdir(parents=True, exist_ok=True)

    if cfg.verbose:
        print(f"\n{'='*65}")
        print(f"  Building {kind.upper()} ROQ basis  (NumPy)")
        print(f"{'='*65}")
        print(f"  Frequency range: [{cfg.f_min}, {cfg.f_max}] Hz")
        print(f"  Frequency points: {cfg.n_freq}")
        print(f"  Tolerance: {tolerance:.1e}")
        print(f"  Pre-basis size target: {n_pre}")
        print(f"  Resume mode: {'ON' if resume else 'OFF'}")

    t_start = time.time()

    # ── Already fully built? ─────────────────────────────────────────
    if resume and _phase_completed(out_dir, kind, "eim"):
        if cfg.verbose:
            print(f"\n  ✓ {kind.upper()} basis already fully built — loading from disk")
        basis = np.load(out_dir / f"basis_{kind}.npy")
        basis_params = np.load(out_dir / f"basis_waveform_params_{kind}.npy")
        nodes = np.load(out_dir / f"empirical_nodes_{kind}.npy")
        interpolant = np.load(out_dir / f"basis_interpolant_{kind}.npy")
        empirical_freqs = np.load(out_dir / f"empirical_frequencies_{kind}.npy")
        err_path = out_dir / f"preselection_{kind}_basis_residual_modula.npy"
        errors_history = np.load(err_path) if err_path.exists() else np.array([])
        return {
            "basis": basis, "basis_params": basis_params,
            "nodes": nodes, "interpolant": interpolant,
            "frequencies": frequencies, "empirical_frequencies": empirical_freqs,
            "errors_history": errors_history,
        }

    # ── Load model ───────────────────────────────────────────────────
    if cfg.verbose:
        print(f"\n  Loading waveform model from {cfg.model_path}...")
    t0 = time.time()
    predictor = load_numpy_predictor(cfg.model_path)
    if cfg.verbose:
        print(f"  ✓ Model loaded in {time.time() - t0:.1f}s")

    # ── Phase 1: Pre-selection (streaming greedy with step checkpoints) ──
    if resume and _phase_completed(out_dir, kind, "preselection"):
        if cfg.verbose:
            print(f"\n  Phase 1: Pre-selection — RESUMING from checkpoint")
        basis = np.load(out_dir / f"preselection_{kind}_basis.npy")
        basis_params = np.load(out_dir / f"preselection_{kind}_basis_waveform_params.npy")
        err_path = out_dir / f"preselection_{kind}_basis_residual_modula.npy"
        errors_history = np.load(err_path) if err_path.exists() else np.array([])
        if cfg.verbose:
            print(f"    Loaded pre-selection basis: {len(basis)} vectors")
    else:
        if cfg.verbose:
            print(f"\n  Phase 1: Pre-selection (streaming, with step checkpoints)")

        rng = np.random.default_rng(cfg.random_seed)
        corner_p = corner_parameters(cfg)
        n_random_seed = max(0, cfg.n_pre_basis_search_iter * n_pre - len(corner_p))
        random_seed_p = sample_parameters(rng, n_random_seed, cfg)
        pre_params = np.vstack([corner_p, random_seed_p])

        if cfg.verbose:
            print(f"    Pre-selection pool: {len(pre_params)} parameter sets "
                  f"({len(corner_p)} corners + {n_random_seed} random)")

        basis, basis_params, errors_history = greedy_basis_streaming(
            pre_params, predictor, frequencies, delta_f, tolerance,
            max_basis=n_pre,
            quadratic=quadratic,
            batch_size=cfg.waveform_batch_size,
            proj_batch=cfg.projection_batch_size,
            n_workers=cfg.n_workers,
            verbose=cfg.verbose,
            out_dir=out_dir,
            kind=kind,
            resume=resume,
        )

        np.save(out_dir / f"preselection_{kind}_basis.npy", basis)
        np.save(out_dir / f"preselection_{kind}_basis_waveform_params.npy", basis_params)
        np.save(out_dir / f"preselection_{kind}_basis_residual_modula.npy", errors_history)
        _save_phase_status(out_dir, kind, "preselection",
                           {"basis_size": int(len(basis))})
        if cfg.verbose:
            print(f"    Saved pre-selection checkpoint to {out_dir}")

    # ── Phase 2: Enrichment (streaming) ─────────────────────────────
    if resume and _phase_completed(out_dir, kind, "enrichment"):
        if cfg.verbose:
            print(f"\n  Phase 2: Enrichment — RESUMING from checkpoint")
        basis = np.load(out_dir / f"basis_{kind}.npy")
        basis_params = np.load(out_dir / f"basis_waveform_params_{kind}.npy")
        if cfg.verbose:
            print(f"    Loaded enriched basis: {len(basis)} vectors")
    else:
        if cfg.verbose:
            print(f"\n  Phase 2: Enrichment (streaming)")
        rng_enrich = np.random.default_rng(cfg.random_seed + 2000)
        basis, basis_params = enrich_basis(
            basis, basis_params, predictor, frequencies, cfg,
            tolerance=tolerance, quadratic=quadratic, rng=rng_enrich,
            n_workers=cfg.n_workers,
        )
        np.save(out_dir / f"basis_{kind}.npy", basis)
        np.save(out_dir / f"basis_waveform_params_{kind}.npy", basis_params)
        _save_phase_status(out_dir, kind, "enrichment",
                           {"basis_size": int(len(basis))})
        if cfg.verbose:
            print(f"    Saved enriched basis to {out_dir}")

    # ── Phase 3: EIM ─────────────────────────────────────────────────
    if cfg.verbose:
        print(f"\n  Phase 3: Empirical Interpolation")

    nodes, interpolant = empirical_interpolation(basis, delta_f, verbose=cfg.verbose)
    empirical_freqs = frequencies[nodes]

    np.save(out_dir / f"empirical_nodes_{kind}.npy", nodes)
    np.save(out_dir / f"empirical_frequencies_{kind}.npy", empirical_freqs)
    np.save(out_dir / f"basis_interpolant_{kind}.npy", interpolant)
    _save_phase_status(out_dir, kind, "eim", {"n_nodes": int(len(nodes))})

    t_total = time.time() - t_start
    if cfg.verbose:
        print(f"\n  ✓ {kind.upper()} basis complete:")
        print(f"    Basis size         : {len(basis)}")
        print(f"    Frequency points   : {cfg.n_freq}")
        print(f"    Empirical nodes    : {len(nodes)}")
        print(f"    Compression ratio  : {cfg.n_freq / len(nodes):.0f}x")
        print(f"    Total time         : {t_total:.1f}s")

    return {
        "basis": basis, "basis_params": basis_params,
        "nodes": nodes, "interpolant": interpolant,
        "frequencies": frequencies, "empirical_frequencies": empirical_freqs,
        "errors_history": errors_history,
    }


# ─────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

    # Config file: first CLI arg, else sibling config_roq_basis_2.ini
    if len(sys.argv) > 1:
        CONFIG_FILE = sys.argv[1]
    else:
        CONFIG_FILE = os.path.join(SCRIPT_DIR, "config_roq_basis_2.ini")

    OUT_DIR = os.path.join(SCRIPT_DIR, "ROQ_basis_2")

    cfg = ROQConfig.from_ini(CONFIG_FILE)
    cfg.output_dir = OUT_DIR
    os.makedirs(OUT_DIR, exist_ok=True)

    print(f"\n{'='*65}")
    print("Building ROQ_basis_2 for mlgw_bns_jax  [NumPy backend]")
    print(f"  Config           : {CONFIG_FILE}")
    print(f"  Frequency range  : [{cfg.f_min}, {cfg.f_max}] Hz")
    print(f"  Segment length   : {cfg.seglen} s  (df = {cfg.delta_f:.4f} Hz)")
    print(f"  Frequency points : {cfg.n_freq}")
    print(f"  Pre-basis (lin)  : {cfg.n_pre_basis_lin}  (Npre)")
    print(f"  Pre-basis (qua)  : {cfg.n_pre_basis_qua}  (Npre)")
    print(f"  Search iter/step : {cfg.n_pre_basis_search_iter}  (Nstep)")
    print(f"  Enrichment cycles: {len(cfg.training_set_sizes)}")
    for i, (n, rt) in enumerate(zip(cfg.training_set_sizes, cfg.training_set_rel_tol), 1):
        print(f"    Cycle {i}: {n:>7,} waveforms, rel_tol={rt:.1f}")
    print(f"  Tolerance (lin)  : {cfg.tolerance_lin:.1e}")
    print(f"  Tolerance (qua)  : {cfg.tolerance_qua:.1e}")
    print(f"  Output directory : {OUT_DIR}")
    print(f"  Resume mode      : ON (step-level checkpoints enabled)")
    print(f"{'='*65}\n")

    t_global = time.time()

    # ── Phase 1: LINEAR basis ─────────────────────────────────────────
    print("=" * 65)
    print("  Building LINEAR ROQ basis")
    print("=" * 65)

    results_lin = build_roq_basis(cfg, kind="linear", resume=True)
    n_lin = len(results_lin["nodes"])
    print(f"\n✓ Linear basis: {n_lin} empirical nodes  "
          f"({cfg.n_freq / n_lin:.0f}x compression)")

    del results_lin["basis"]
    gc.collect()

    # ── Phase 2: QUADRATIC basis ──────────────────────────────────────
    print("\n" + "=" * 65)
    print("  Building QUADRATIC ROQ basis")
    print("=" * 65)

    results_qua = build_roq_basis(cfg, kind="quadratic", resume=True)
    n_qua = len(results_qua["nodes"])
    print(f"\n✓ Quadratic basis: {n_qua} empirical nodes  "
          f"({cfg.n_freq / n_qua:.0f}x compression)")

    # ── Summary ───────────────────────────────────────────────────────
    t_total = time.time() - t_global

    print(f"\n{'='*65}")
    print("ROQ_basis_2 construction complete!")
    print(f"  Full frequency grid : {cfg.n_freq} points")
    print(f"  Linear basis        : {n_lin} nodes  "
          f"({cfg.n_freq / n_lin:.0f}x reduction)")
    print(f"  Quadratic basis     : {n_qua} nodes  "
          f"({cfg.n_freq / n_qua:.0f}x reduction)")
    print(f"  Total wall time     : {t_total:.1f}s  ({t_total / 3600:.2f}h)")
    print(f"  Results saved in    : {OUT_DIR}/ROQ_data/")
    print(f"{'='*65}")
