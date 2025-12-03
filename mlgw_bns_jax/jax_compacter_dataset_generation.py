"""Functionality for the generation of a training dataset.
"""
#from __future__ import annotations

#from abc import ABC, abstractmethod
#from collections.abc import Iterator
#from dataclasses import dataclass
#from functools import lru_cache
#from typing import Any, Callable, ClassVar, Optional, Type, Union

#import h5py
import jax
import jax.numpy as np
import numpy
from numpy.random import default_rng
#from tqdm import tqdm  # type: ignore







SUN_MASS_SECONDS: float = 4.92549094830932e-6  # M_sun * G / c**3
EULER_GAMMA = 0.57721566490153286060
TF2_BASE: float = 3.668693487138444e-19
# ( Msun * G / c**3)**(5/6) * Hz**(-7/6) * c / Mpc / s
AMP_SI_BASE: float = 4.2425873413901263e24
# Mpc / Msun**2 / Hz

#saved default data (CHANGE THAT FOR A PROPER IMPORTING FROM THE CODE DATA)
TOTAL_MASS_SAVED = 2.8

mass_range_saved = (2.0, 4.0)
q_range_saved = (1.0, 3.0)
lambda1_range_saved = (5.0, 5000.0)
lambda2_range_saved  = (5.0, 5000.0)
chi1_range_saved  = (-0.5, 0.5)
chi2_range_saved  = (-0.5, 0.5)


#data set features 
initial_frequency_hz_saved = 5.0
srate_hz_saved = 4096.0

#saved mass sum seconds
mass_sum_seconds_saved= TOTAL_MASS_SAVED * SUN_MASS_SECONDS



@jax.jit
def expand_frequency_range_jax(
    initial_frequency: float,
    final_frequency: float,
    mass_range: tuple[float, float],
    reference_mass: float,
) -> tuple[float, float]:
    r"""Widen the frequency range to account for the
    different masses the user requires.

    Parameters
    ----------
    initial_frequency : float
        Lower bound for the frequency.
        Typically in Hz, but this function just requires it
        to be consistent with the other parameters.
    final_frequency : float
        Upper bound for the frequency.
        It can also be given as the time-domain
        signal rate :math:`r = 1 / \Delta t`, which is
        twice che maximum frequency because of the Nyquist bound.

        Since all this function does is multiply it by a certain factor,
        the formulations can be exchanged.
    mass_range : tuple[float, float]
        Range of allowed masses, in the same unit as the
        reference mass (typically, solar masses).
    reference_mass : float
        Reference mass the model uses to convert frequencies
        to the dimensionless :math:`Mf`.

    Returns
    -------
    tuple[float, float]
        New lower and upper bounds for the frequency range.
    """

    m_min, m_max = mass_range
    #assert m_min <= m_max

    return (
        initial_frequency * (m_min / reference_mass),
        final_frequency * (m_max / reference_mass),
    )

### defining effective initial and srate frequency

(effective_initial_frequency_hz_saved, effective_srate_hz_saved,) = expand_frequency_range_jax(initial_frequency_hz_saved,srate_hz_saved,mass_range_saved,TOTAL_MASS_SAVED,)

### FUNCTIONS

###IMPUT DATA
#    mass_ratio: float
#    lambda_1: float
#    lambda_2: float
#    chi_1: float
#    chi_2: float
#    number_of_parameters: ClassVar[int] = 5

@jax.jit
def eta(mass_ratio):
        r"""Symmetric mass ratio of the binary.

        It is defined as :math:`\eta = \mu / M`, where
        :math:`\mu  = (1 / m_1 + 1/ m_2)^{-1}`
        and :math:`M = m_1 + m_2`.

        It can also be expressed as
        :math:`\eta = m_1 m_2 / (m_1 + m_2)^2 = q / (1+q)^2`,
        where :math:`q = m_1 / m_2` is the mass ratio.

        It is also sometimes denoted as :math:`\nu`.
        It goes from 0 in the test-mass limit (one mass vanishing)
        to :math:`1/4` in the equal-mass limit.
        """
        return mass_ratio / (1.0 + mass_ratio) ** 2

@jax.jit
def m_1(mass_ratio):
        """Mass of the heavier star in the system, in solar masses."""
        return TOTAL_MASS_SAVED / (1 + 1 / mass_ratio)

@jax.jit
def m_2(mass_ratio):
        """Mass of the lighter star in the system, in solar masses."""
        return TOTAL_MASS_SAVED / (1 + mass_ratio)


@jax.jit
def compute_lambda_tilde(m1, m2, l1, l2):
    """Compute Lambda Tilde from masses and tides components
    --------
    m1 = primary mass component [solar masses]
    m2 = secondary mass component [solar masses]
    l1 = primary tidal component [dimensionless]
    l2 = secondary tidal component [dimensionless]
    """
    M = m1 + m2
    m1_4 = m1 ** 4.0
    m2_4 = m2 ** 4.0
    M5 = M ** 5.0
    comb1 = m1 + 12.0 * m2
    comb2 = m2 + 12.0 * m1
    return (16.0 / 13.0) * (comb1 * m1_4 * l1 + comb2 * m2_4 * l2) / M5


@jax.jit
def compute_delta_lambda(m1, m2, l1, l2):
    """Compute delta Lambda Tilde from masses and tides components
    --------
    m1 = primary mass component [solar masses]
    m2 = secondary mass component [solar masses]
    l1 = primary tidal component [dimensionless]
    l2 = secondary tidal component [dimensionless]
    """
    M = m1 + m2
    q = m1 / m2
    eta = q / ((1.0 + q) * (1.0 + q))
    X = np.sqrt(1.0 - 4.0 * eta)
    m1_4 = m1 ** 4.0
    m2_4 = m2 ** 4.0
    M4 = M ** 4.0
    comb1 = (1690.0 * eta / 1319.0 - 4843.0 / 1319.0) * (m1_4 * l1 - m2_4 * l2) / M4
    comb2 = (6162.0 * X / 1319.0) * (m1_4 * l1 + m2_4 * l2) / M4
    return comb1 + comb2

@jax.jit
def lambdatilde(mass_ratio,lambda_1,lambda_2):
        r"""Symmetrized tidal deformability parameter :math:`\widetilde\Lambda`,
        which gives the largest contribution to the waveform phase.
        For the precise definition see equation 5 of `this paper <http://arxiv.org/abs/1805.11579>`__."""
        m1=m_1(mass_ratio)
        m2=m_2(mass_ratio)
        return compute_lambda_tilde(m1, m2, lambda_1, lambda_2)

@jax.jit
def dlambda(mass_ratio,lambda_1,lambda_2):
        r"""Antisymmetrized tidal deformability parameter :math:`\delta \widetilde\Lambda`,
        which gives the next-to-largest contribution to the waveform phase.
        For the precise definition see equation 27 of `this paper <http://arxiv.org/abs/2102.00017>`__."""
        m1=m_1(mass_ratio)
        m2=m_2(mass_ratio)
        return compute_delta_lambda(m1, m2, lambda_1, lambda_2)




@jax.jit
def taylor_f2(frequencies, mass_ratio, chi_1, chi_2, lambda_1, lambda_2):
        """Parameter dictionary in a format compatible with
        the custom implemnentation of TaylorF2 implemented within ``mlgw_bns``.

        Parameters
        ----------
        frequencies : np.ndarray
                The frequencies where to compute the
                waveform, to be given in natural units
        """

        return {
            "f": frequencies / mass_sum_seconds_saved,
            "q": mass_ratio,
            "s1x": 0,
            "s1y": 0,
            "s1z": chi_1,
            "s2y": 0,
            "s2x": 0,
            "s2z": chi_2,
            "lambda1": lambda_1,
            "lambda2": lambda_2,
            "f_min": effective_initial_frequency_hz_saved,
            "phi_ref": 0,
            "phaseorder": 11,
            "tidalorder": 15,
            "usenewtides": 1,
            "usequadrupolemonopole": 1,
            "mtot": TOTAL_MASS_SAVED,
            "s1x": 0,
            "s1y": 0,
            "s2x": 0,
            "s2y": 0,
            "Deff": 1.0,
            "phiRef": 0.0,
            "timeShift": 0.0,
            "iota": 0.0,
        }

@jax.jit
def taylor_f2_prefactor(etaa):
        """Prefactor by which to multiply the waveform
        generated by TaylorF2.

        Parameters
        ----------
        eta : float
                Mass ratio of the binary
        """
        return TF2_BASE * AMP_SI_BASE / etaa / TOTAL_MASS_SAVED ** 2

@jax.jit
def array_parameters(mass_ratio, lambda_1, lambda_2, chi_1, chi_2):
        r"""Represent the parameters as a numpy array.

        Returns
        -------
        np.ndarray
            Array representation of the parameters, specifically
            :math:`[q, \Lambda_1, \Lambda_2, \chi_1, \chi_2]`.
        """
        return np.array(
            [mass_ratio, lambda_1, lambda_2, chi_1, chi_2]
        )


@jax.jit
def mlgw_bns_prefactor_jax(eta, total_mass):
        """Prefactor by which to multiply the waveform
        generated by `mlgw_bns`.

        Parameters
        ----------
        eta : float
                Mass ratio of the binary
        total_mass : Optional[float]
                Total mass of the binary.
                Defaults to None, in which case the `total_mass`
                attribute of the Dataset will be used.
        """
        return total_mass ** 2 / AMP_SI_BASE * eta



