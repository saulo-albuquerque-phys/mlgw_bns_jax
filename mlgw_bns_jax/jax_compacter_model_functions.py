import jax
import jax.numpy as np


@jax.jit
def combine_amp_phase_jax(
    amp: np.ndarray, phase: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    r"""Combine amplitude and phase arrays into a Cartesian waveform,
    according to
    :math:`h = A e^{i \phi}`.

    This function is separated out just so that it can be decorated with ``@njit``.

    Parameters
    ----------
    amp : np.ndarray
    phase : np.ndarray

    Returns
    -------
    tuple[np.ndarray, np.ndarray]:
        Real and imaginary parts of the waveform, respectively.
    """
    return (amp * np.cos(phase), amp * np.sin(phase))


@jax.jit
def combine_residuals_amp_jax(amp: np.ndarray, amp_pn: np.ndarray) -> np.ndarray:
    r"""Combine amplitude residuals with their Post-Newtonian counterparts,
    according to
    :math:`A = A_{PN} e^{\Delta A}`.

    This function is separated out just so that it can be decorated with ``@njit``.

    Parameters
    ----------
    amp : np.ndarray
    amp_pn : np.ndarray

    Returns
    -------
    np.ndarray
    """
    return amp_pn * np.exp(amp)


@jax.jit
def combine_residuals_phi_jax(phi: np.ndarray, phi_pn: np.ndarray) -> np.ndarray:
    r"""Combine amplitude residuals with their Post-Newtonian counterparts,
    according to
    :math:`\phi = \phi_{PN} + \Delta \phi`.

    This function is separated out just so that it can be decorated with ``@njit``.

    Parameters
    ----------
    phi : np.ndarray
    phi_pn : np.ndarray

    Returns
    -------
    np.ndarray
    """
    return phi_pn + phi


@jax.jit
def compute_polarizations_jax(
    waveform_real,
    waveform_imag,
    pre_plus,
    pre_cross,
):
    """Compute the two polarizations of the waveform,
    assuming they are the same but for a differerent prefactor
    (which is the case for compact binary coalescences).

    This function is separated out so that it can be decorated with
    `numba.njit <https://numba.pydata.org/numba-doc/latest/reference/jit-compilation.html>`_
    which allows it to be compiled --- this can speed up the computation somewhat.

    Parameters
    ----------
    waveform_real : np.ndarray
        Real part of the cartesian complex-valued waveform.
    waveform_imag : np.ndarray
        Imaginary part of the cartesian complex-valued waveform.
    pre_plus : complex
        Real-valued prefactor for the plus polarization of the waveform.
    pre_cross : complex
        Real-valued prefactor for the cross polarization of the waveform.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Plus and cross polarizations: complex-valued arrays.
    """

    hp = pre_plus * waveform_real + 1j * pre_plus * waveform_imag
    hc = pre_cross * waveform_imag - 1j * pre_cross * waveform_real

    return hp, hc