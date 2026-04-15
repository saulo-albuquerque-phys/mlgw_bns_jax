"""
JenpyROQ waveform wrapper using the original mlgw_bns model (NumPy/SciPy).

This avoids JAX entirely, eliminating LLVM JIT compilation memory issues
on low-RAM machines.  The waveforms are identical to the JAX version.

Usage:
    import mlgw_bns_roq_wrapper  # auto-registers on import

Then set ``approximant = mlgw-bns-jax`` in the JenpyROQ config file.
"""

import numpy as np

# NumPy 2.0 removed VisibleDeprecationWarning; JenpyROQ still references it
if not hasattr(np, "VisibleDeprecationWarning"):
    np.VisibleDeprecationWarning = FutureWarning

from mlgw_bns import Model
from mlgw_bns.model import ParametersWithExtrinsic
from JenpyROQ.waveform_wrappers import WfWrapper, __non_lal_approx_names__

# Load model once
_model = Model.default()


class WfMLGWBNS:
    """JenpyROQ wrapper for the original mlgw_bns model (NumPy)."""

    def __init__(self, approximant, additional_waveform_params=None):
        self.approximant = approximant
        self.waveform_params = additional_waveform_params or {}

    def generate_waveform(self, p, deltaF, f_min, f_max, distance):
        """Generate (h_plus, h_cross) on a uniform frequency grid.

        Parameters
        ----------
        p : dict
            Binary parameters with keys: m1, m2, s1z, s2z, lambda1, lambda2,
            iota, phiref.
        deltaF : float
            Frequency spacing (= 1/seglen).
        f_min, f_max : float
            Frequency bounds [Hz].
        distance : float
            Luminosity distance [Mpc].

        Returns
        -------
        hp, hc : np.ndarray (complex128)
            Plus and cross polarisations.
        """
        m1, m2 = p["m1"], p["m2"]
        q = m1 / m2

        s1z = p.get("s1z", 0.0)
        s2z = p.get("s2z", 0.0)
        lambda1 = p.get("lambda1", 0.0)
        lambda2 = p.get("lambda2", 0.0)
        inclination = p.get("iota", 0.0)
        phiref = p.get("phiref", 0.0)

        # Enforce q >= 1 convention
        if q < 1.0:
            m1, m2 = m2, m1
            q = 1.0 / q
            s1z, s2z = s2z, s1z
            lambda1, lambda2 = lambda2, lambda1

        total_mass = m1 + m2
        frequencies = np.arange(f_min, f_max + deltaF, deltaF)

        params = ParametersWithExtrinsic(
            mass_ratio=q,
            lambda_1=lambda1,
            lambda_2=lambda2,
            chi_1=s1z,
            chi_2=s2z,
            distance_mpc=float(distance),
            inclination=inclination,
            total_mass=total_mass,
            reference_phase=phiref,
        )

        hp, hc = _model.predict(frequencies, params)

        return hp.astype(np.complex128), hc.astype(np.complex128)


# Register the wrapper
WfWrapper["mlgw-bns-jax"] = WfMLGWBNS
__non_lal_approx_names__.append("mlgw-bns-jax")
