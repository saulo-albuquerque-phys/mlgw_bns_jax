"""
JenpyROQ waveform wrapper for the mlgw_bns_jax model.

This wrapper allows JenpyROQ to generate waveforms using our JAX-based
mlgw_bns surrogate model, so that a Reduced Order Quadrature (ROQ) basis
can be constructed for accelerated parameter estimation.

Usage:
    After importing this module, register the wrapper before running JenpyROQ:

        import mlgw_bns_jax_roq_wrapper  # auto-registers on import

    Then set ``approximant = mlgw-bns-jax`` in the JenpyROQ config file.
"""

import os
import numpy as np

# NumPy 2.0 removed VisibleDeprecationWarning; JenpyROQ still references it
if not hasattr(np, "VisibleDeprecationWarning"):
    np.VisibleDeprecationWarning = FutureWarning

# Force CPU for deterministic basis construction
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

from jax_import_n_predict import load_predict
from JenpyROQ.waveform_wrappers import WfWrapper, __non_lal_approx_names__


# Global model — loaded once
_MODEL_PATH = os.path.join(os.path.dirname(__file__), "mlgw_bns_jax_model.h5")
_mlgw_predict = load_predict(_MODEL_PATH)


class WfMLGWJAX:
    """JenpyROQ wrapper for the mlgw_bns_jax waveform model."""

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

        mlgw_params = jnp.array([q, lambda1, lambda2, s1z, s2z])
        jnp_total_mass = jnp.array(total_mass)
        jnp_distance = jnp.array(float(distance))
        jnp_inclination = jnp.array(inclination)

        # Evaluate in frequency chunks to avoid LLVM OOM during JIT
        # compilation.  JAX caches the compiled code per input shape,
        # so all chunks use the same size (last one is padded).
        CHUNK = 32768
        n_freq = len(frequencies)

        if n_freq <= CHUNK:
            hp_jax, hc_jax = _mlgw_predict(
                mlgw_params, jnp.array(frequencies),
                total_mass=jnp_total_mass,
                distance_mpc=jnp_distance,
                inclination=jnp_inclination,
            )
            hp = np.array(hp_jax, dtype=np.complex128)
            hc = np.array(hc_jax, dtype=np.complex128)
        else:
            hp_parts, hc_parts = [], []
            for i in range(0, n_freq, CHUNK):
                chunk = frequencies[i : i + CHUNK]
                # Pad last chunk so JAX reuses the same compiled code
                pad = CHUNK - len(chunk)
                if pad > 0:
                    chunk = np.concatenate([chunk, np.full(pad, chunk[-1])])
                hp_c, hc_c = _mlgw_predict(
                    mlgw_params, jnp.array(chunk),
                    total_mass=jnp_total_mass,
                    distance_mpc=jnp_distance,
                    inclination=jnp_inclination,
                )
                if pad > 0:
                    hp_c = hp_c[:CHUNK - pad]
                    hc_c = hc_c[:CHUNK - pad]
                hp_parts.append(np.array(hp_c, dtype=np.complex128))
                hc_parts.append(np.array(hc_c, dtype=np.complex128))
            hp = np.concatenate(hp_parts)
            hc = np.concatenate(hc_parts)

        # Apply coalescence phase
        if abs(phiref) > 1e-15:
            phase_factor = np.exp(-1j * phiref)
            hp *= phase_factor
            hc *= phase_factor

        return hp, hc


# Register the wrapper
WfWrapper["mlgw-bns-jax"] = WfMLGWJAX
__non_lal_approx_names__.append("mlgw-bns-jax")
