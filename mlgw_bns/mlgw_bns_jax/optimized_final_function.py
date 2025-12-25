"""
Optimized main waveform generation function with pre-loaded data and efficient operations.
"""
import jax
import jax.numpy as jnp
from .optimized_nn import full_reconstruct_data_NN_optimized, from_combined_residuals_optimized
from .data_cache import (
    frequencies, total_mass_training, 
    frequencies_saved_input_amp, frequencies_saved_input_phase,
    int_indexes_amp, int_indexes_phase
)
from .jax_compacter_taylorf2jax_functions import phase_5h_post_newtonian_tidal_jax, amplitude_3h_post_newtonian_jax
from .jax_compacter_model_functions import combine_amp_phase_jax, combine_residuals_amp_jax, combine_residuals_phi_jax, compute_polarizations_jax
from .jax_compacter_downsampling_interpolation import linear_resample_jax
from .jax_compacter_dataset_generation import mlgw_bns_prefactor_jax, eta

# Pre-compute static arrays that don't change
@jax.jit
def precompute_pn_amplitudes_phases(mass_ratio, chi_1, chi_2, lambda_1, lambda_2):
    """Pre-compute PN amplitudes and phases for the full frequency grid."""
    new_pn_amplitude = amplitude_3h_post_newtonian_jax(frequencies, mass_ratio, chi_1, chi_2, lambda_1, lambda_2)
    new_pn_phase = phase_5h_post_newtonian_tidal_jax(frequencies, mass_ratio, chi_1, chi_2, lambda_1, lambda_2)
    
    # Extract downsampled values
    pn_amplitude_final = new_pn_amplitude[int_indexes_amp]
    pn_phase_final = new_pn_phase[int_indexes_phase]
    
    return pn_amplitude_final, pn_phase_final

@jax.jit
def mlgw_bns_one_waveform_optimized(
    frequency_given, total_mass, mass_ratio, lambda_1, lambda_2, 
    chi_1, chi_2, distance_mpc, reference_phase, time_shift, inclination
):
    """
    Optimized waveform generation with pre-loaded data and efficient operations.
    
    This version eliminates I/O bottlenecks and optimizes array operations.
    """
    # Create parameter array
    param_array = jnp.array([[mass_ratio, lambda_1, lambda_2, chi_1, chi_2]])
    
    # Get neural network residuals (optimized)
    combined_residuals = full_reconstruct_data_NN_optimized(param_array)
    amp_residuals, phase_residuals = from_combined_residuals_optimized(combined_residuals)
    
    # Pre-compute PN amplitudes and phases
    pn_amplitude_final, pn_phase_final = precompute_pn_amplitudes_phases(
        mass_ratio, chi_1, chi_2, lambda_1, lambda_2
    )
    
    # Combine residuals with PN
    amp_ds = combine_residuals_amp_jax(amp_residuals, pn_amplitude_final)
    phi_ds = combine_residuals_phi_jax(phase_residuals, pn_phase_final)
    
    # Compute prefactor
    etaa = eta(mass_ratio)
    pre = mlgw_bns_prefactor_jax(etaa, total_mass)
    
    # Frequency rescaling
    rescaled_frequencies = frequency_given * (total_mass / total_mass_training)
    
    # Linear interpolation (optimized)
    resampled_amp_linear = linear_resample_jax(frequencies_saved_input_amp, rescaled_frequencies, amp_ds)
    resampled_phase_linear = linear_resample_jax(frequencies_saved_input_phase, rescaled_frequencies, phi_ds)
    
    # Final amplitude and phase
    amp = (resampled_amp_linear * pre / distance_mpc)
    phi = (resampled_phase_linear + reference_phase + (2 * jnp.pi * time_shift) * frequency_given)
    
    # Convert to Cartesian
    cartesian_waveform_real, cartesian_waveform_imag = combine_amp_phase_jax(amp, phi)
    
    # Compute polarizations
    cosi = jnp.cos(inclination)
    pre_plus = (1 + cosi ** 2) / 2
    pre_cross = cosi
    
    polarizations_jax_plus, polarizations_jax_cross = compute_polarizations_jax(
        cartesian_waveform_real, cartesian_waveform_imag, pre_plus, pre_cross
    )
    
    return polarizations_jax_plus, polarizations_jax_cross

# For backwards compatibility
mlgw_bns_one_waveform = mlgw_bns_one_waveform_optimized
