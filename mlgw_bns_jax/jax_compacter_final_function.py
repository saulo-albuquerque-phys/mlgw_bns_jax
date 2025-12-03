import jax
import jax.numpy as jnp
import numpy
from .loading_default_nn import full_reconstruct_data_NN_jax,from_combined_residuals_jit
from .loading_default_nn import int_indexes_amp, int_indexes_phase
from .jax_compacter_taylorf2jax_functions import phase_5h_post_newtonian_tidal_jax,  amplitude_3h_post_newtonian_jax
from .jax_compacter_model_functions import combine_amp_phase_jax,combine_residuals_amp_jax,combine_residuals_phi_jax,compute_polarizations_jax
from .jax_compacter_downsampling_interpolation import resample, linear_resample_jax
from .jax_compacter_dataset_generation import mlgw_bns_prefactor_jax, eta

model_dataset_bibl=numpy.load("mlgw_bns_jax/data_default_NN/mlp_jax_dataset_training_hyperparams.npz")

frequencies_hz=model_dataset_bibl['frequencies_hz']
frequencies=model_dataset_bibl['frequencies']
total_mass_training=model_dataset_bibl['total_mass']

###################
#Downsampled grid
###################

frequencies_saved_input_amp=frequencies_hz[int_indexes_amp]
frequencies_saved_input_phase=frequencies_hz[int_indexes_phase]

################
#parameter array
# param_array=jnp.array([[q,lamba1,lambda2,chi1, chi2]])
#################

### function
@jax.jit
def mlgw_bns_one_waveform(frequency_given, total_mass, mass_ratio,lambda_1,lambda_2, chi_1, chi_2, distance_mpc, reference_phase, time_shift, inclination):
    param_array=jnp.array([[mass_ratio,lambda_1,lambda_2,chi_1, chi_2]])
    combined_residuals=full_reconstruct_data_NN_jax(param_array)
    amp_residuals, phase_residuals=from_combined_residuals_jit(combined_residuals)
    new_pn_amplitude=amplitude_3h_post_newtonian_jax(frequencies, mass_ratio, chi_1, chi_2, lambda_1, lambda_2)
    new_pn_phase=phase_5h_post_newtonian_tidal_jax(frequencies, mass_ratio, chi_1, chi_2, lambda_1, lambda_2)
    pn_amplitude_final=new_pn_amplitude[int_indexes_amp]
    pn_phase_final=new_pn_phase[int_indexes_phase]
    # downsampled amplitude array
    amp_ds = combine_residuals_amp_jax(amp_residuals, pn_amplitude_final)
    phi_ds = combine_residuals_phi_jax(phase_residuals, pn_phase_final)
    etaa=eta(mass_ratio)
    pre = mlgw_bns_prefactor_jax(etaa, total_mass)
    rescaled_frequencies = frequency_given * (total_mass / total_mass_training)
    resampled_amp_linear=linear_resample_jax(frequencies_saved_input_amp,rescaled_frequencies, amp_ds)
    resampled_phase_linear=linear_resample_jax(frequencies_saved_input_phase,rescaled_frequencies, phi_ds)
    amp = (resampled_amp_linear* pre/ distance_mpc)
    phi = (resampled_phase_linear + reference_phase  + (2 * jnp.pi * time_shift) * frequency_given)
    cartesian_waveform_real, cartesian_waveform_imag = combine_amp_phase_jax(amp, phi)
    cosi = jnp.cos(inclination)
    pre_plus = (1 + cosi ** 2) / 2
    pre_cross = cosi
    polarizations_jax_plus,polarizations_jax_cross=compute_polarizations_jax(cartesian_waveform_real, cartesian_waveform_imag, pre_plus, pre_cross)
    return polarizations_jax_plus, polarizations_jax_cross