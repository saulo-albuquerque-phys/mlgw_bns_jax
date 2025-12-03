import jax
#import jax.numpy as np
import numpy as np
import mlgw_bns
from mlgw_bns import Model



###################################################################################################
### Choosing the saved trained model to save data and metadata 
### (here we choose the default trained model)
###################################################################################################

model = Model.default()

###################################################################################################
### Creating file to save all metadata and model features
###################################################################################################

import os
os.makedirs("mlgw_bns_jax/data_default_NN/", exist_ok=True)

###################################################################################################
### Trained model metadata saving
###################################################################################################

model_dataset=model.dataset

model_dataset_bibl={}


#######################################
### Training set hyperparameters input
#######################################



initial_frequency_hz=model.dataset.initial_frequency_hz
srate_hz=model.dataset.srate_hz
#model.dataset.waveform_generator
#model.dataset.parameter_generator_class
#model.dataset.parameter_generator_class
#model.dataset.parameter_ranges
mass_range=model.dataset.parameter_ranges.mass_range
q_range=model.dataset.parameter_ranges.q_range
lambda1_range=model.dataset.parameter_ranges.lambda1_range
lambda2_range=model.dataset.parameter_ranges.lambda2_range
chi1_range=model.dataset.parameter_ranges.chi1_range
chi2_range=model.dataset.parameter_ranges.chi2_range
multibanding_bool=model.dataset.multibanding
total_mass=model.dataset.total_mass

model_dataset_bibl['initial_frequency_hz']=initial_frequency_hz
model_dataset_bibl['srate_hz']=srate_hz
model_dataset_bibl['mass_range']=mass_range
model_dataset_bibl['q_range']=q_range
model_dataset_bibl['lambda1_range']=lambda1_range
model_dataset_bibl['lambda2_range']=lambda2_range
model_dataset_bibl['chi1_range']=chi1_range
model_dataset_bibl['chi2_range']=chi2_range
model_dataset_bibl['multibanding_bool']=multibanding_bool
model_dataset_bibl['total_mass']=total_mass



####################################################
### Metadata calculated from the formulas bellow
#from .multibanding import reduced_frequency_array
#from .dataset_generation import expand_frequency_range
#(effective_initial_frequency_hz, effective_srate_hz,) = expand_frequency_range(
#            model.dataset.initial_frequency_hz,
#            model.dataset.srate_hz,
#            model.dataset.parameter_ranges.mass_range,
#            model.dataset.total_mass,
#        )


#reduced_frequency_array=reduced_frequency_array(
#                model.dataset.effective_initial_frequency_hz,
#                model.dataset.effective_srate_hz / 2,
#                model.dataset.f_pivot_hz,
#            )
#frequencies=model.dataset.hz_to_natural_units(reduced_frequency_array)

####################################################


effective_initial_frequency_hz=model.dataset.effective_initial_frequency_hz
effective_srate_hz=model.dataset.effective_srate_hz
f_pivot_hz=model.dataset.f_pivot_hz
frequencies_hz=model.dataset.frequencies_hz
frequencies=model.dataset.frequencies


model_dataset_bibl['effective_initial_frequency_hz']=effective_initial_frequency_hz
model_dataset_bibl['effective_srate_hz']=effective_srate_hz
model_dataset_bibl['f_pivot_hz']=f_pivot_hz
model_dataset_bibl['frequencies_hz']=frequencies_hz
model_dataset_bibl['frequencies']=frequencies


####################################################
### saving model dataset metadata
####################################################


np.savez("mlgw_bns_jax/data_default_NN/mlp_jax_dataset_training_hyperparams.npz",**model_dataset_bibl)




###################################################################################################
### Model training features (parameters such as weights, bias from the NN),
### Eigenvalues and eigenvectors of the PCA matrix
### model SCALER
### extra parameters such as PCA exponent
###################################################################################################

####################################################
### neural network trained features
####################################################

NN=model.nn
mlp = NN.nn      # or net._model depending on the wrapper

weights = mlp.coefs_
biases  = mlp.intercepts_
scaler = NN.param_scaler

mean = scaler.mean_      # shape (input_dim,)
scale = scaler.scale_    # shape (input_dim,)

###################################################
### saving neural networks features
###################################################

save_dict = {}
for i, W in enumerate(weights):
    save_dict[f"W{i}"] = W
for i, b in enumerate(biases):
    save_dict[f"b{i}"] = b

np.savez("mlgw_bns_jax/data_default_NN/mlp_jax_params.npz", **save_dict)
np.savez("mlgw_bns_jax/data_default_NN/scaler_params.npz", mean=mean.astype(np.float32), scale=scale.astype(np.float32))

#################################################
### printing important info on NN model
#################################################

print("Hidden layers:", mlp.hidden_layer_sizes)
print("Number of layers:", mlp.n_layers_)
print("Input layer size:", mlp.coefs_[0].shape[0])
print("Output layer size:", mlp.coefs_[-1].shape[1])

print("Hidden activation:", mlp.activation)


###################################################
### PCA eigenvalues and eigenvectors
###################################################

pca_dict={}

pca_exponent_data=model.nn.hyper.pc_exponent
pca_data=model.pca_data
pca_data_scaling=pca_data.principal_components_scaling
pca_data_eigenvectors=pca_data.eigenvectors
pca_data_eigenvalues=pca_data.eigenvalues
pca_data_mean=pca_data.mean

pca_dict['pca_exponent_data']=pca_exponent_data
pca_dict['pca_data_scaling']=pca_data_scaling
pca_dict['pca_data_eigenvectors']=pca_data_eigenvectors
pca_dict['pca_data_eigenvalues']=pca_data_eigenvalues
pca_dict['pca_data_mean']=pca_data_mean

np.savez("mlgw_bns_jax/data_default_NN/mlp_jax_pca_params.npz", **pca_dict)

###################################################
### downsampling indexes data
###################################################

indexes_downsampling=model.downsampling_indices.numbers_of_points

np.savetxt("mlgw_bns_jax/data_default_NN/mlp_jax_downsampling_indexes.dat", indexes_downsampling)


indexes_downsampling_amplitude=model.downsampling_indices.amplitude_indices
indexes_downsampling_phase=model.downsampling_indices.phase_indices

np.savetxt("mlgw_bns_jax/data_default_NN/mlp_jax_downsampling_indexes_amplitude.dat", indexes_downsampling_amplitude)
np.savetxt("mlgw_bns_jax/data_default_NN/mlp_jax_downsampling_indexes_phase.dat", indexes_downsampling_phase)
