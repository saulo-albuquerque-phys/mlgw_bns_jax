import numpy as np
import jax
import jax.numpy as jnp
from importlib.resources import files

path = files("mlgw_bns").joinpath("mlgw_bns_jax/data_default_NN/")

loadweightbiases = np.load(path.joinpath("mlp_jax_params.npz"))

#loadweightbiases = np.load("mlgw_bns_jax/data_default_NN/mlp_jax_params.npz")

weightsload = []
biasesload = []

i = 0
while f"W{i}" in loadweightbiases:
    weightsload.append(jnp.array(loadweightbiases[f"W{i}"]))
    biasesload.append(jnp.array(loadweightbiases[f"b{i}"]))
    i += 1

parameters_NN = (weightsload, biasesload)

scaler_data = np.load(path.joinpath("scaler_params.npz")
mean = jnp.array(scaler_data["mean"])
scale = jnp.array(scaler_data["scale"])

indexes_downsampling=np.loadtxt(path.joinpath("mlp_jax_downsampling_indexes.dat")
indexes_downsampling_amplitude=np.loadtxt(path.joinpath("mlp_jax_downsampling_indexes_amplitude.dat")
indexes_downsampling_phase=np.loadtxt(path.joinpath("mlp_jax_downsampling_indexes_phase.dat")
int_indexes_amp = indexes_downsampling_amplitude.astype(int)
int_indexes_phase= indexes_downsampling_phase.astype(int)
int_list_indexes_amp = indexes_downsampling_amplitude.astype(int).tolist()
int_list_indexes_phase= indexes_downsampling_phase.astype(int).tolist()


pca_diction=np.load(path.joinpath("mlp_jax_pca_params.npz")

pca_data_exponent=jnp.array(pca_diction['pca_exponent_data'])
pc_exponent=pca_data_exponent
pca_data_scaling=jnp.array(pca_diction['pca_data_scaling'])
pca_data_eigenvectors=jnp.array(pca_diction['pca_data_eigenvectors'])
pca_data_eigenvalues=jnp.array(pca_diction['pca_data_eigenvalues'])
pca_data_mean=jnp.array(pca_diction['pca_data_mean'])



@jax.jit
def mlp_forward_jax(x):
    # 1️⃣ scale input
    x_scaled = (x - mean) / scale

    # 2️⃣ forward pass (hard-coded)
    W0, W1, W2 =parameters_NN[0]
    b0, b1, b2 =parameters_NN[1]

    z1 = x_scaled @ W0 + b0
    a1 = jnp.tanh(z1)

    z2 = a1 @ W1 + b1
    a2 = jnp.tanh(z2)

    z3 = a2 @ W2 + b2
    return z3

@jax.jit
def full_reconstruct_data_NN_jax(x):
        """Reconstruct the data.

        Parameters
        ----------
        reduced_data : np.ndarray
            With shape ``(number_of_points, number_of_components)``.
        pca_data : PrincipalComponentData
            To use in the reconstruction.

        Returns
        -------
        reconstructed_data: np.ndarray
            With shape ``(number_of_points, number_of_dimensions)``.
        """
        pca_reduced_data=mlp_forward_jax(x)
        reduced_data= pca_reduced_data/( pca_data_eigenvalues**pc_exponent)
        # (npoints, npca) = (npoints, npca) * (npca)
        scaled_data = (
            reduced_data *pca_data_scaling[jnp.newaxis, :]
        )

        # (npoints, ndims) = (npoints, npca) @ (npca, ndims)
        zero_mean_data = scaled_data @ pca_data_eigenvectors.T

        return zero_mean_data + pca_data_mean


indexes_downsampling_new_int=(int(indexes_downsampling[0]),int(indexes_downsampling[1]))
amp_points, phase_points =indexes_downsampling_new_int

@jax.jit
def from_combined_residuals_jit(combined_residuals):
        #assert combined_residuals.shape[1] == amp_points + phase_points
        return combined_residuals[:, :amp_points], combined_residuals[:, -phase_points:]
