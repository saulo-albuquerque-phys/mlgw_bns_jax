"""
Pre-load and cache all model data for optimal performance.
This module loads all data once at import time and provides JAX arrays.
"""
import jax.numpy as jnp
import numpy as np
from pathlib import Path

# Define data directory
DATA_DIR = Path(__file__).parent / "data_default_NN"

class ModelDataCache:
    """Singleton class to hold all pre-loaded model data."""
    
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelDataCache, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not ModelDataCache._initialized:
            self._load_all_data()
            ModelDataCache._initialized = True
    
    def _load_all_data(self):
        """Load all data once and convert to JAX arrays."""
        # Neural network parameters
        loadweightbiases = np.load(DATA_DIR / "mlp_jax_params.npz")
        
        self.weights = []
        self.biases = []
        
        i = 0
        while f"W{i}" in loadweightbiases:
            self.weights.append(jnp.array(loadweightbiases[f"W{i}"]))
            self.biases.append(jnp.array(loadweightbiases[f"b{i}"]))
            i += 1
        
        self.parameters_NN = (self.weights, self.biases)
        
        # Scaler parameters
        scaler_data = np.load(DATA_DIR / "scaler_params.npz")
        self.mean = jnp.array(scaler_data["mean"])
        self.scale = jnp.array(scaler_data["scale"])
        
        # Downsampling indexes
        self.indexes_downsampling = np.loadtxt(DATA_DIR / "mlp_jax_downsampling_indexes.dat")
        self.indexes_downsampling_amplitude = np.loadtxt(DATA_DIR / "mlp_jax_downsampling_indexes_amplitude.dat")
        self.indexes_downsampling_phase = np.loadtxt(DATA_DIR / "mlp_jax_downsampling_indexes_phase.dat")
        
        self.int_indexes_amp = self.indexes_downsampling_amplitude.astype(int)
        self.int_indexes_phase = self.indexes_downsampling_phase.astype(int)
        
        # PCA parameters
        pca_diction = np.load(DATA_DIR / "mlp_jax_pca_params.npz")
        
        self.pca_data_exponent = jnp.array(pca_diction['pca_exponent_data'])
        self.pc_exponent = self.pca_data_exponent
        self.pca_data_scaling = jnp.array(pca_diction['pca_data_scaling'])
        self.pca_data_eigenvectors = jnp.array(pca_diction['pca_data_eigenvectors'])
        self.pca_data_eigenvalues = jnp.array(pca_diction['pca_data_eigenvalues'])
        self.pca_data_mean = jnp.array(pca_diction['pca_data_mean'])
        
        # Dataset parameters
        model_dataset_bibl = np.load(DATA_DIR / "mlp_jax_dataset_training_hyperparams.npz")
        
        self.frequencies_hz = jnp.array(model_dataset_bibl['frequencies_hz'])
        self.frequencies = jnp.array(model_dataset_bibl['frequencies'])
        self.total_mass_training = jnp.array(model_dataset_bibl['total_mass'])
        
        # Pre-compute downsampled frequency grids
        self.frequencies_saved_input_amp = self.frequencies_hz[self.int_indexes_amp]
        self.frequencies_saved_input_phase = self.frequencies_hz[self.int_indexes_phase]
        
        # Cache split points
        self.indexes_downsampling_new_int = (int(self.indexes_downsampling[0]), int(self.indexes_downsampling[1]))
        self.amp_points, self.phase_points = self.indexes_downsampling_new_int

# Create global cache instance
_cache = ModelDataCache()

# Export cached data
parameters_NN = _cache.parameters_NN
mean = _cache.mean
scale = _cache.scale
int_indexes_amp = _cache.int_indexes_amp
int_indexes_phase = _cache.int_indexes_phase
pca_data_exponent = _cache.pca_data_exponent
pc_exponent = _cache.pc_exponent
pca_data_scaling = _cache.pca_data_scaling
pca_data_eigenvectors = _cache.pca_data_eigenvectors
pca_data_eigenvalues = _cache.pca_data_eigenvalues
pca_data_mean = _cache.pca_data_mean
frequencies_hz = _cache.frequencies_hz
frequencies = _cache.frequencies
total_mass_training = _cache.total_mass_training
frequencies_saved_input_amp = _cache.frequencies_saved_input_amp
frequencies_saved_input_phase = _cache.frequencies_saved_input_phase
amp_points = _cache.amp_points
phase_points = _cache.phase_points
