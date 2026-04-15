# Reduced Order Quadrature (ROQ) for mlgw\_bns\_jax

This directory contains everything needed to build an ROQ basis for the
**mlgw\_bns** waveform model and run ROQ-accelerated parameter estimation
for GW170817, as described in the
[mlgw\_bns paper](https://arxiv.org/abs/2210.15684).

---

## Why ROQ?

For GW170817 parameter estimation with a 128 s segment at 4 kHz sampling
rate, the full frequency grid has **~253,000 points** (from 23 Hz to
2000 Hz at Δf = 1/128 Hz). This makes likelihood evaluations extremely
expensive:

- On an NVIDIA A100, only ~80 particles fit in memory — far too few for
  reliable parameter estimation.
- Each likelihood evaluation requires inner products over the full 253k grid.

The ROQ compresses this to **O(1000) empirical nodes**, giving:

- **~100–250× speedup** in likelihood evaluation
- **500+ particles** for robust sampling with SHARPy

## Why NumPy for Building the ROQ (not JAX)?

JAX's LLVM JIT compilation has significant memory overhead. During ROQ
basis construction, JenpyROQ generates tens of thousands of waveforms,
and JAX's compilation cache and tracing allocations lead to
**out-of-memory (OOM)** errors on machines with ≤16 GB RAM (Google
Colab, IGWN JupyterHub).

**Solution:** Use the original NumPy-based `mlgw_bns` model for the
ROQ build phase. The resulting ROQ interpolants are pure NumPy arrays
that can be loaded by JAX for parameter estimation.

---

## Quick Start

### Step 1: Build the ROQ basis (NumPy, no JAX)

**Option A — Notebook** (recommended for Colab / IGWN Jupyter):
```
build_roq_basis_numpy.ipynb
```

**Option B — Script** (recommended for command-line / batch jobs):
```bash
python build_roq_basis_numpy.py
```

Both produce the same output in `roq_basis_mlgw_bns_jax/ROQ_data/`.

### Step 2: Parameter estimation with ROQ (JAX + SHARPy)

```
roq_pe_mlgw_bns_jax.ipynb
```

This notebook loads the pre-built ROQ basis and runs SHARPy's SMC sampler
with the ROQ-accelerated likelihood.

---

## Environment Setup

### Google Colab

The notebooks auto-install all dependencies. No manual setup needed.
Just open the notebook and run all cells.

### IGWN JupyterHub (LIGO/Virgo Jupyter)

**Option 1: Conda environment** (recommended):

```bash
# Create the environment
conda env create -f environment.yml
conda activate mlgw-bns-jax

# Install SHARPy
git clone --depth 1 https://github.com/gabrieledemasi/sharpy.git _sharpy_repo
pip install -e ./_sharpy_repo
pip install -e .

# Register Jupyter kernel
python -m ipykernel install --user --name mlgw-bns-jax --display-name "mlgw-bns-jax"
```

If `conda env create` fails due to base environment conflicts:

```bash
conda create -n mlgw-bns-jax -c conda-forge --override-channels \
  python=3.11 pip framel lalsuite gwpy ipykernel \
  numpy=2.0 scipy matplotlib h5py astropy numba pandas scikit-learn tqdm

conda activate mlgw-bns-jax

pip install "jax[cpu]==0.4.38" jaxopt optax equinox flax ripplegw corner \
  anesthetic seaborn "NetKet==3.20.5" PyYAML toml dacite joblib \
  sortedcontainers requests \
  "blackjax @ git+https://github.com/gabrieledemasi/blackjax@main" \
  "JenpyROQ @ git+https://github.com/GCArullo/JenpyROQ.git"

pip install -e ./_sharpy_repo && pip install -e .
```

**Option 2: Minimal install** (ROQ build only, no JAX):

```bash
pip install h5py scikit-learn poetry-core matplotlib
pip install "JenpyROQ @ git+https://github.com/GCArullo/JenpyROQ.git"
pip install -e .
```

### Local Machine

Same as IGWN instructions above. For GPU-accelerated PE, install
`jax[cuda12]` instead of `jax[cpu]`.

---

## File Reference

### ROQ Build (NumPy — no JAX)

| File | Description |
|------|-------------|
| `build_roq_basis_numpy.py` | Standalone script to build ROQ basis using NumPy |
| `build_roq_basis_numpy.ipynb` | Notebook version (with Colab/IGWN setup) |
| `mlgw_bns_roq_wrapper.py` | JenpyROQ wrapper using original mlgw\_bns (NumPy) |
| `config_roq_mlgw_bns_jax_gw170817.ini` | JenpyROQ configuration for GW170817 |

### ROQ Build (JAX — alternative, needs more RAM)

| File | Description |
|------|-------------|
| `build_roq_basis.py` | Standalone script using JAX wrapper |
| `build_roq_basis.ipynb` | Notebook version using JAX wrapper |
| `mlgw_bns_jax_roq_wrapper.py` | JenpyROQ wrapper using JAX model |

### Parameter Estimation

| File | Description |
|------|-------------|
| `roq_pe_mlgw_bns_jax.ipynb` | ROQ PE notebook (SHARPy SMC + JAX) |
| `gw170817_pe_sharpy.py` | Full-grid PE script (no ROQ, for comparison) |

### Model & Data

| File | Description |
|------|-------------|
| `mlgw_bns_jax_model.h5` | Pre-trained mlgw\_bns model (HDF5) |
| `jax_import_n_predict.py` | Standalone JAX predictor (loads HDF5 model) |
| `environment.yml` | Conda environment specification |

---

## Output Structure

After running the ROQ build, the output directory contains:

```
roq_basis_mlgw_bns_jax/
├── ROQ_data/
│   ├── linear/
│   │   ├── basis_interpolant_linear.npy    # B_lin: (N_full, N_lin) matrix
│   │   ├── empirical_nodes_linear.npy      # Frequency indices for ⟨d|h⟩
│   │   ├── empirical_frequencies_linear.npy
│   │   └── ...
│   └── quadratic/
│       ├── basis_interpolant_quadratic.npy  # B_qua: (N_full, N_qua) matrix
│       ├── empirical_nodes_quadratic.npy    # Frequency indices for ⟨h|h⟩
│       ├── empirical_frequencies_quadratic.npy
│       └── ...
├── config_roq_mlgw_bns_jax_gw170817.ini
└── git_info.txt
```

## How ROQ Works

The standard GW log-likelihood requires inner products over the full
frequency grid:

```
⟨d|h⟩ = 4 Δf Σ_k conj(d_k) h_k / S_n(f_k)
⟨h|h⟩ = 4 Δf Σ_k |h_k|² / S_n(f_k)
```

With ROQ, these are replaced by sums over O(1000) empirical nodes:

```
⟨d|h⟩ ≈ Σ_j w_j^lin · d(f_j^lin)* · h(f_j^lin)
⟨h|h⟩ ≈ Σ_j w_j^qua · |h(f_j^qua)|²
```

The weights `w_j` are precomputed from the basis interpolant matrices
and the PSD, and the waveform only needs to be evaluated at the
empirical node frequencies.

---

## References

- **mlgw\_bns paper:** Tissino et al., *Machine learning gravitational
  waveforms for binary neutron star mergers*, [arXiv:2210.15684](https://arxiv.org/abs/2210.15684)
- **JenpyROQ:** [github.com/GCArullo/JenpyROQ](https://github.com/GCArullo/JenpyROQ)
- **SHARPy:** [github.com/gabrieledemasi/sharpy](https://github.com/gabrieledemasi/sharpy)
