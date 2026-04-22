# ROQ Basis Generation — NumPy backend
## Setup guide for LIGO Jupyter servers

This guide walks you through creating a dedicated conda environment and
Jupyter kernel to run `build_roq_basis_numpy.py` on any LIGO Jupyter
server **without JAX**.

---

## Why a separate environment?

| | JAX backend | NumPy backend |
|---|---|---|
| Requires JAX / CUDA | ✓ | ✗ |
| Memory fragmentation | frequent OOM | none |
| Step-level resume | ✗ | ✓ |
| Parallel (multi-core) | GPU only | ThreadPoolExecutor |
| Python version | 3.9–3.11 | 3.10+ |

---

## Step 1 — Open a terminal on the LIGO Jupyter server

From JupyterLab: **File → New → Terminal**

---

## Step 2 — Create the conda environment

```bash
# Pick a name that is recognisable in the kernel list
ENV_NAME="roq_numpy"

conda create -y -n "${ENV_NAME}" python=3.11
conda activate "${ENV_NAME}"
```

> **Tip:** If `conda` is not on the PATH, try `source /opt/conda/etc/profile.d/conda.sh` first,
> or replace `conda` with `mamba` if Mamba is installed.

---

## Step 3 — Install Python dependencies

```bash
pip install --upgrade pip

# Core scientific stack (no JAX, no GPU packages)
pip install numpy scipy h5py

# Jupyter kernel support
pip install ipykernel ipywidgets

# Optional: progress bars in notebooks
pip install tqdm
```

Exact minimum versions tested:

| Package | Minimum | Recommended |
|---------|---------|-------------|
| Python | 3.10 | 3.11 |
| numpy | 1.24 | 2.0+ |
| scipy | 1.10 | 1.13+ |
| h5py | 3.6 | 3.11+ |
| ipykernel | 6.0 | 6.29+ |

---

## Step 4 — Register the Jupyter kernel

```bash
python -m ipykernel install \
    --user \
    --name "${ENV_NAME}" \
    --display-name "Python (ROQ NumPy)"
```

Refresh the JupyterLab page.  The new kernel will appear as
**"Python (ROQ NumPy)"** in the kernel selector.

---

## Step 5 — Copy the required files to your working directory

```
mlgw_bns_jax_model.h5          ← the trained model (must already exist)
config_roq_basis_2.ini          ← config copied from the repo
build_roq_basis_numpy.py        ← this file from the repo
```

All three files should be in the **same directory** (e.g. `~/roq_work/`).

---

## Step 6 — Edit the configuration (optional)

Open `config_roq_basis_2.ini` and confirm these sections are correct
for your run.  The defaults below match the problem statement:

```ini
[Waveform_and_parametrisation]
f-min   = 23.0
f-max   = 2000.0
seglen  = 128.0          ; → df = 1/128 ≈ 0.0078 Hz, 253 057 freq points

[ROQ]
tolerance-lin          = 1.0e-4
tolerance-qua          = 1.0e-6
n-pre-basis-lin        = 200
n-pre-basis-qua        = 10
n-pre-basis-search-iter= 1000
n-training-set-cycles  = 3
training-set-sizes     = 10000,100000,100000
training-set-rel-tol   = 0.1,1.0,1.0

[Training_range]
mc-min = 1.18 ; mc-max = 1.21
q-min  = 1.0  ; q-max  = 2.0
...

[I/O]
output      = ./ROQ_basis_2
random-seed = 170817
verbose     = 1
```

---

## Step 7 — Run from a Jupyter notebook

Create a new notebook, select the **"Python (ROQ NumPy)"** kernel, then:

```python
# ── Cell 1: imports ──────────────────────────────────────────────────
import os, sys, time
sys.path.insert(0, "/path/to/directory/containing/build_roq_basis_numpy")

import build_roq_basis_numpy as roq

# ── Cell 2: configuration ─────────────────────────────────────────────
cfg = roq.ROQConfig.from_ini("config_roq_basis_2.ini")

# Set output directory (must be writable, ideally on fast local disk)
cfg.output_dir = "./ROQ_basis_2"

# ── IMPORTANT: set n_workers to the number of physical CPU cores ──────
# Check available cores:
print("Available CPUs:", os.cpu_count())
cfg.n_workers = 8      # adjust to the number of cores on your node
                       # typical LIGO nodes: 16–32 cores

# projection_batch_size controls peak RAM:
#   500  waveforms × 253 057 freq × 16 bytes (complex128) ≈ 2 GB
#   1000 waveforms                                         ≈ 4 GB
cfg.projection_batch_size = 500   # safe default; increase if you have >32 GB RAM

os.makedirs(cfg.output_dir, exist_ok=True)
print(f"Frequency points : {cfg.n_freq}")
print(f"Workers          : {cfg.n_workers}")
print(f"Proj batch size  : {cfg.projection_batch_size}")

# ── Cell 3: build LINEAR basis ────────────────────────────────────────
# resume=True means the run can be interrupted and restarted at any time.
# Step-level checkpoints are saved after EVERY greedy step.
t0 = time.time()
results_lin = roq.build_roq_basis(cfg, kind="linear", resume=True)
print(f"\nLinear basis done in {(time.time()-t0)/3600:.2f} h")
print(f"  Empirical nodes: {len(results_lin['nodes'])}")

# ── Cell 4: build QUADRATIC basis ─────────────────────────────────────
t0 = time.time()
results_qua = roq.build_roq_basis(cfg, kind="quadratic", resume=True)
print(f"\nQuadratic basis done in {(time.time()-t0)/3600:.2f} h")
print(f"  Empirical nodes: {len(results_qua['nodes'])}")
```

---

## Step 8 — Resume an interrupted run

Simply **re-run the same cells**.  The script checks for checkpoint files
automatically:

| Checkpoint file | What it saves |
|---|---|
| `ROQ_basis_2/ROQ_data/linear/_greedy_ckpt_linear/` | Basis + params + error history after **every greedy step** |
| `ROQ_basis_2/ROQ_data/linear/preselection_linear_basis.npy` | Full pre-selection output |
| `ROQ_basis_2/ROQ_data/linear/basis_linear.npy` | Enriched basis |
| `ROQ_basis_2/ROQ_data/linear/_status_linear.json` | Phase completion flags |

The step-level checkpoint is the most important one: even if a single
greedy step takes ~1 900 s (as observed in the JAX run), a restart will
skip all previously completed steps and continue from the last one.

---

## Step 9 — Run from the command line (alternative)

If you prefer a terminal instead of a notebook:

```bash
conda activate roq_numpy
cd ~/roq_work

# Edit n_workers in the script or pass a custom config
python build_roq_basis_numpy.py config_roq_basis_2.ini
```

To keep the job running after you log out, use `nohup` or `screen`:

```bash
# With nohup (output goes to nohup.out)
nohup python build_roq_basis_numpy.py config_roq_basis_2.ini &

# With screen (interactive, attach later with: screen -r roq)
screen -S roq
python build_roq_basis_numpy.py config_roq_basis_2.ini
# Detach: Ctrl-A then D
```

---

## Performance guidance

### Choosing `n_workers`

`n_workers` controls the number of threads used for spline evaluation
(the dominant cost per waveform).  scipy releases the GIL during C-level
evaluation, so threads give real speedup.

| Cores available | Recommended `n_workers` | Expected speedup vs 1 core |
|---|---|---|
| 4 | 4 | ~2.5× |
| 8 | 8 | ~4–5× |
| 16 | 16 | ~7–9× |
| 32 | 32 | ~10–14× |

Do not set `n_workers` higher than the physical core count; hyperthreading
does not help for memory-bound spline evaluation.

### Choosing `projection_batch_size`

| RAM available | Recommended `projection_batch_size` |
|---|---|
| 16 GB | 250 |
| 32 GB | 500 (default) |
| 64 GB | 1 000 |
| 128 GB | 2 000 |

Larger batches reduce the number of Python-loop iterations but increase
peak memory.  The dominant allocation is:
`batch × n_freq × 16 bytes` (one complex128 waveform array).

### Estimated wall times (linear basis, 200 steps)

These are rough estimates for the pre-selection phase only.

| Cores | ms/wf | Time/step | Total (200 steps) |
|---|---|---|---|
| 1 | ~30 | ~4 600 s | ~256 h |
| 4 | ~12 | ~2 400 s | ~133 h |
| 8 | ~7  | ~1 600 s | ~89 h |
| 16 | ~4  | ~1 000 s | ~56 h |
| 32 | ~3  | ~750 s | ~42 h |

> For context, the JAX/GPU run was measuring ~1 926 s/step on a single GPU.
> With 16+ CPU cores the NumPy backend is **comparable or faster**.

---

## Verifying outputs

After the build completes, check the outputs:

```python
import numpy as np

out = "./ROQ_basis_2/ROQ_data"

nodes_lin = np.load(f"{out}/linear/empirical_nodes_linear.npy")
nodes_qua = np.load(f"{out}/quadratic/empirical_nodes_quadratic.npy")
freqs_lin = np.load(f"{out}/linear/empirical_frequencies_linear.npy")

print(f"Linear   : {len(nodes_lin)} nodes, "
      f"freq range [{freqs_lin.min():.1f}, {freqs_lin.max():.1f}] Hz")
print(f"Quadratic: {len(nodes_qua)} nodes")
```

The outputs are identical in format to those produced by the JAX builder
and can be used directly with the existing likelihood / inference code.

---

## Troubleshooting

| Issue | Solution |
|---|---|
| `ModuleNotFoundError: No module named 'jax'` | You are using the wrong kernel; switch to **"Python (ROQ NumPy)"** |
| `MemoryError` or kernel killed | Reduce `projection_batch_size` |
| Run hangs with 100% CPU on one core | Check that `cfg.n_workers > 1` |
| `FileNotFoundError: mlgw_bns_jax_model.h5` | Ensure the model file is in the working directory |
| Want to restart from scratch | Delete `ROQ_basis_2/` and re-run |
| Want to restart from a specific phase | Delete the `_status_*.json` file for that phase |
| Want to restart from a specific greedy step | Delete `_greedy_ckpt_*/meta.json` for that step |
