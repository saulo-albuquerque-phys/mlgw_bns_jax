#!/usr/bin/env bash
# ============================================================================
#  One-shot environment setup for IGWN / LIGO JupyterHub
# ============================================================================
#
#  Run this ONCE from a terminal on the IGWN Jupyter server:
#
#      cd ~/mlgw_bns_jax          # or wherever you cloned the repo
#      bash setup_igwn.sh
#
#  Then select the "mlgw-bns-jax" kernel in any notebook.
#
#  What this script does:
#    1. Creates an isolated conda environment (Python 3.11)
#    2. Installs conda-forge packages (framel, lalsuite, gwpy, ...)
#    3. Installs pip packages (JenpyROQ, mlgw_bns, setuptools, ...)
#    4. Pins JAX to 0.4.38 (some deps try to upgrade it)
#    5. Patches JenpyROQ for numpy 2.0 compatibility
#    6. Registers the kernel for Jupyter
#
#  If the script fails, delete the env and retry:
#      conda env remove -n mlgw-bns-jax
#      bash setup_igwn.sh
# ============================================================================

set -euo pipefail

ENV_NAME="mlgw-bns-jax"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "============================================================"
echo "  Setting up ${ENV_NAME} environment"
echo "  Repo: ${REPO_DIR}"
echo "============================================================"

# ── 1. Create conda environment ────────────────────────────────────
if conda env list | grep -q "^${ENV_NAME} "; then
    echo ""
    echo "Environment '${ENV_NAME}' already exists."
    echo "To recreate, first run:  conda env remove -n ${ENV_NAME}"
    echo "Then re-run this script."
    echo ""
    read -rp "Continue with existing env? [y/N] " yn
    case $yn in
        [Yy]*) ;;
        *) echo "Aborted."; exit 1 ;;
    esac
else
    echo ""
    echo ">>> Creating conda environment '${ENV_NAME}' ..."
    conda create -n "${ENV_NAME}" -c conda-forge --override-channels -y \
        python=3.11 pip \
        framel lalsuite gwpy ipykernel \
        numpy=2.0 scipy matplotlib h5py astropy numba \
        pandas scikit-learn tqdm
fi

# ── 2. Activate ────────────────────────────────────────────────────
echo ""
echo ">>> Activating environment ..."
eval "$(conda shell.bash hook)"
conda activate "${ENV_NAME}"

echo "  Python: $(python --version)"
echo "  Prefix: ${CONDA_PREFIX}"

# ── 3. Install pip packages ────────────────────────────────────────
echo ""
echo ">>> Installing pip packages ..."

# JAX stack (pinned)
pip install --quiet "jax[cpu]==0.4.38" jaxopt optax equinox flax

# Waveform & PE tools
pip install --quiet ripplegw corner anesthetic seaborn \
    PyYAML toml dacite joblib sortedcontainers requests

# NetKet (may try to upgrade JAX — we re-pin below)
pip install --quiet "NetKet==3.20.5"

# Git-only packages
pip install --quiet \
    "blackjax @ git+https://github.com/gabrieledemasi/blackjax@main" \
    "JenpyROQ @ git+https://github.com/GCArullo/JenpyROQ.git"

# ── 4. Re-pin JAX (NetKet/SHARPy may have upgraded it) ────────────
echo ""
echo ">>> Pinning JAX to 0.4.38 ..."
pip install --quiet "jax[cpu]==0.4.38"

# ── 5. Ensure setuptools has pkg_resources ─────────────────────────
echo ""
echo ">>> Ensuring setuptools < 71 (for pkg_resources) ..."
pip install --quiet "setuptools<71"

# ── 6. Clone and install SHARPy ────────────────────────────────────
cd "${REPO_DIR}"

if [ ! -d "_sharpy_repo" ]; then
    echo ""
    echo ">>> Cloning SHARPy ..."
    git clone https://github.com/saulo-albuquerque-phys/sharpy.git _sharpy_repo
fi

echo ""
echo ">>> Installing SHARPy (editable) ..."
pip install --quiet --no-deps -e ./_sharpy_repo

# ── 7. Install mlgw_bns (this repo) ───────────────────────────────
echo ""
echo ">>> Installing mlgw_bns (editable) ..."
pip install --quiet --no-deps -e .

# ── 8. Re-pin JAX again (sharpy install may upgrade) ───────────────
pip install --quiet "jax[cpu]==0.4.38"

# ── 9. Patch JenpyROQ for numpy 2.0 ───────────────────────────────
echo ""
echo ">>> Patching JenpyROQ for numpy 2.0 compatibility ..."
SITE_PKGS="$(python -c 'import site; print(site.getsitepackages()[0])')"
for f in "${SITE_PKGS}/JenpyROQ/jenpyroq.py" "${SITE_PKGS}/JenpyROQ/__main__.py"; do
    if [ -f "$f" ] && grep -q "np\.VisibleDeprecationWarning" "$f"; then
        sed -i 's/category=np\.VisibleDeprecationWarning/category=getattr(np, "VisibleDeprecationWarning", FutureWarning)/' "$f"
        echo "  Patched: $f"
    fi
done

# ── 10. Register Jupyter kernel ────────────────────────────────────
echo ""
echo ">>> Registering Jupyter kernel ..."
python -m ipykernel install --user --name "${ENV_NAME}" --display-name "${ENV_NAME}"

# ── 11. Verify ─────────────────────────────────────────────────────
echo ""
echo ">>> Verifying imports ..."
python -c "
import os; os.environ['JAX_PLATFORMS'] = 'cpu'
import jax;         print(f'  jax:        {jax.__version__}')
import numpy as np; print(f'  numpy:      {np.__version__}')
import scipy;       print(f'  scipy:      {scipy.__version__}')
from JenpyROQ.jenpyroq import JenpyROQ; print('  JenpyROQ:   OK')
import blackjax;    print('  blackjax:   OK')
import mlgw_bns;    print(f'  mlgw_bns:   {mlgw_bns.__version__}')
import sharpy;      print('  sharpy:     OK')
"

echo ""
echo "============================================================"
echo "  Setup complete!"
echo ""
echo "  To use in a notebook:"
echo "    Select kernel: ${ENV_NAME}"
echo ""
echo "  To use from a terminal:"
echo "    conda activate ${ENV_NAME}"
echo "    cd ${REPO_DIR}"
echo "    python build_roq_basis_numpy.py"
echo ""
echo "  To build the ROQ basis from a notebook:"
echo "    Open build_roq_basis_numpy.ipynb"
echo "    Select kernel: ${ENV_NAME}"
echo "    Run All Cells"
echo "============================================================"
