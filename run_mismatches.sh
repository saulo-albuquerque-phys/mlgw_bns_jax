#!/bin/bash
# Run phase 1 and phase 2 in a loop until complete, handling restarts.
set -e
cd /workspaces/mlgw_bns_jax

TARGET=1000

# Phase 1: JAX vs Original
while true; do
    if [ -f mismatches_jax_vs_original.npy ]; then
        count=$(python -c "import numpy as np; print(len(np.load('mismatches_jax_vs_original.npy')))")
        echo "Phase 1: $count / $TARGET"
        if [ "$count" -ge "$TARGET" ]; then
            echo "Phase 1 complete!"
            break
        fi
    fi
    echo "Running phase 1..."
    python -u mismatch_p1_resumable.py || true
done

# Phase 2: JAX vs TEOBResumS
while true; do
    if [ -f mismatches_jax_vs_teob.npy ]; then
        count=$(python -c "import numpy as np; print(len(np.load('mismatches_jax_vs_teob.npy')))")
        echo "Phase 2: $count / $TARGET"
        if [ "$count" -ge "$TARGET" ]; then
            echo "Phase 2 complete!"
            break
        fi
    fi
    echo "Running phase 2..."
    python -u mismatch_p2_resumable.py || true
done

# Phase 3: Plot
echo "Plotting..."
python mismatch_phase3_plot.py
echo "All done!"
