#!/usr/bin/env python
"""
Validate the ROQ basis after build and package small files for Colab PE.

Usage (run after build_roq_basis_numpy.py finishes):
    python validate_and_package_roq.py

What it does:
    1. Loads the completed ROQ basis (linear + quadratic)
    2. Validates via random mismatch tests
    3. Packages only the small files needed for PE into a zip
    4. Prints summary and instructions

The output zip can be uploaded to Google Drive / Colab for GPU-accelerated PE.
"""

import os
import sys
import zipfile
import configparser
import numpy as np

# ── Configuration ──────────────────────────────────────────────────────
CONFIG_FILE = "config_roq_mlgw_bns_jax_gw170817.ini"
OUT_DIR = "./roq_basis_mlgw_bns_jax"
PACKAGE_NAME = "roq_pe_package.zip"
N_VALIDATION = 200  # number of random waveforms for mismatch test


def load_config():
    """Load ROQ build configuration."""
    config = configparser.ConfigParser()
    config.read(CONFIG_FILE)
    return config


def check_build_complete(out_dir):
    """Check that all expected output files exist."""
    required_files = {
        "linear": [
            "basis_linear.npy",
            "basis_waveform_params_linear.npy",
            "basis_interpolant_linear.npy",
            "empirical_frequencies_linear.npy",
            "empirical_nodes_linear.npy",
        ],
        "quadratic": [
            "basis_quadratic.npy",
            "basis_waveform_params_quadratic.npy",
            "basis_interpolant_quadratic.npy",
            "empirical_frequencies_quadratic.npy",
            "empirical_nodes_quadratic.npy",
        ],
    }

    missing = []
    for phase, files in required_files.items():
        phase_dir = os.path.join(out_dir, "ROQ_data", phase)
        for f in files:
            path = os.path.join(phase_dir, f)
            if not os.path.isfile(path):
                missing.append(path)

    return missing


def validate_basis(out_dir, config):
    """Run mismatch validation on the completed basis."""
    from JenpyROQ.linear_algebra import normalise_vector, scalar_product
    from mlgw_bns_roq_wrapper import WfMLGWBNS

    # Load config parameters
    fmin = float(config["Waveform_and_parametrisation"]["f-min"])
    fmax = float(config["Waveform_and_parametrisation"]["f-max"])
    seglen = float(config["Waveform_and_parametrisation"]["seglen"])
    tol_lin = float(config["ROQ"]["tolerance-lin"])
    tol_qua = float(config["ROQ"]["tolerance-qua"])
    deltaF = 1.0 / seglen

    # Load basis data
    lin_dir = os.path.join(out_dir, "ROQ_data", "linear")
    qua_dir = os.path.join(out_dir, "ROQ_data", "quadratic")

    B_lin = np.load(os.path.join(lin_dir, "basis_interpolant_linear.npy"))
    nodes_lin = np.load(os.path.join(lin_dir, "empirical_nodes_linear.npy"))
    B_qua = np.load(os.path.join(qua_dir, "basis_interpolant_quadratic.npy"))
    nodes_qua = np.load(os.path.join(qua_dir, "empirical_nodes_quadratic.npy"))

    n_freq = int((fmax - fmin) / deltaF) + 1

    print(f"\n{'='*60}")
    print(f"ROQ BASIS VALIDATION")
    print(f"{'='*60}")
    print(f"Frequency range    : [{fmin}, {fmax}] Hz")
    print(f"Segment length     : {seglen} s")
    print(f"Full grid points   : {n_freq}")
    print(f"Linear ROQ nodes   : {len(nodes_lin)} ({n_freq / len(nodes_lin):.0f}x speedup)")
    print(f"Quadratic ROQ nodes: {len(nodes_qua)} ({n_freq / len(nodes_qua):.0f}x speedup)")
    print(f"Tolerance (linear) : {tol_lin}")
    print(f"Tolerance (quad)   : {tol_qua}")

    # Training range for random parameter generation
    ranges = {
        "mc":      (float(config["Training_range"]["mc-min"]),      float(config["Training_range"]["mc-max"])),
        "q":       (float(config["Training_range"]["q-min"]),       float(config["Training_range"]["q-max"])),
        "s1z":     (float(config["Training_range"]["s1z-min"]),     float(config["Training_range"]["s1z-max"])),
        "s2z":     (float(config["Training_range"]["s2z-min"]),     float(config["Training_range"]["s2z-max"])),
        "lambda1": (float(config["Training_range"]["lambda1-min"]), float(config["Training_range"]["lambda1-max"])),
        "lambda2": (float(config["Training_range"]["lambda2-min"]), float(config["Training_range"]["lambda2-max"])),
        "iota":    (float(config["Training_range"]["iota-min"]),    float(config["Training_range"]["iota-max"])),
        "phiref":  (float(config["Training_range"]["phiref-min"]),  float(config["Training_range"]["phiref-max"])),
    }

    # Generate random test waveforms
    wf = WfMLGWBNS("mlgw-bns-jax")
    rng = np.random.default_rng(42)

    errors_lin = []
    errors_qua = []

    print(f"\nValidating with {N_VALIDATION} random waveforms...")

    for i in range(N_VALIDATION):
        # Random parameters from training range
        mc = rng.uniform(*ranges["mc"])
        q = rng.uniform(*ranges["q"])
        # Convert mc, q to m1, m2
        factor = mc * (1.0 + q) ** 0.2
        m1 = factor * q ** (-0.6)
        m2 = factor * q ** 0.4

        p = {
            "m1": m1, "m2": m2,
            "s1z": rng.uniform(*ranges["s1z"]),
            "s2z": rng.uniform(*ranges["s2z"]),
            "lambda1": rng.uniform(*ranges["lambda1"]),
            "lambda2": rng.uniform(*ranges["lambda2"]),
            "iota": rng.uniform(*ranges["iota"]),
            "phiref": rng.uniform(*ranges["phiref"]),
        }

        hp_full, _ = wf.generate_waveform(p, deltaF, fmin, fmax, 10.0)

        # Linear validation
        hp_norm = normalise_vector(hp_full, deltaF)
        hp_roq_lin = np.dot(B_lin, hp_norm[nodes_lin])
        residual_lin = hp_norm - hp_roq_lin
        eie_lin = np.real(scalar_product(residual_lin, residual_lin, deltaF))
        errors_lin.append(eie_lin)

        # Quadratic validation
        hp_qua = np.abs(hp_full) ** 2
        hp_qua_norm = normalise_vector(hp_qua, deltaF)
        hp_roq_qua = np.dot(B_qua, hp_qua_norm[nodes_qua])
        residual_qua = hp_qua_norm - hp_roq_qua
        eie_qua = np.real(scalar_product(residual_qua, residual_qua, deltaF))
        errors_qua.append(eie_qua)

        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{N_VALIDATION} done...")

    errors_lin = np.array(errors_lin)
    errors_qua = np.array(errors_qua)

    # Results
    print(f"\n{'─'*60}")
    print(f"LINEAR  — max error: {errors_lin.max():.2e}, "
          f"mean: {errors_lin.mean():.2e}, "
          f"tolerance: {tol_lin}")
    print(f"QUAD    — max error: {errors_qua.max():.2e}, "
          f"mean: {errors_qua.mean():.2e}, "
          f"tolerance: {tol_qua}")

    lin_ok = errors_lin.max() < tol_lin * 10  # allow 10x margin for unseen waveforms
    qua_ok = errors_qua.max() < tol_qua * 10

    if lin_ok and qua_ok:
        print(f"\n✓ VALIDATION PASSED")
    else:
        print(f"\n✗ VALIDATION FAILED — errors exceed tolerance")
        if not lin_ok:
            print(f"  Linear max error {errors_lin.max():.2e} > {tol_lin * 10:.2e}")
        if not qua_ok:
            print(f"  Quadratic max error {errors_qua.max():.2e} > {tol_qua * 10:.2e}")

    return lin_ok and qua_ok, errors_lin, errors_qua


def package_for_colab(out_dir, package_name):
    """Package only the small files needed for Colab PE into a zip."""
    # Files needed for PE (small: interpolants, nodes, frequencies)
    pe_files = [
        "ROQ_data/linear/basis_interpolant_linear.npy",
        "ROQ_data/linear/empirical_frequencies_linear.npy",
        "ROQ_data/linear/empirical_nodes_linear.npy",
        "ROQ_data/quadratic/basis_interpolant_quadratic.npy",
        "ROQ_data/quadratic/empirical_frequencies_quadratic.npy",
        "ROQ_data/quadratic/empirical_nodes_quadratic.npy",
        "ROQ_data/linear/basis_waveform_params_linear.npy",
        "ROQ_data/quadratic/basis_waveform_params_quadratic.npy",
    ]

    # Files NOT included (too large, not needed for PE):
    # - basis_linear.npy (~400+ MB)
    # - basis_quadratic.npy (~GB)
    # - preselection_* files

    total_size = 0
    with zipfile.ZipFile(package_name, "w", zipfile.ZIP_DEFLATED) as zf:
        for rel_path in pe_files:
            full_path = os.path.join(out_dir, rel_path)
            if os.path.isfile(full_path):
                zf.write(full_path, arcname=rel_path)
                size = os.path.getsize(full_path)
                total_size += size
                print(f"  + {rel_path} ({size / 1024:.1f} KB)")
            else:
                print(f"  ! MISSING: {rel_path}")

        # Also include the config for reference
        if os.path.isfile(CONFIG_FILE):
            zf.write(CONFIG_FILE, arcname="config.ini")
            print(f"  + config.ini")

    print(f"\nPackage: {package_name} ({total_size / 1024 / 1024:.1f} MB)")
    return package_name


def main():
    print("=" * 60)
    print("ROQ POST-BUILD: Validate & Package for Colab")
    print("=" * 60)

    # 1. Load config
    if not os.path.isfile(CONFIG_FILE):
        print(f"ERROR: Config file not found: {CONFIG_FILE}")
        sys.exit(1)
    config = load_config()

    # 2. Check build completeness
    print("\n[1/3] Checking build outputs...")
    missing = check_build_complete(OUT_DIR)
    if missing:
        print("ERROR: Build incomplete. Missing files:")
        for f in missing:
            print(f"  - {f}")
        print("\nThe ROQ build did not finish. Re-run: python build_roq_basis_numpy.py")
        sys.exit(1)
    print("  All output files present.")

    # 3. Validate
    print("\n[2/3] Validating ROQ basis...")
    passed, errors_lin, errors_qua = validate_basis(OUT_DIR, config)

    # Save validation results
    np.savez(
        os.path.join(OUT_DIR, "validation_results.npz"),
        errors_lin=errors_lin,
        errors_qua=errors_qua,
    )
    print(f"  Validation results saved to {OUT_DIR}/validation_results.npz")

    # 4. Package for Colab
    print(f"\n[3/3] Packaging small files for Colab PE...")
    package_for_colab(OUT_DIR, PACKAGE_NAME)

    # 5. Instructions
    print(f"\n{'='*60}")
    print("NEXT STEPS")
    print(f"{'='*60}")
    print(f"""
1. Upload '{PACKAGE_NAME}' to Google Drive

2. In Colab, mount Drive and unzip:
   from google.colab import drive
   drive.mount('/content/drive')
   !unzip /content/drive/MyDrive/{PACKAGE_NAME} -d roq_basis_mlgw_bns_jax/

3. Clone the repo and run the PE notebook:
   !git clone -b copilot/add-reduced-order-quadrature-model \\
       https://github.com/saulo-albuquerque-phys/mlgw_bns_jax.git
   # Open roq_pe_mlgw_bns_jax.ipynb and run from cell 1

The PE notebook will:
  - Load the ROQ interpolants and nodes from the zip
  - Compute ROQ weights using GW170817 data + PSD
  - Run SHARPy sampling with JAX on GPU
""")

    if not passed:
        print("⚠  WARNING: Validation had issues. Check errors before running PE.")
        sys.exit(1)


if __name__ == "__main__":
    main()
