#!/usr/bin/env python3
"""
Tests for the JAX ROQ basis builder.

These tests verify each component of the ROQ pipeline independently
and then run a small end-to-end build to confirm correctness.
"""

import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

# Ensure we can import from the repo root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.setdefault("JAX_PLATFORMS", "cpu")
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from roq_builder_jax import (
    ROQConfig,
    corner_parameters,
    empirical_interpolation,
    generate_waveforms_batch,
    gram_schmidt_add,
    greedy_basis,
    greedy_basis_streaming,
    normalise,
    projection_error,
    projection_error_jax,
    sample_parameters,
    validate_basis,
    build_roq_basis,
    warmup_jit,
    _load_predictor,
    _save_phase_status,
    _phase_completed,
)


# ─────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────

@pytest.fixture
def small_cfg():
    """A small ROQ config for quick testing."""
    return ROQConfig(
        f_min=30.0,
        f_max=100.0,
        seglen=4.0,
        tolerance_lin=1e-2,
        tolerance_qua=1e-3,
        n_pre_basis_lin=10,
        n_pre_basis_qua=3,
        n_pre_basis_search_iter=5,
        n_training_set_cycles=1,
        training_set_sizes=[50],
        output_dir=tempfile.mkdtemp(prefix="roq_test_"),
        model_path=os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..", "mlgw_bns_jax_model.h5"
        ),
        random_seed=42,
        verbose=1,
        waveform_batch_size=4,
        projection_batch_size=20,
    )


@pytest.fixture
def rng():
    return np.random.default_rng(42)


# ─────────────────────────────────────────────────────────────────────
# Unit tests
# ─────────────────────────────────────────────────────────────────────

class TestNormalise:
    def test_normalised_norm_is_one(self):
        delta_f = 0.5
        h = np.random.default_rng(0).standard_normal((5, 100)) + 0j
        h_norm = normalise(h, delta_f)
        norms = delta_f * np.sum(np.abs(h_norm) ** 2, axis=-1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-12)

    def test_normalise_zero_vector(self):
        delta_f = 1.0
        h = np.zeros((1, 50), dtype=complex)
        h_norm = normalise(h, delta_f)
        # Should not produce NaN or inf
        assert np.all(np.isfinite(h_norm))

    def test_single_vector(self):
        delta_f = 0.25
        h = np.array([1.0 + 2j, 3.0 - 1j, 0.5 + 0.5j])
        h_norm = normalise(h.reshape(1, -1), delta_f)
        norm = delta_f * np.sum(np.abs(h_norm) ** 2)
        np.testing.assert_allclose(norm, 1.0, atol=1e-14)


class TestGramSchmidt:
    def test_orthonormality(self):
        delta_f = 0.5
        rng = np.random.default_rng(123)
        vecs = rng.standard_normal((5, 50)) + 1j * rng.standard_normal((5, 50))

        basis = np.empty((0, 50), dtype=complex)
        for v in vecs:
            basis = gram_schmidt_add(basis, v, delta_f)

        # Check orthonormality
        gram = delta_f * (np.conj(basis) @ basis.T)
        np.testing.assert_allclose(gram, np.eye(5), atol=1e-12)

    def test_single_vector_basis(self):
        delta_f = 1.0
        v = np.array([1.0, 2.0, 3.0]) + 0j
        basis = gram_schmidt_add(np.empty((0, 3), dtype=complex), v, delta_f)
        assert basis.shape == (1, 3)
        norm = delta_f * np.sum(np.abs(basis[0]) ** 2)
        np.testing.assert_allclose(norm, 1.0, atol=1e-14)


class TestProjectionError:
    def test_zero_error_for_basis_element(self):
        delta_f = 0.5
        rng = np.random.default_rng(456)
        vecs = rng.standard_normal((3, 30)) + 1j * rng.standard_normal((3, 30))
        basis = np.empty((0, 30), dtype=complex)
        for v in vecs:
            basis = gram_schmidt_add(basis, v, delta_f)

        # Project basis elements should have zero error
        errors = projection_error(basis, basis, delta_f)
        np.testing.assert_allclose(errors, 0.0, atol=1e-12)

    def test_positive_error_for_external(self):
        delta_f = 1.0
        basis = np.array([[1, 0, 0], [0, 1, 0]], dtype=complex)
        # Normalise
        basis = normalise(basis, delta_f)
        h = np.array([[0, 0, 1]], dtype=complex)
        h = normalise(h, delta_f)
        errors = projection_error(h, basis, delta_f)
        np.testing.assert_allclose(errors, 1.0, atol=1e-12)

    def test_jax_matches_numpy(self):
        delta_f = 0.25
        rng = np.random.default_rng(789)
        basis = rng.standard_normal((5, 40)) + 1j * rng.standard_normal((5, 40))
        basis_orth = np.empty((0, 40), dtype=complex)
        for v in basis:
            basis_orth = gram_schmidt_add(basis_orth, v, delta_f)

        h = rng.standard_normal((10, 40)) + 1j * rng.standard_normal((10, 40))
        h = normalise(h, delta_f)

        err_np = projection_error(h, basis_orth, delta_f)
        err_jax = np.array(projection_error_jax(
            jnp.array(h), jnp.array(basis_orth), delta_f
        ))
        np.testing.assert_allclose(err_np, err_jax, atol=1e-12)


class TestParameterSampling:
    def test_sample_parameters_shape(self):
        cfg = ROQConfig()
        rng = np.random.default_rng(0)
        params = sample_parameters(rng, 100, cfg)
        assert params.shape == (100, 8)

    def test_sample_parameters_in_range(self):
        cfg = ROQConfig()
        rng = np.random.default_rng(0)
        params = sample_parameters(rng, 1000, cfg)
        assert np.all(params[:, 0] >= cfg.mc_range[0])
        assert np.all(params[:, 0] <= cfg.mc_range[1])
        assert np.all(params[:, 1] >= cfg.q_range[0])
        assert np.all(params[:, 1] <= cfg.q_range[1])

    def test_corner_parameters(self):
        cfg = ROQConfig()
        corners = corner_parameters(cfg)
        assert corners.shape == (64, 8)
        # All corners should have iota=0, phiref=0
        np.testing.assert_allclose(corners[:, 6], 0.0)
        np.testing.assert_allclose(corners[:, 7], 0.0)


# ─────────────────────────────────────────────────────────────────────
# Integration tests (require model file)
# ─────────────────────────────────────────────────────────────────────

MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "mlgw_bns_jax_model.h5"
)
HAS_MODEL = os.path.isfile(MODEL_PATH)


@pytest.mark.skipif(not HAS_MODEL, reason="mlgw_bns_jax_model.h5 not found")
class TestWaveformGeneration:
    def test_generate_single_waveform(self):
        from roq_builder_jax import _load_predictor
        predict_fn = _load_predictor(MODEL_PATH)
        cfg = ROQConfig(f_min=30.0, f_max=50.0, seglen=4.0)
        params = np.array([[1.19, 1.0, 0.0, 0.0, 300.0, 300.0, 0.0, 0.0]])
        wf = generate_waveforms_batch(
            predict_fn, params, cfg.frequencies,
            batch_size=4,
        )
        assert wf.shape == (1, cfg.n_freq)
        assert np.all(np.isfinite(wf))
        assert np.any(np.abs(wf) > 0)

    def test_generate_batch(self):
        from roq_builder_jax import _load_predictor
        predict_fn = _load_predictor(MODEL_PATH)
        cfg = ROQConfig(f_min=30.0, f_max=50.0, seglen=4.0)
        rng = np.random.default_rng(0)
        params = sample_parameters(rng, 8, cfg)
        wf = generate_waveforms_batch(
            predict_fn, params, cfg.frequencies,
            batch_size=4,
        )
        assert wf.shape == (8, cfg.n_freq)
        assert np.all(np.isfinite(wf))


@pytest.mark.skipif(not HAS_MODEL, reason="mlgw_bns_jax_model.h5 not found")
class TestGreedyBasis:
    def test_greedy_builds_basis(self):
        from roq_builder_jax import _load_predictor
        predict_fn = _load_predictor(MODEL_PATH)
        cfg = ROQConfig(f_min=30.0, f_max=50.0, seglen=4.0)
        rng = np.random.default_rng(0)
        params = sample_parameters(rng, 50, cfg)
        wf = generate_waveforms_batch(
            predict_fn, params, cfg.frequencies,
            batch_size=32,
        )
        wf = normalise(wf, cfg.delta_f)

        basis, bp, errors = greedy_basis(
            wf, params, cfg.delta_f, tolerance=0.1,
            max_basis=20, verbose=0,
        )
        assert basis.shape[1] == cfg.n_freq
        assert len(basis) <= 20
        assert len(bp) == len(basis)

        # Basis should be orthonormal
        gram = cfg.delta_f * (np.conj(basis) @ basis.T)
        np.testing.assert_allclose(gram, np.eye(len(basis)), atol=1e-10)


@pytest.mark.skipif(not HAS_MODEL, reason="mlgw_bns_jax_model.h5 not found")
class TestEIM:
    def test_eim_reconstruction(self):
        """Test that EIM reconstruction is accurate for basis elements."""
        from roq_builder_jax import _load_predictor
        predict_fn = _load_predictor(MODEL_PATH)
        cfg = ROQConfig(f_min=30.0, f_max=50.0, seglen=4.0)
        rng = np.random.default_rng(0)
        params = sample_parameters(rng, 50, cfg)
        wf = generate_waveforms_batch(
            predict_fn, params, cfg.frequencies,
            batch_size=32,
        )
        wf = normalise(wf, cfg.delta_f)

        basis, bp, _ = greedy_basis(
            wf, params, cfg.delta_f, tolerance=0.1,
            max_basis=10, verbose=0,
        )

        nodes, interpolant = empirical_interpolation(basis, cfg.delta_f, verbose=0)

        assert len(nodes) == len(basis)
        assert interpolant.shape == (cfg.n_freq, len(basis))

        # Reconstruction of basis elements should be exact
        for i in range(len(basis)):
            h = basis[i]
            h_approx = interpolant @ h[nodes]
            err = cfg.delta_f * np.sum(np.abs(h - h_approx) ** 2)
            assert err < 1e-10, f"Basis element {i} reconstruction error: {err}"


@pytest.mark.skipif(not HAS_MODEL, reason="mlgw_bns_jax_model.h5 not found")
class TestEndToEnd:
    def test_small_build(self, small_cfg):
        """End-to-end test: build a small linear basis."""
        results = build_roq_basis(small_cfg, kind="linear")
        assert "basis" in results
        assert "nodes" in results
        assert "interpolant" in results
        assert len(results["basis"]) > 0
        assert len(results["nodes"]) == len(results["basis"])

        # Check output files exist
        out_dir = os.path.join(small_cfg.output_dir, "ROQ_data", "linear")
        assert os.path.isfile(os.path.join(out_dir, "basis_linear.npy"))
        assert os.path.isfile(os.path.join(out_dir, "empirical_nodes_linear.npy"))
        assert os.path.isfile(os.path.join(out_dir, "basis_interpolant_linear.npy"))


@pytest.mark.skipif(not HAS_MODEL, reason="mlgw_bns_jax_model.h5 not found")
class TestConfigFromIni:
    def test_load_ini(self):
        ini_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..", "config_roq_mlgw_bns_jax_gw170817.ini"
        )
        if os.path.isfile(ini_path):
            cfg = ROQConfig.from_ini(ini_path)
            assert cfg.f_min == 23.0
            assert cfg.f_max == 2000.0
            assert cfg.seglen == 128.0
            assert cfg.tolerance_lin == 1e-4


# ─────────────────────────────────────────────────────────────────────
# Resume / checkpoint tests
# ─────────────────────────────────────────────────────────────────────

class TestPhaseStatus:
    """Test the _save_phase_status / _phase_completed helpers."""

    def test_save_and_check(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            assert not _phase_completed(p, "linear", "preselection")
            _save_phase_status(p, "linear", "preselection", {"basis_size": 42})
            assert _phase_completed(p, "linear", "preselection")
            assert not _phase_completed(p, "linear", "enrichment")

    def test_multiple_phases(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            _save_phase_status(p, "linear", "preselection")
            _save_phase_status(p, "linear", "enrichment")
            _save_phase_status(p, "linear", "eim")
            assert _phase_completed(p, "linear", "preselection")
            assert _phase_completed(p, "linear", "enrichment")
            assert _phase_completed(p, "linear", "eim")
            # quadratic should be independent
            assert not _phase_completed(p, "quadratic", "preselection")

    def test_nonexistent_dir(self):
        p = Path("/tmp/roq_test_does_not_exist_xyz")
        assert not _phase_completed(p, "linear", "preselection")


@pytest.mark.skipif(not HAS_MODEL, reason="mlgw_bns_jax_model.h5 not found")
class TestResume:
    def test_resume_skips_completed_phases(self, small_cfg):
        """Build once, then resume — second call should reload from disk."""
        # First build
        results1 = build_roq_basis(small_cfg, kind="linear", resume=False)
        basis_size_1 = len(results1["basis"])
        assert basis_size_1 > 0

        # Status files should exist
        out_dir = Path(small_cfg.output_dir) / "ROQ_data" / "linear"
        assert _phase_completed(out_dir, "linear", "preselection")
        assert _phase_completed(out_dir, "linear", "enrichment")
        assert _phase_completed(out_dir, "linear", "eim")

        # Resume should load from disk and return same results
        results2 = build_roq_basis(small_cfg, kind="linear", resume=True)
        np.testing.assert_array_equal(results1["basis"], results2["basis"])
        np.testing.assert_array_equal(results1["nodes"], results2["nodes"])

    def test_resume_after_preselection_only(self, small_cfg):
        """Simulate crash after preselection: only mark preselection done."""
        # Do a full build first to generate checkpoint files
        results_full = build_roq_basis(small_cfg, kind="linear", resume=False)

        # Now clear enrichment + eim markers to simulate crash after preselection
        out_dir = Path(small_cfg.output_dir) / "ROQ_data" / "linear"
        import json as _json
        status_path = out_dir / "_status_linear.json"
        with open(status_path) as f:
            status = _json.load(f)
        # Keep only preselection
        status = {"preselection": status["preselection"]}
        with open(status_path, "w") as f:
            _json.dump(status, f)

        # Resume should redo enrichment + EIM but skip preselection
        results_resumed = build_roq_basis(small_cfg, kind="linear", resume=True)
        assert len(results_resumed["basis"]) > 0
        assert len(results_resumed["nodes"]) > 0
        # Preselection basis should be the same since it was loaded
        pre_basis_orig = np.load(out_dir / "preselection_linear_basis.npy")
        assert pre_basis_orig.shape[0] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
