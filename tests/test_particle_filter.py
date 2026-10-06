"""Automated tests for the local 2D particle-filter localizer.

Run with:  python -m pytest tests/ -s
Every scenario prints its localization error and assertion outcome to the
terminal; no visualization windows are used.
"""

from __future__ import annotations

import numpy as np
import pytest

from particle_filter import ParticleFilter, circular_mean, wrap_to_pi
from particle_filter.simulation import (
    make_landmarks,
    noisy_controls,
    observe,
    simulate_trajectory,
)

DT = 0.1


def report(name: str, est, truth, pos_tol, ang_tol) -> None:
    """Print localization error to the terminal and assert on it."""
    pos_err = float(np.hypot(est[0] - truth[0], est[1] - truth[1]))
    ang_err = abs(float(wrap_to_pi(est[2] - truth[2])))
    print(
        f"\n[{name}] estimate=({est[0]:+.3f}, {est[1]:+.3f}, {est[2]:+.3f}) "
        f"truth=({truth[0]:+.3f}, {truth[1]:+.3f}, {truth[2]:+.3f}) "
        f"pos_err={pos_err:.4f} m (tol {pos_tol}) ang_err={np.degrees(ang_err):.2f} deg (tol {np.degrees(ang_tol):.1f})"
    )
    assert pos_err < pos_tol, f"{name}: position error {pos_err:.4f} >= {pos_tol}"
    assert ang_err < ang_tol, f"{name}: angle error {ang_err:.4f} >= {ang_tol}"
    print(f"[{name}] assertions PASSED")


def run_filter(controls, truth, landmarks, seed=7, n_particles=800,
               std_range=0.3, std_bearing=0.1, resample_method="systematic"):
    """Feed noisy controls + observations through a filter; return final estimate."""
    rng = np.random.default_rng(seed + 1000)
    pf = ParticleFilter(
        n_particles=n_particles,
        init_mean=tuple(truth[0]),
        init_std=(0.4, 0.4, 0.25),
        motion_noise=(0.08, 0.03),
        resample_method=resample_method,
        ess_threshold=0.5,
        seed=seed,
    )
    odom = noisy_controls(controls, rng)
    for t, control in enumerate(odom):
        pf.predict(control, DT)
        vis_lm, ranges, bearings = observe(
            truth[t + 1], landmarks, rng, std_range, std_bearing
        )
        pf.update(vis_lm, ranges, bearings, std_range, std_bearing)
        pf.maybe_resample()
    return pf, pf.estimate()


# --------------------------------------------------------------------- scenarios
def test_stationary():
    landmarks = make_landmarks(np.random.default_rng(1), n=8)
    controls = [(0.0, 0.0)] * 60
    truth = simulate_trajectory(controls, DT, start=(1.0, -2.0, 0.7))
    pf, est = run_filter(controls, truth, landmarks, seed=11)
    report("stationary", est, truth[-1], pos_tol=0.35, ang_tol=0.20)


def test_constant_velocity():
    landmarks = make_landmarks(np.random.default_rng(2), n=10)
    controls = [(1.0, 0.3)] * 80
    truth = simulate_trajectory(controls, DT, start=(0.0, 0.0, 0.0))
    pf, est = run_filter(controls, truth, landmarks, seed=22)
    report("constant_velocity", est, truth[-1], pos_tol=0.6, ang_tol=0.25)


def test_noisy_observations():
    """Larger observation noise must still converge, with looser tolerance."""
    landmarks = make_landmarks(np.random.default_rng(3), n=10)
    controls = [(0.8, -0.2)] * 80
    truth = simulate_trajectory(controls, DT, start=(-1.0, 1.0, 1.2))
    pf, est = run_filter(controls, truth, landmarks, seed=33,
                         std_range=0.6, std_bearing=0.25)
    report("noisy_observations", est, truth[-1], pos_tol=1.0, ang_tol=0.35)


def test_range_only_observations():
    landmarks = make_landmarks(np.random.default_rng(4), n=12)
    controls = [(0.6, 0.25)] * 80
    truth = simulate_trajectory(controls, DT, start=(0.5, 0.5, -0.5))
    rng = np.random.default_rng(55)
    pf = ParticleFilter(n_particles=1200, init_mean=tuple(truth[0]),
                        init_std=(0.4, 0.4, 0.3), motion_noise=(0.08, 0.03), seed=44)
    odom = noisy_controls(controls, rng)
    for t, control in enumerate(odom):
        pf.predict(control, DT)
        vis_lm, ranges, _ = observe(truth[t + 1], landmarks, rng, 0.3, None)
        pf.update(vis_lm, ranges, None, std_range=0.3)
        pf.maybe_resample()
    est = pf.estimate()
    report("range_only", est, truth[-1], pos_tol=1.0, ang_tol=1.0)


# ------------------------------------------------------------------- resampling
@pytest.mark.parametrize("method", ["systematic", "stratified"])
def test_resampling_methods(method):
    landmarks = make_landmarks(np.random.default_rng(5), n=8)
    controls = [(0.5, 0.4)] * 60
    truth = simulate_trajectory(controls, DT, start=(0.0, 0.0, 0.0))
    pf, est = run_filter(controls, truth, landmarks, seed=66, resample_method=method)
    print(f"\n[resample:{method}] resample_events={pf.resample_events} "
          f"ESS_final={pf.effective_sample_size():.1f}")
    assert pf.resample_events > 0, "resampling should have triggered"
    report(f"resample:{method}", est, truth[-1], pos_tol=0.8, ang_tol=0.3)


def test_ess_threshold_and_weight_reset():
    pf = ParticleFilter(n_particles=200, seed=5, ess_threshold=0.5)
    pf.weights = np.zeros(200)
    pf.weights[0] = 1.0
    ess = pf.effective_sample_size()
    print(f"\n[ess] degenerate ESS={ess:.2f} < threshold {0.5 * 200}")
    assert ess < 0.5 * 200
    assert pf.maybe_resample()
    assert np.allclose(pf.weights, 1.0 / 200), "weights reset to uniform after resample"
    print("[ess] assertions PASSED")


def test_weight_degeneracy_recovery():
    """Impossible observations drive all weights to zero -> uniform recovery."""
    pf = ParticleFilter(n_particles=300, seed=9)
    landmarks = np.array([[0.0, 0.0]])
    absurd_ranges = np.array([1e200])  # squared error overflows -> -inf log-likelihood
    pf.update(landmarks, absurd_ranges, None, std_range=1e-9)
    print(f"\n[degeneracy] degeneracy_events={pf.degeneracy_events} "
          f"weights_sum={pf.weights.sum():.6f}")
    assert pf.degeneracy_events == 1
    assert np.allclose(pf.weights, 1.0 / 300)
    assert np.isfinite(pf.weights).all()
    print("[degeneracy] assertions PASSED")


# ------------------------------------------------------------------ angle wrap
def test_angle_wrap_circular_mean():
    thetas = np.array([np.pi - 0.05, -np.pi + 0.05, np.pi - 0.02, -np.pi + 0.08])
    mean = circular_mean(thetas)
    naive = float(np.mean(thetas))
    print(f"\n[angle_wrap] circular_mean={mean:+.4f} naive_mean={naive:+.4f} (expected ~ +/-pi)")
    assert abs(abs(mean) - np.pi) < 0.1, "circular mean must stay near +/-pi"
    assert abs(abs(naive) - np.pi) > 0.5, "naive mean is wrong, confirming the test setup"
    wrapped = wrap_to_pi(np.array([3 * np.pi, -3 * np.pi, np.pi + 0.1]))
    assert np.all(wrapped <= np.pi) and np.all(wrapped >= -np.pi)
    print("[angle_wrap] assertions PASSED")


def test_localization_across_pi_boundary():
    """Track a robot rotating through the +/-pi heading boundary."""
    landmarks = make_landmarks(np.random.default_rng(6), n=10)
    controls = [(0.4, 0.8)] * 70
    truth = simulate_trajectory(controls, DT, start=(0.0, 0.0, np.pi - 0.3))
    assert np.any(truth[:, 2] < -3.0), "trajectory must cross the -pi boundary"
    pf, est = run_filter(controls, truth, landmarks, seed=77)
    report("angle_crossing", est, truth[-1], pos_tol=0.8, ang_tol=0.3)


# ------------------------------------------------------- seeding / persistence
def test_fixed_seed_reproducible():
    landmarks = make_landmarks(np.random.default_rng(7), n=8)
    controls = [(0.7, 0.2)] * 50
    truth = simulate_trajectory(controls, DT)
    _, est_a = run_filter(controls, truth, landmarks, seed=123)
    _, est_b = run_filter(controls, truth, landmarks, seed=123)
    print(f"\n[fixed_seed] run A={est_a} run B={est_b}")
    assert np.allclose(est_a, est_b, atol=1e-12), "same seed must reproduce exactly"
    print("[fixed_seed] assertions PASSED")


def test_state_save_restore(tmp_path):
    landmarks = make_landmarks(np.random.default_rng(8), n=8)
    controls = [(0.6, -0.3)] * 60
    truth = simulate_trajectory(controls, DT)
    rng = np.random.default_rng(900)
    odom = noisy_controls(controls, rng)
    observations = [observe(truth[t + 1], landmarks, rng, 0.3, 0.1)
                    for t in range(len(controls))]

    def step(pf, t):
        pf.predict(odom[t], DT)
        vis_lm, ranges, bearings = observations[t]
        pf.update(vis_lm, ranges, bearings, 0.3, 0.1)
        pf.maybe_resample()

    state_file = tmp_path / "filter_state.pkl"

    pf_full = ParticleFilter(n_particles=400, init_mean=tuple(truth[0]),
                             init_std=(0.4, 0.4, 0.25), motion_noise=(0.08, 0.03), seed=321)
    for t in range(len(controls)):
        step(pf_full, t)

    pf_split = ParticleFilter(n_particles=400, init_mean=tuple(truth[0]),
                              init_std=(0.4, 0.4, 0.25), motion_noise=(0.08, 0.03), seed=321)
    for t in range(30):
        step(pf_split, t)
    pf_split.save_state(state_file)
    pf_restored = ParticleFilter.load_state(state_file)
    for t in range(30, len(controls)):
        step(pf_restored, t)

    est_full = pf_full.estimate()
    est_restored = pf_restored.estimate()
    pos_err = float(np.hypot(est_restored[0] - truth[-1][0], est_restored[1] - truth[-1][1]))
    print(f"\n[save_restore] uninterrupted={est_full} restored={est_restored} "
          f"pos_err_vs_truth={pos_err:.4f} m")
    assert np.allclose(est_full, est_restored, atol=1e-12), \
        "restored filter must reproduce the uninterrupted run exactly"
    assert np.allclose(pf_full.particles, pf_restored.particles, atol=1e-15)
    print("[save_restore] assertions PASSED")
