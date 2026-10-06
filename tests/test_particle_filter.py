"""Automated tests for the local 2D particle filter.

All tests print localization errors / diagnostics to the terminal
(run with `pytest -s`) and assert on them. No visualization windows,
no ROS, no hardware, no external services.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from particle_filter import ParticleFilter, Simulator, LandmarkMap, wrap_angle
from particle_filter.particle_filter import circular_mean


def make_map(seed=7):
    return LandmarkMap.random(12, xlim=(-8, 8), ylim=(-8, 8), seed=seed)


def make_filter(seed=42, n_particles=400, **kwargs):
    kwargs.setdefault("init_std", (0.5, 0.5, 0.3))
    return ParticleFilter(n_particles=n_particles, seed=seed, **kwargs)


def run_scenario(controls, dt=0.1, seed=42, sim_seed=1, init_mean=(0, 0, 0),
                 with_bearing=True):
    pf = make_filter(seed=seed, init_mean=init_mean)
    sim = Simulator(make_map(), seed=sim_seed)
    return pf, sim, sim.run(pf, init_mean, controls, dt,
                            with_bearing=with_bearing)


def test_stationary_robot():
    controls = [(0.0, 0.0)] * 50
    _, _, res = run_scenario(controls)
    final = res["pos_error"][-1]
    print(f"\n[stationary] final pos error = {final:.4f} m, "
          f"final yaw error = {res['yaw_error'][-1]:.4f} rad")
    assert final < 0.5
    assert res["yaw_error"][-1] < 0.3


def test_constant_velocity():
    controls = [(1.0, 0.3)] * 100
    _, _, res = run_scenario(controls)
    mean_err = res["pos_error"][10:].mean()
    print(f"\n[constant velocity] mean pos error (after convergence) = "
          f"{mean_err:.4f} m, final = {res['pos_error'][-1]:.4f} m")
    assert mean_err < 0.6
    assert res["pos_error"][-1] < 0.8


def test_noisy_observations_range_only():
    # Large observation noise, range-only: filter must still track.
    pf = make_filter(seed=3)
    sim = Simulator(make_map(), range_std=0.8, seed=11)
    controls = [(0.8, 0.2)] * 80
    res = sim.run(pf, (0.0, 0.0, 0.0), controls, 0.1, with_bearing=False)
    mean_err = res["pos_error"][10:].mean()
    print(f"\n[noisy range-only obs] mean pos error = {mean_err:.4f} m")
    assert mean_err < 1.5


def test_resampling_triggers_and_helps():
    pf = make_filter(seed=5, ess_threshold=0.9)
    sim = Simulator(make_map(), seed=2)
    controls = [(1.0, 0.4)] * 60
    res = sim.run(pf, (0.0, 0.0, 0.0), controls, 0.1)
    ess = pf.effective_sample_size()
    print(f"\n[resampling] ESS after run = {ess:.1f} / {pf.n_particles}, "
          f"final pos error = {res['pos_error'][-1]:.4f} m")
    # After resampling weights are uniform, so ESS must be healthy.
    assert ess > 0.5 * pf.n_particles
    assert res["pos_error"][-1] < 0.8
    # Weights must remain a valid distribution.
    assert abs(pf.weights.sum() - 1.0) < 1e-9


def test_stratified_resampling():
    pf = make_filter(seed=6, resample_method="stratified")
    sim = Simulator(make_map(), seed=3)
    controls = [(0.5, -0.2)] * 50
    res = sim.run(pf, (0.0, 0.0, 0.0), controls, 0.1)
    print(f"\n[stratified] final pos error = {res['pos_error'][-1]:.4f} m")
    assert res["pos_error"][-1] < 0.8


def test_weight_degeneracy_recovery():
    pf = make_filter(seed=8)
    # Zero measurement std -> every Gaussian log-likelihood becomes
    # -inf -> all weights degenerate -> explicit recovery must kick in.
    landmarks = np.array([[3.0, 4.0]])
    obs = np.array([[5.0, 0.5]])
    pf.measurement_update(obs, landmarks, range_std=0.0, bearing_std=0.0)
    print(f"\n[degeneracy] weights after recovery: min={pf.weights.min():.2e}, "
          f"max={pf.weights.max():.2e}, sum={pf.weights.sum():.6f}")
    assert np.all(np.isfinite(pf.weights))
    assert np.all(pf.weights > 0)
    assert abs(pf.weights.sum() - 1.0) < 1e-9
    # Filter keeps working after recovery.
    sim = Simulator(make_map(), seed=4)
    res = sim.run(pf, (0.0, 0.0, 0.0), [(0.5, 0.1)] * 40, 0.1)
    print(f"[degeneracy] pos error after recovery run = "
          f"{res['pos_error'][-1]:.4f} m")
    assert res["pos_error"][-1] < 1.5


def test_angle_wrapping():
    # Particles straddling +/-pi must average near pi, not near 0.
    angles = np.array([np.pi - 0.1, -np.pi + 0.1, np.pi - 0.05, -np.pi + 0.05])
    mean = circular_mean(angles)
    print(f"\n[angle wrap] circular mean near pi = {mean:.4f} rad "
          f"(naive mean = {angles.mean():.4f})")
    assert abs(abs(mean) - np.pi) < 0.2
    # wrap_angle sanity checks.
    assert abs(wrap_angle(3 * np.pi) - np.pi) < 1e-9 or \
        abs(wrap_angle(3 * np.pi) + np.pi) < 1e-9
    assert abs(wrap_angle(np.pi + 0.1) - (-np.pi + 0.1)) < 1e-9
    # Filter estimate near the boundary: initialize at theta ~ pi.
    pf = make_filter(seed=9, init_mean=(0.0, 0.0, np.pi - 0.05),
                     init_std=(0.3, 0.3, 0.1))
    sim = Simulator(make_map(), seed=5)
    controls = [(0.0, 0.0)] * 30
    res = sim.run(pf, (0.0, 0.0, np.pi - 0.05), controls, 0.1)
    print(f"[angle wrap] yaw error near pi = {res['yaw_error'][-1]:.4f} rad")
    assert res["yaw_error"][-1] < 0.3


def test_fixed_seed_determinism():
    controls = [(1.0, 0.3)] * 40
    _, _, res1 = run_scenario(controls, seed=123, sim_seed=10)
    _, _, res2 = run_scenario(controls, seed=123, sim_seed=10)
    print(f"\n[fixed seed] max estimate diff between identical runs = "
          f"{np.abs(res1['estimates'] - res2['estimates']).max():.2e}")
    np.testing.assert_array_equal(res1["estimates"], res2["estimates"])
    # A different seed should (almost surely) differ.
    _, _, res3 = run_scenario(controls, seed=124, sim_seed=10)
    assert not np.array_equal(res1["estimates"], res3["estimates"])


def test_state_save_restore(tmp_path):
    state_file = tmp_path / "pf_state.pkl"
    controls = [(1.0, 0.3)] * 60
    dt = 0.1

    # Pre-generate the full observation stream with one simulator so the
    # uninterrupted run and the restored run see identical data.
    sim = Simulator(make_map(), seed=20)
    pose = np.zeros(3)
    obs_stream = []
    for control in controls:
        pose = sim.step_truth(pose, control, dt)
        obs_stream.append(sim.observe(pose, with_bearing=True))

    def drive(pf, control_subset, obs_subset):
        for control, (obs, lms) in zip(control_subset, obs_subset):
            pf.motion_update(control, dt)
            pf.measurement_update(obs, lms, sim.range_std, sim.bearing_std)
            pf.maybe_resample()
        return pf.estimate()

    # Uninterrupted reference run.
    pf_full = make_filter(seed=77)
    est_full = drive(pf_full, controls, obs_stream)

    # Split run: first half, save, restore, second half.
    pf_split = make_filter(seed=77)
    drive(pf_split, controls[:30], obs_stream[:30])
    pf_split.save_state(state_file)

    pf_restored = ParticleFilter.load_state(state_file)
    np.testing.assert_array_equal(pf_restored.particles, pf_split.particles)
    np.testing.assert_array_equal(pf_restored.weights, pf_split.weights)
    est_restored = drive(pf_restored, controls[30:], obs_stream[30:])

    diff = np.abs(est_full - est_restored).max()
    print(f"\n[save/restore] full vs restored estimate diff = {diff:.2e}")
    np.testing.assert_array_equal(est_full, est_restored)

    # Restoring twice from the same file must also reproduce exactly.
    pf_again = ParticleFilter.load_state(state_file)
    est_again = drive(pf_again, controls[30:], obs_stream[30:])
    np.testing.assert_array_equal(est_restored, est_again)
