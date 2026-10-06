"""Core 2D particle filter.

State per particle: (x, y, theta).
Control input: (v, omega) -- linear velocity [m/s] and yaw rate [rad/s].
Observations: landmark range or range+bearing measurements.

Everything lives in memory or local files; no ROS, hardware, map servers
or external services are used.
"""

from __future__ import annotations

import pickle

import numpy as np

TWO_PI = 2.0 * np.pi


def wrap_angle(angle):
    """Wrap angle(s) to (-pi, pi]."""
    return (np.asarray(angle) + np.pi) % TWO_PI - np.pi


def circular_mean(angles, weights=None):
    """Mean of angles that is correct across the +/-pi boundary.

    Uses the vector average atan2(sum(w*sin), sum(w*cos)) instead of a
    plain arithmetic mean, so particles near +/-pi do not average to ~0.
    """
    angles = np.asarray(angles, dtype=float)
    if weights is None:
        s = np.sin(angles).sum()
        c = np.cos(angles).sum()
    else:
        w = np.asarray(weights, dtype=float)
        s = (w * np.sin(angles)).sum()
        c = (w * np.cos(angles)).sum()
    return float(np.arctan2(s, c))


class ParticleFilter:
    """Particle filter for 2D pose (x, y, theta) localization."""

    def __init__(
        self,
        n_particles=500,
        init_mean=(0.0, 0.0, 0.0),
        init_std=(0.5, 0.5, 0.3),
        motion_noise=(0.05, 0.02),
        resample_method="systematic",
        ess_threshold=0.5,
        seed=None,
    ):
        if resample_method not in ("systematic", "stratified"):
            raise ValueError(f"unknown resample_method: {resample_method}")
        self.n_particles = int(n_particles)
        self.motion_std_v = float(motion_noise[0])
        self.motion_std_omega = float(motion_noise[1])
        self.resample_method = resample_method
        self.ess_threshold = float(ess_threshold)
        self.rng = np.random.default_rng(seed)

        mean = np.asarray(init_mean, dtype=float)
        std = np.asarray(init_std, dtype=float)
        self.particles = self.rng.normal(mean, std, size=(self.n_particles, 3))
        self.particles[:, 2] = wrap_angle(self.particles[:, 2])
        self.weights = np.full(self.n_particles, 1.0 / self.n_particles)

    # ------------------------------------------------------------------
    # Motion update
    # ------------------------------------------------------------------
    def motion_update(self, control, dt):
        """Propagate particles with a noisy velocity motion model.

        control = (v, omega); noise std scales with the commanded
        velocities so that a stationary robot stays (almost) stationary.
        """
        v, omega = float(control[0]), float(control[1])
        noisy_v = v + self.rng.normal(0.0, self.motion_std_v, self.n_particles)
        noisy_omega = omega + self.rng.normal(
            0.0, self.motion_std_omega, self.n_particles
        )

        theta = self.particles[:, 2]
        # Exact arc integration for non-zero turn, straight line otherwise.
        small = np.abs(noisy_omega) < 1e-9
        new_theta = theta + noisy_omega * dt

        dx = np.empty(self.n_particles)
        dy = np.empty(self.n_particles)
        ratio = np.empty(self.n_particles)
        ratio[small] = noisy_v[small] * dt
        nz = ~small
        ratio[nz] = noisy_v[nz] / noisy_omega[nz] * np.sin(noisy_omega[nz] * dt)
        dx[small] = ratio[small] * np.cos(theta[small])
        dy[small] = ratio[small] * np.sin(theta[small])
        dx[nz] = noisy_v[nz] / noisy_omega[nz] * (
            np.sin(new_theta[nz]) - np.sin(theta[nz])
        )
        dy[nz] = noisy_v[nz] / noisy_omega[nz] * (
            -np.cos(new_theta[nz]) + np.cos(theta[nz])
        )

        self.particles[:, 0] += dx
        self.particles[:, 1] += dy
        self.particles[:, 2] = wrap_angle(new_theta)

    # ------------------------------------------------------------------
    # Measurement update
    # ------------------------------------------------------------------
    def measurement_update(self, observations, landmarks, range_std, bearing_std=None):
        """Weight particles by landmark observation likelihood.

        observations: (M,) ranges or (M, 2) [range, bearing] rows.
        landmarks:    (M, 2) landmark (x, y) positions, aligned with rows.

        Log-likelihoods are accumulated and the maximum is subtracted
        before exponentiation, which avoids probability underflow.
        If every weight degenerates to zero, a uniform-weight recovery
        with small pose jitter is applied.
        """
        observations = np.asarray(observations, dtype=float)
        if observations.ndim == 1:
            observations = observations.reshape(-1, 1)
        landmarks = np.atleast_2d(np.asarray(landmarks, dtype=float))
        if observations.shape[0] != landmarks.shape[0]:
            raise ValueError("observations and landmarks must have same length")
        use_bearing = observations.shape[1] == 2 and bearing_std is not None

        log_w = np.zeros(self.n_particles)
        with np.errstate(divide="ignore", invalid="ignore"):
            for obs, lm in zip(observations, landmarks):
                dx = lm[0] - self.particles[:, 0]
                dy = lm[1] - self.particles[:, 1]
                pred_range = np.hypot(dx, dy)
                err_r = (obs[0] - pred_range) / range_std
                log_w += -0.5 * err_r**2
                if use_bearing:
                    pred_bearing = np.arctan2(dy, dx) - self.particles[:, 2]
                    err_b = wrap_angle(obs[1] - pred_bearing) / bearing_std
                    log_w += -0.5 * err_b**2

        # Numerically stable softmax: subtract max before exp.
        max_log = np.max(log_w)
        if not np.isfinite(max_log):
            self._recover_from_degeneracy()
            return
        w = np.exp(log_w - max_log)
        total = w.sum()
        if total <= 0.0 or not np.isfinite(total):
            self._recover_from_degeneracy()
            return
        self.weights = w / total

    def _recover_from_degeneracy(self):
        """Explicit recovery when all weights collapse to zero.

        Reset to uniform weights and inject small pose jitter so the
        particle set regains diversity instead of staying stuck.
        """
        self.weights = np.full(self.n_particles, 1.0 / self.n_particles)
        jitter = self.rng.normal(
            0.0, [0.1, 0.1, 0.05], size=(self.n_particles, 3)
        )
        self.particles += jitter
        self.particles[:, 2] = wrap_angle(self.particles[:, 2])

    # ------------------------------------------------------------------
    # Resampling
    # ------------------------------------------------------------------
    def effective_sample_size(self):
        return 1.0 / np.sum(self.weights**2)

    def maybe_resample(self):
        """Resample only when ESS drops below threshold * N. Returns bool."""
        if self.effective_sample_size() < self.ess_threshold * self.n_particles:
            self.resample()
            return True
        return False

    def resample(self):
        if self.resample_method == "systematic":
            indexes = self._systematic_indexes()
        else:
            indexes = self._stratified_indexes()
        self.particles = self.particles[indexes]
        self.weights = np.full(self.n_particles, 1.0 / self.n_particles)

    def _systematic_indexes(self):
        positions = (self.rng.random() + np.arange(self.n_particles)) / self.n_particles
        return self._indexes_from_positions(positions)

    def _stratified_indexes(self):
        positions = (self.rng.random(self.n_particles) + np.arange(self.n_particles)) / self.n_particles
        return self._indexes_from_positions(positions)

    def _indexes_from_positions(self, positions):
        cumulative = np.cumsum(self.weights)
        cumulative[-1] = 1.0  # guard against float rounding
        return np.searchsorted(cumulative, positions)

    # ------------------------------------------------------------------
    # Estimation
    # ------------------------------------------------------------------
    def estimate(self):
        """Weighted pose estimate with circular mean for the heading."""
        x = float(np.sum(self.weights * self.particles[:, 0]))
        y = float(np.sum(self.weights * self.particles[:, 1]))
        theta = circular_mean(self.particles[:, 2], self.weights)
        return np.array([x, y, theta])

    # ------------------------------------------------------------------
    # Persistence (local files only)
    # ------------------------------------------------------------------
    def save_state(self, path):
        """Save full filter state (particles, weights, RNG state) to a file."""
        state = {
            "n_particles": self.n_particles,
            "motion_std_v": self.motion_std_v,
            "motion_std_omega": self.motion_std_omega,
            "resample_method": self.resample_method,
            "ess_threshold": self.ess_threshold,
            "particles": self.particles,
            "weights": self.weights,
            "rng_state": self.rng.bit_generator.state,
        }
        with open(path, "wb") as fh:
            pickle.dump(state, fh)

    @classmethod
    def load_state(cls, path):
        """Restore a filter from save_state(); continued runs reproduce
        deterministically because the RNG state is restored too."""
        with open(path, "rb") as fh:
            state = pickle.load(fh)
        pf = cls(
            n_particles=state["n_particles"],
            motion_noise=(state["motion_std_v"], state["motion_std_omega"]),
            resample_method=state["resample_method"],
            ess_threshold=state["ess_threshold"],
        )
        pf.particles = state["particles"]
        pf.weights = state["weights"]
        pf.rng.bit_generator.state = state["rng_state"]
        return pf
