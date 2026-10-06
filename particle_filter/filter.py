"""Core particle filter: weighting, normalization, resampling, save/restore."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

from .models import (
    circular_mean,
    motion_update,
    range_bearing_log_likelihood,
    range_log_likelihood,
)


class ParticleFilter:
    """2D (x, y, theta) particle filter localizer.

    - Weights are accumulated in log-space to avoid probability underflow.
    - If every weight degenerates to zero, weights are reset to uniform
      (explicit recovery strategy) and the event is recorded.
    - Supports 'systematic' and 'stratified' resampling with an
      effective-sample-size (ESS) threshold.
    """

    def __init__(
        self,
        n_particles: int = 500,
        init_mean: tuple[float, float, float] = (0.0, 0.0, 0.0),
        init_std: tuple[float, float, float] = (0.5, 0.5, 0.3),
        motion_noise: tuple[float, float] = (0.05, 0.02),
        resample_method: str = "systematic",
        ess_threshold: float = 0.5,
        seed: int | None = None,
    ):
        if resample_method not in ("systematic", "stratified"):
            raise ValueError(f"unknown resample_method: {resample_method}")
        self.n_particles = int(n_particles)
        self.motion_noise = tuple(motion_noise)
        self.resample_method = resample_method
        self.ess_threshold = float(ess_threshold)
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        init_mean = np.asarray(init_mean, dtype=float)
        init_std = np.asarray(init_std, dtype=float)
        self.particles = init_mean[None, :] + self.rng.normal(
            0.0, init_std[None, :], size=(self.n_particles, 3)
        )
        self.weights = np.full(self.n_particles, 1.0 / self.n_particles)
        self.degeneracy_events = 0
        self.resample_events = 0

    # ------------------------------------------------------------------ steps
    def predict(self, control: tuple[float, float], dt: float) -> None:
        """Motion update with control (v, omega) and process noise."""
        self.particles = motion_update(
            self.particles, control, dt, self.motion_noise, self.rng
        )

    def update(
        self,
        landmarks: np.ndarray,
        ranges: np.ndarray,
        bearings: np.ndarray | None = None,
        std_range: float = 0.3,
        std_bearing: float = 0.1,
    ) -> None:
        """Measurement update from landmark ranges (and optional bearings)."""
        landmarks = np.asarray(landmarks, dtype=float)
        if bearings is None:
            log_lik = range_log_likelihood(self.particles, landmarks, ranges, std_range)
        else:
            log_lik = range_bearing_log_likelihood(
                self.particles, landmarks, ranges, bearings, std_range, std_bearing
            )
        self._apply_log_likelihood(log_lik)

    def _apply_log_likelihood(self, log_lik: np.ndarray) -> None:
        """Multiply likelihoods into weights in log-space, then normalize."""
        log_lik = np.asarray(log_lik, dtype=float)
        with np.errstate(under="ignore", invalid="ignore"):
            log_w = np.log(self.weights) + log_lik
        finite = np.isfinite(log_w)
        if not np.any(finite):
            # Recovery strategy: every weight degenerated to zero -> restart
            # from a uniform distribution instead of propagating NaNs.
            self.weights = np.full(self.n_particles, 1.0 / self.n_particles)
            self.degeneracy_events += 1
            return
        # Shift by the max so exp() cannot underflow for the best particle.
        log_w = log_w - np.max(log_w[finite])
        w = np.where(finite, np.exp(log_w), 0.0)
        total = w.sum()
        if not np.isfinite(total) or total <= 0.0:
            self.weights = np.full(self.n_particles, 1.0 / self.n_particles)
            self.degeneracy_events += 1
            return
        self.weights = w / total

    # ------------------------------------------------------------- resampling
    def effective_sample_size(self) -> float:
        return float(1.0 / np.sum(self.weights * self.weights))

    def maybe_resample(self) -> bool:
        """Resample when ESS drops below ess_threshold * N. Returns True if done."""
        if self.effective_sample_size() < self.ess_threshold * self.n_particles:
            self.resample()
            return True
        return False

    def resample(self) -> None:
        if self.resample_method == "systematic":
            indexes = self._systematic_indexes()
        else:
            indexes = self._stratified_indexes()
        self.particles = self.particles[indexes]
        self.weights = np.full(self.n_particles, 1.0 / self.n_particles)
        self.resample_events += 1

    def _systematic_indexes(self) -> np.ndarray:
        n = self.n_particles
        positions = (self.rng.uniform() + np.arange(n)) / n
        return self._indexes_from_positions(positions)

    def _stratified_indexes(self) -> np.ndarray:
        n = self.n_particles
        positions = (self.rng.uniform(size=n) + np.arange(n)) / n
        return self._indexes_from_positions(positions)

    def _indexes_from_positions(self, positions: np.ndarray) -> np.ndarray:
        cumulative = np.cumsum(self.weights)
        cumulative[-1] = 1.0  # guard against float rounding
        return np.searchsorted(cumulative, positions, side="left")

    # --------------------------------------------------------------- estimate
    def estimate(self) -> tuple[float, float, float]:
        """Weighted mean pose; heading uses the circular mean (wrap-safe)."""
        x = float(np.sum(self.weights * self.particles[:, 0]))
        y = float(np.sum(self.weights * self.particles[:, 1]))
        theta = circular_mean(self.particles[:, 2], self.weights)
        return x, y, theta

    # ---------------------------------------------------------- save/restore
    def save_state(self, path: str | Path) -> None:
        """Persist the full filter state (particles, weights, RNG) to a file."""
        state = {
            "n_particles": self.n_particles,
            "motion_noise": self.motion_noise,
            "resample_method": self.resample_method,
            "ess_threshold": self.ess_threshold,
            "seed": self.seed,
            "particles": self.particles,
            "weights": self.weights,
            "rng_state": self.rng.bit_generator.state,
            "degeneracy_events": self.degeneracy_events,
            "resample_events": self.resample_events,
        }
        with open(path, "wb") as fh:
            pickle.dump(state, fh)

    @classmethod
    def load_state(cls, path: str | Path) -> "ParticleFilter":
        """Restore a filter saved with save_state; continues deterministically."""
        with open(path, "rb") as fh:
            state = pickle.load(fh)
        pf = cls(
            n_particles=state["n_particles"],
            motion_noise=state["motion_noise"],
            resample_method=state["resample_method"],
            ess_threshold=state["ess_threshold"],
            seed=state["seed"],
        )
        pf.particles = state["particles"]
        pf.weights = state["weights"]
        pf.rng.bit_generator.state = state["rng_state"]
        pf.degeneracy_events = state["degeneracy_events"]
        pf.resample_events = state["resample_events"]
        return pf
