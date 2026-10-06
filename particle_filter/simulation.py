"""Synthetic test-data generation: maps, trajectories, controls, observations.

All data is generated in code and kept in memory / local files only.
"""

from __future__ import annotations

import numpy as np

from .models import wrap_to_pi


def make_landmarks(rng: np.random.Generator, n: int = 8, extent: float = 10.0) -> np.ndarray:
    """Random landmark map inside a square of half-size `extent`."""
    return rng.uniform(-extent, extent, size=(n, 2))


def simulate_trajectory(
    controls: list[tuple[float, float]],
    dt: float,
    start: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> np.ndarray:
    """Noise-free ground-truth trajectory for a control sequence -> (T+1, 3)."""
    pose = np.asarray(start, dtype=float)
    traj = [pose.copy()]
    for v, omega in controls:
        pose = pose.copy()
        mid = pose[2] + 0.5 * omega * dt
        pose[0] += v * dt * np.cos(mid)
        pose[1] += v * dt * np.sin(mid)
        pose[2] = wrap_to_pi(pose[2] + omega * dt)
        traj.append(pose.copy())
    return np.asarray(traj)


def noisy_controls(
    controls: list[tuple[float, float]],
    rng: np.random.Generator,
    std_v: float = 0.05,
    std_omega: float = 0.02,
) -> list[tuple[float, float]]:
    """Odometry-style controls corrupted by Gaussian noise."""
    out = []
    for v, omega in controls:
        out.append(
            (v + rng.normal(0.0, std_v), omega + rng.normal(0.0, std_omega))
        )
    return out


def observe(
    pose: np.ndarray,
    landmarks: np.ndarray,
    rng: np.random.Generator,
    std_range: float = 0.3,
    std_bearing: float | None = 0.1,
    max_range: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate noisy landmark observations from a pose.

    Returns (visible_landmarks, ranges, bearings). Bearings are all zeros when
    std_bearing is None (range-only mode).
    """
    dx = landmarks[:, 0] - pose[0]
    dy = landmarks[:, 1] - pose[1]
    r = np.hypot(dx, dy)
    b = wrap_to_pi(np.arctan2(dy, dx) - pose[2])
    if max_range is not None:
        mask = r <= max_range
        landmarks, r, b = landmarks[mask], r[mask], b[mask]
    r_obs = r + rng.normal(0.0, std_range, size=r.shape)
    if std_bearing is None:
        b_obs = np.zeros_like(b)
    else:
        b_obs = wrap_to_pi(b + rng.normal(0.0, std_bearing, size=b.shape))
    return landmarks, r_obs, b_obs
