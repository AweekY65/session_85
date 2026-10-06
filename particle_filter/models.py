"""Motion model, observation models and angle helpers for the 2D particle filter.

Everything is pure numpy / in-memory. No ROS, no hardware, no external services.
"""

from __future__ import annotations

import numpy as np

LOG_2PI = np.log(2.0 * np.pi)


def wrap_to_pi(angle: np.ndarray | float) -> np.ndarray | float:
    """Wrap angle(s) to (-pi, pi]."""
    return (np.asarray(angle) + np.pi) % (2.0 * np.pi) - np.pi


def circular_mean(theta: np.ndarray, weights: np.ndarray | None = None) -> float:
    """Weighted circular mean of headings.

    A plain arithmetic mean of angles near +/-pi gives a direction pointing
    the opposite way; averaging unit vectors avoids that.
    """
    theta = np.asarray(theta, dtype=float)
    if weights is None:
        weights = np.full(theta.shape, 1.0 / theta.size)
    else:
        weights = np.asarray(weights, dtype=float)
        weights = weights / weights.sum()
    sin_mean = float(np.sum(weights * np.sin(theta)))
    cos_mean = float(np.sum(weights * np.cos(theta)))
    return float(np.arctan2(sin_mean, cos_mean))


def motion_update(
    particles: np.ndarray,
    control: tuple[float, float],
    dt: float,
    noise_std: tuple[float, float],
    rng: np.random.Generator,
) -> np.ndarray:
    """Velocity motion model with additive Gaussian noise.

    particles: (N, 3) array of [x, y, theta].
    control:   (v, omega) commanded linear / angular velocity.
    noise_std: (std_v, std_omega) standard deviations of the applied noise.
    """
    v_cmd, omega_cmd = control
    std_v, std_omega = noise_std
    n = particles.shape[0]

    v = v_cmd + rng.normal(0.0, std_v, size=n)
    omega = omega_cmd + rng.normal(0.0, std_omega, size=n)

    out = particles.copy()
    out[:, 2] = wrap_to_pi(particles[:, 2] + omega * dt)
    # Integrate translation along the mid-step heading (2nd order approximation).
    mid_theta = wrap_to_pi(particles[:, 2] + 0.5 * omega * dt)
    out[:, 0] = particles[:, 0] + v * dt * np.cos(mid_theta)
    out[:, 1] = particles[:, 1] + v * dt * np.sin(mid_theta)
    return out


def predict_ranges(particles: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
    """Predicted range from every particle to every landmark -> (N, M)."""
    dx = landmarks[None, :, 0] - particles[:, None, 0]
    dy = landmarks[None, :, 1] - particles[:, None, 1]
    return np.hypot(dx, dy)


def predict_bearings(particles: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
    """Predicted bearing (relative to heading) per particle/landmark -> (N, M)."""
    dx = landmarks[None, :, 0] - particles[:, None, 0]
    dy = landmarks[None, :, 1] - particles[:, None, 1]
    global_bearing = np.arctan2(dy, dx)
    return wrap_to_pi(global_bearing - particles[:, None, 2])


def range_log_likelihood(
    particles: np.ndarray,
    landmarks: np.ndarray,
    ranges: np.ndarray,
    std: float,
) -> np.ndarray:
    """Log-likelihood of range-only observations for each particle -> (N,)."""
    pred = predict_ranges(particles, landmarks)
    err = pred - np.asarray(ranges)[None, :]
    with np.errstate(over="ignore", invalid="ignore"):
        sq = np.sum(err * err, axis=1)
    return -0.5 * sq / (std * std) - 0.5 * pred.shape[1] * np.log(
        2.0 * np.pi * std * std
    )


def range_bearing_log_likelihood(
    particles: np.ndarray,
    landmarks: np.ndarray,
    ranges: np.ndarray,
    bearings: np.ndarray,
    std_range: float,
    std_bearing: float,
) -> np.ndarray:
    """Log-likelihood of range+bearing observations for each particle -> (N,)."""
    pred_r = predict_ranges(particles, landmarks)
    pred_b = predict_bearings(particles, landmarks)
    err_r = pred_r - np.asarray(ranges)[None, :]
    err_b = wrap_to_pi(pred_b - np.asarray(bearings)[None, :])
    m = pred_r.shape[1]
    ll = -0.5 * np.sum(err_r * err_r, axis=1) / (std_range * std_range)
    ll += -0.5 * np.sum(err_b * err_b, axis=1) / (std_bearing * std_bearing)
    ll -= 0.5 * m * (np.log(2.0 * np.pi * std_range * std_range) + np.log(2.0 * np.pi * std_bearing * std_bearing))
    return ll
