"""Synthetic data generation: landmarks, ground-truth trajectory,
controls and noisy observations. Everything is generated in code."""

from __future__ import annotations

import numpy as np

from .particle_filter import wrap_angle


class LandmarkMap:
    """A simple local landmark map (list of 2D points)."""

    def __init__(self, landmarks):
        self.landmarks = np.asarray(landmarks, dtype=float)

    @classmethod
    def random(cls, n_landmarks, xlim=(-10, 10), ylim=(-10, 10), seed=0):
        rng = np.random.default_rng(seed)
        xs = rng.uniform(*xlim, n_landmarks)
        ys = rng.uniform(*ylim, n_landmarks)
        return cls(np.column_stack([xs, ys]))


class Simulator:
    """Generates ground truth + sensor data and runs a filter against it."""

    def __init__(
        self,
        landmark_map,
        range_std=0.3,
        bearing_std=0.1,
        sensor_range=15.0,
        seed=1,
    ):
        self.map = landmark_map
        self.range_std = float(range_std)
        self.bearing_std = float(bearing_std)
        self.sensor_range = float(sensor_range)
        self.rng = np.random.default_rng(seed)

    def step_truth(self, pose, control, dt):
        """Noise-free ground-truth integration of (v, omega)."""
        v, omega = control
        x, y, theta = pose
        if abs(omega) < 1e-9:
            x += v * dt * np.cos(theta)
            y += v * dt * np.sin(theta)
        else:
            x += v / omega * (np.sin(theta + omega * dt) - np.sin(theta))
            y += v / omega * (-np.cos(theta + omega * dt) + np.cos(theta))
        theta = float(wrap_angle(theta + omega * dt))
        return np.array([x, y, theta])

    def observe(self, pose, with_bearing=True):
        """Noisy landmark observations from a pose.

        Returns (observations, landmarks) where observations are
        [range, bearing] rows (or [range] if with_bearing=False) for the
        landmarks within sensor range.
        """
        rel = self.map.landmarks - pose[:2]
        ranges = np.hypot(rel[:, 0], rel[:, 1])
        visible = ranges <= self.sensor_range
        lms = self.map.landmarks[visible]
        rel = rel[visible]
        ranges = ranges[visible]
        noisy_ranges = ranges + self.rng.normal(0.0, self.range_std, len(ranges))
        if not with_bearing:
            return noisy_ranges, lms
        bearings = np.arctan2(rel[:, 1], rel[:, 0]) - pose[2]
        noisy_bearings = wrap_angle(
            bearings + self.rng.normal(0.0, self.bearing_std, len(bearings))
        )
        return np.column_stack([noisy_ranges, noisy_bearings]), lms

    def run(self, pf, initial_pose, controls, dt, with_bearing=True):
        """Run the filter over a control sequence.

        Returns dict with ground-truth trajectory, estimates and errors.
        All arrays are plain numpy arrays kept in memory.
        """
        n_steps = len(controls)
        truth = np.zeros((n_steps + 1, 3))
        estimates = np.zeros((n_steps + 1, 3))
        truth[0] = initial_pose
        estimates[0] = pf.estimate()

        pose = np.asarray(initial_pose, dtype=float).copy()
        for k, control in enumerate(controls):
            pose = self.step_truth(pose, control, dt)
            truth[k + 1] = pose

            pf.motion_update(control, dt)
            obs, lms = self.observe(pose, with_bearing=with_bearing)
            if len(lms) > 0:
                pf.measurement_update(
                    obs, lms, self.range_std,
                    self.bearing_std if with_bearing else None,
                )
            pf.maybe_resample()
            estimates[k + 1] = pf.estimate()

        pos_error = np.hypot(
            estimates[:, 0] - truth[:, 0], estimates[:, 1] - truth[:, 1]
        )
        yaw_error = np.abs(wrap_angle(estimates[:, 2] - truth[:, 2]))
        return {
            "truth": truth,
            "estimates": estimates,
            "pos_error": pos_error,
            "yaw_error": yaw_error,
        }
