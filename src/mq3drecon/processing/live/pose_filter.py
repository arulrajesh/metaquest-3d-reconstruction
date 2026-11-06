"""Constant-velocity Kalman filter over the headset trajectory (lightweight: numpy only).

Used to:

* detect tracking jumps statistically: an innovation that is implausible under
  the motion model (Mahalanobis test) and large in meters;
* estimate linear and angular speed, which predict motion blur in a camera
  frame (``predicted_blur_px``) for keyframe rejection.

Positions are meters and quaternions are (x, y, z, w), in any consistent world frame.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PoseFilterResult:
    timestamp_s: float
    mahalanobis2: float
    is_jump: bool
    speed_mps: float
    angular_speed_dps: float

    def predicted_blur_px(self, focal_px: float, exposure_s: float) -> float:
        """Image-plane motion during the exposure from head rotation (dominant for head-mounted cameras)."""
        return float(np.deg2rad(self.angular_speed_dps) * exposure_s * focal_px)


def _angle_between(q0: np.ndarray, q1: np.ndarray) -> float:
    dot = abs(float(np.dot(q0 / np.linalg.norm(q0), q1 / np.linalg.norm(q1))))
    return 2.0 * float(np.arccos(min(1.0, dot)))


class PoseKalmanFilter:
    """State ``[p, v]`` with white-acceleration process noise; measurement ``p``."""

    def __init__(
        self,
        *,
        accel_noise_mps2: float = 4.0,
        position_noise_m: float = 0.005,
        jump_mahalanobis2: float = 60.0,
        jump_min_m: float = 0.15,
        angular_smoothing: float = 0.5,
    ):
        self.accel_noise = accel_noise_mps2
        self.r = position_noise_m ** 2
        self.jump_mahalanobis2 = jump_mahalanobis2
        self.jump_min_m = jump_min_m
        self.angular_smoothing = angular_smoothing
        self.x: np.ndarray | None = None
        self.p = np.eye(6)
        self.t: float | None = None
        self.q: np.ndarray | None = None
        self.angular_speed = 0.0

    def reset(self, t: float, position: np.ndarray, quat: np.ndarray) -> None:
        self.x = np.concatenate([np.asarray(position, float), np.zeros(3)])
        self.p = np.diag([self.r] * 3 + [1.0] * 3)
        self.t = t
        self.q = np.asarray(quat, float)
        self.angular_speed = 0.0

    def update(self, t: float, position, quat) -> PoseFilterResult:
        position = np.asarray(position, float)
        quat = np.asarray(quat, float)
        if self.x is None or self.t is None or self.q is None or t <= self.t:
            self.reset(t, position, quat)
            return PoseFilterResult(t, 0.0, False, 0.0, 0.0)
        dt = t - self.t
        f = np.eye(6)
        f[:3, 3:] = np.eye(3) * dt
        g = self.accel_noise ** 2
        q11, q12, q22 = dt ** 4 / 4 * g, dt ** 3 / 2 * g, dt ** 2 * g
        q = np.block([[np.eye(3) * q11, np.eye(3) * q12], [np.eye(3) * q12, np.eye(3) * q22]])
        x_pred = f @ self.x
        p_pred = f @ self.p @ f.T + q
        innovation = position - x_pred[:3]
        s = p_pred[:3, :3] + np.eye(3) * self.r
        mahalanobis2 = float(innovation @ np.linalg.solve(s, innovation))
        is_jump = mahalanobis2 > self.jump_mahalanobis2 and float(np.linalg.norm(innovation)) > self.jump_min_m

        omega = np.rad2deg(_angle_between(self.q, quat)) / dt
        if is_jump:
            self.reset(t, position, quat)
            return PoseFilterResult(t, mahalanobis2, True, 0.0, 0.0)
        gain = p_pred[:, :3] @ np.linalg.inv(s)
        self.x = x_pred + gain @ innovation
        self.p = (np.eye(6) - gain[:, :3] @ np.eye(3, 6)) @ p_pred
        self.angular_speed = self.angular_smoothing * self.angular_speed + (1 - self.angular_smoothing) * omega
        self.t, self.q = t, quat
        return PoseFilterResult(t, mahalanobis2, False, float(np.linalg.norm(self.x[3:])), float(self.angular_speed))
