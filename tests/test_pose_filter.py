import numpy as np
from scipy.spatial.transform import Rotation as R

from mq3drecon.processing.live.pose_filter import PoseKalmanFilter


def _walk(n=60, fps=10.0, speed=0.8, yaw_dps=30.0):
    for i in range(n):
        t = i / fps
        yield t, np.array([speed * t, 1.6, 0.0]), R.from_euler("y", yaw_dps * t, degrees=True).as_quat()


def test_walking_is_not_a_jump_and_speeds_are_estimated():
    kf = PoseKalmanFilter()
    results = [kf.update(t, p, q) for t, p, q in _walk()]
    assert not any(r.is_jump for r in results)
    assert abs(results[-1].speed_mps - 0.8) < 0.1
    assert abs(results[-1].angular_speed_dps - 30.0) < 3.0
    assert 0.0 < results[-1].predicted_blur_px(focal_px=870.0, exposure_s=0.008) < 10.0


def test_tracking_reset_is_detected_once_and_filter_recovers():
    kf = PoseKalmanFilter()
    jumps = []
    for t, p, q in _walk():
        if t >= 3.0:
            p = p + np.array([0.0, 2.2, 1.5])  # tracking origin reset mid-walk
        jumps.append(kf.update(t, p, q).is_jump)
    assert sum(jumps) == 1 and jumps[30]
