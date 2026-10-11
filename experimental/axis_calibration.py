"""Robot-frame orientation after four-point perspective/metric calibration."""

import sys
from pathlib import Path

import numpy as np

from gui004 import calibrated_dwell_coordinates


AXIS_TARGETS_M = {"Origin": (0.0, 0.0), "+X": (0.01, 0.0), "+Y": (0.0, 0.001)}


class RobotAxes:
    def __init__(self, corners, width_cm, height_cm, samples):
        pixels = np.asarray([samples[name] for name in AXIS_TARGETS_M], dtype=float)
        if pixels.shape != (3, 2) or not np.isfinite(pixels).all():
            raise ValueError("Mark the robot at Origin, +X and +Y first.")
        if np.min(np.linalg.norm(pixels[1:] - pixels[0], axis=1)) < 2:
            raise ValueError("Axis movement is too small to resolve (less than 2 pixels). Mark settled positions again.")
        metric = calibrated_dwell_coordinates(
            [{"center": point} for point in pixels], corners, width_cm, height_cm,
        )
        vectors = metric[1:] - metric[0]
        lengths = np.linalg.norm(vectors, axis=1)
        if np.any(lengths < 1e-8):
            raise ValueError("Axis reference positions must be distinct.")
        basis = (vectors / lengths[:, None]).T
        if not np.isfinite(basis).all() or abs(np.linalg.det(basis)) < .5:
            raise ValueError("The marked X and Y directions are nearly parallel. Check the robot positions.")
        # Fit orthogonal directions, including mirrored camera views. Preserve
        # the physical scale from the four-dot rectangle, not command errors.
        u, _, vt = np.linalg.svd(basis)
        self.rotation = u @ vt
        self.origin = metric[0]
        self.lengths_cm = lengths
        self.corners, self.width_cm, self.height_cm = corners, width_cm, height_cm

    def coordinates(self, positions):
        metric = calibrated_dwell_coordinates(
            [{"center": point} for point in positions],
            self.corners, self.width_cm, self.height_cm,
        )
        return (metric - self.origin) @ self.rotation


def axis_target_angles(case_name, target):
    """Use the same physical model and synthesis as Usage 2; never move here."""
    project = str(Path(__file__).resolve().parents[1])
    if project not in sys.path:
        sys.path.insert(0, project)
    from case_loader import build_common_config, load_case
    from control_workflow import run_control_workflow

    params = load_case(case_name)
    params["TARGET_SCHEDULE"] = [(0.0, np.asarray(target, dtype=float), 1.0, 0.0)]
    params["INITIAL_ROBOT_POSITIONS"] = np.zeros((1, 2))
    params["OPTIMIZATION_FAILURE_MODE"] = "raise"
    result = run_control_workflow(build_common_config(params), params, compute_fields=False)
    return result.opt_infos[0]["angles_deg"]
