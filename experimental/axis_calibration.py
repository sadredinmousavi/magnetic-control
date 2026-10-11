"""Robot-frame orientation after four-point perspective/metric calibration."""

import numpy as np

from gui004 import calibrated_dwell_coordinates


AXIS_TARGETS_M = {"Origin": (0.0, 0.0), "+X": (0.01, 0.0), "+Y": (0.0, 0.01)}
AXIS_DISPLAY_NAMES = {"Origin": "Center", "+X": "+X", "+Y": "+Y"}


class RobotAxes:
    def __init__(self, corners, width_cm, height_cm, samples):
        pixels = np.asarray([samples[name] for name in AXIS_TARGETS_M], dtype=float)
        if pixels.shape != (3, 2) or not np.isfinite(pixels).all():
            raise ValueError("Capture the settled robot at Center, +X and +Y first.")
        if np.min(np.linalg.norm(pixels[1:] - pixels[0], axis=1)) < 2:
            raise ValueError("Axis movement is too small to resolve (less than 2 pixels). Capture settled positions again.")
        metric = calibrated_dwell_coordinates(
            [{"center": point} for point in pixels], corners, width_cm, height_cm,
        )
        vectors = metric[1:] - metric[0]
        lengths = np.linalg.norm(vectors, axis=1)
        if np.any(lengths < 1e-8):
            raise ValueError("Axis reference positions must be distinct.")
        basis = (vectors / lengths[:, None]).T
        if not np.isfinite(basis).all() or abs(np.linalg.det(basis)) < .5:
            raise ValueError("The captured X and Y directions are nearly parallel. Check the robot positions and recapture.")
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
