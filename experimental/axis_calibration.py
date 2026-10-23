"""Robot-frame orientation after four-point perspective/metric calibration."""

import numpy as np
import cv2

from gui004 import calibrated_dwell_coordinates


AXIS_TARGETS_M = {"Origin": (0.0, 0.0), "+X": (0.01, 0.0), "+Y": (0.0, 0.01)}
AXIS_MOVE_TARGETS_M = {
    'Origin': (0., 0.),
    '-X': (-.01, 0.), '+X': (.01, 0.), '+2X': (.02, 0.),
    '-Y': (0., -.01), '+Y': (0., .01), '+2Y': (0., .02),
}
# Retain saved references from earlier setups without offering these moves.
AXIS_SAVED_TARGETS_M = dict(AXIS_MOVE_TARGETS_M, **{'+3X': (.03, 0.), '+3Y': (0., .03)})
AXIS_REFERENCE_NAMES = {name: 'Origin' if name == 'Origin' else '+' + name[-1]
                        for name in AXIS_SAVED_TARGETS_M}
AXIS_DISPLAY_NAMES = {name: 'Center' if name == 'Origin' else name for name in AXIS_MOVE_TARGETS_M}


def manual_line_axes(corners, width_cm, height_cm, captures):
    """Intersect directed X/Y lines in the metric plane, independent of travel distances."""
    names = ('-X', '+X', '-Y', '+Y')
    if not isinstance(captures, dict) or any(name not in captures for name in names):
        raise ValueError('Select -X, +X, -Y and +Y to define both axis lines.')
    pixels = np.asarray([captures[name] for name in names], dtype=float)
    if pixels.shape != (4, 2) or not np.isfinite(pixels).all():
        raise ValueError('Axis line points must be finite XY pairs.')
    if min(np.linalg.norm(pixels[1] - pixels[0]), np.linalg.norm(pixels[3] - pixels[2])) < 2:
        raise ValueError('Axis line endpoints are less than 2 pixels apart. Select points farther apart.')
    metric = calibrated_dwell_coordinates([{'center': p} for p in pixels], corners, width_cm, height_cm)
    vectors = np.array([metric[1] - metric[0], metric[3] - metric[2]])
    lengths = np.linalg.norm(vectors, axis=1)
    if np.any(lengths < 1e-8):
        raise ValueError('Opposite axis captures must be distinct.')
    directions = vectors / lengths[:, None]
    basis = directions.T
    if abs(np.linalg.det(basis)) < .5:
        raise ValueError('The captured X and Y directions are nearly parallel. Recapture the axes.')
    distances = np.linalg.solve(np.column_stack((directions[0], -directions[1])), metric[2] - metric[0])
    origin = metric[0] + distances[0] * directions[0]
    # Use the existing projection so every tracking/command consumer uses the same fit.
    references = RobotAxes.metric_pixel_positions(
        origin + np.array([[0., 0.], directions[0], directions[1]]), corners, width_cm, height_cm)
    samples = dict(zip(AXIS_TARGETS_M, references.tolist()))
    axes = RobotAxes(corners, width_cm, height_cm, samples)
    angle = np.degrees(np.arccos(np.clip(np.dot(*directions), -1., 1.)))
    return samples, f'Measured axis angle: {angle:.2f} deg (90 deg ideal).'


def symmetric_axes(corners, width_cm, height_cm, captures):
    """Intersect robot travel lines and report optional checks against commanded travel."""
    samples, report = manual_line_axes(corners, width_cm, height_cm, captures)
    axes = RobotAxes(corners, width_cm, height_cm, samples)
    checks = [report]
    for name in ('Origin', '+2X', '+2Y'):
        if name in captures:
            observed = axes.coordinates([captures[name]])[0] * 10
            expected = np.asarray(AXIS_MOVE_TARGETS_M[name]) * 1000
            error = np.linalg.norm(observed - expected)
            label = 'Center offset from intersection' if name == 'Origin' else f'{name} target error'
            checks.append(f'{label}: {error:.2f} mm.')
    if 'Origin' not in captures:
        checks.append('Capture Center to check its offset from the intersection.')
    return samples, ' '.join(checks)


class RobotAxes:
    def __init__(self, corners, width_cm, height_cm, samples, capture_moves=None):
        pixels = np.asarray([samples[name] for name in AXIS_TARGETS_M], dtype=float)
        if pixels.shape != (3, 2) or not np.isfinite(pixels).all():
            raise ValueError("Capture the settled robot at Center, +X and +Y first.")
        if np.min(np.linalg.norm(pixels[1:] - pixels[0], axis=1)) < 2:
            raise ValueError("Axis movement is too small to resolve (less than 2 pixels). Capture settled positions again.")
        metric = calibrated_dwell_coordinates(
            [{"center": point} for point in pixels], corners, width_cm, height_cm,
        )
        vectors = metric[1:] - metric[0]
        # Negative captures observe the opposite of the positive robot axis.
        for index, name in enumerate(('+X', '+Y')):
            if (capture_moves or {}).get(name, name).startswith('-'):
                vectors[index] *= -1
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

    def pixel_positions(self, coordinates_cm):
        """Inverse robot XY mapping for desired-position camera overlays."""
        coordinates = np.asarray(coordinates_cm, dtype=float)
        if coordinates.ndim != 2 or coordinates.shape[1] != 2 or not np.isfinite(coordinates).all():
            raise ValueError("Desired coordinates must be finite XY pairs.")
        metric = coordinates @ self.rotation.T + self.origin
        return self.metric_pixel_positions(metric, self.corners, self.width_cm, self.height_cm)

    @staticmethod
    def metric_pixel_positions(metric, corners, width_cm, height_cm):
        """Project coordinates in the pin rectangle's metric plane back to pixels."""
        plane = np.float32([
            [-width_cm / 2, height_cm / 2],
            [width_cm / 2, height_cm / 2],
            [width_cm / 2, -height_cm / 2],
            [-width_cm / 2, -height_cm / 2],
        ])
        inverse = cv2.getPerspectiveTransform(plane, np.asarray(corners, np.float32))
        homogeneous = np.column_stack((metric, np.ones(len(metric)))) @ inverse.T
        if np.any(np.abs(homogeneous[:, 2]) < 1e-10):
            raise ValueError("Desired point is outside the usable camera projection.")
        pixels = homogeneous[:, :2] / homogeneous[:, 2:3]
        if not np.isfinite(pixels).all():
            raise ValueError("Desired point projection is not finite.")
        return pixels
