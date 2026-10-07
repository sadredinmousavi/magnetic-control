"""Uniform planar scaling for Case 101's non-test conditions."""

from copy import deepcopy

import numpy as np

from .compensator import apply_case_101_compensation


MAX_RADIUS = 0.05


def build_scaled_condition(source):
    """Fit targets and initial robot centers; preserve timing and magnetic setup."""
    params = deepcopy(source.PARAMS)
    schedule = params["TARGET_SCHEDULE"]
    if any(len(entry) != 4 for entry in schedule):
        raise ValueError("Case 101 scaling requires single-equilibrium schedules.")
    targets = np.asarray([entry[1] for entry in schedule], dtype=float)
    initial = np.asarray(params["INITIAL_ROBOT_POSITIONS"], dtype=float)
    maximum = float(np.linalg.norm(np.vstack((targets, initial)), axis=1).max())
    # A tiny inward margin prevents floating-point rounding above 0.05 m.
    scale = min(1.0, np.nextafter(MAX_RADIUS, 0.0) / maximum) if maximum else 1.0
    params["TARGET_SCHEDULE"] = [
        (time, np.asarray(point, dtype=float) * scale, ratio, angle)
        for time, point, ratio, angle in schedule
    ]
    params["INITIAL_ROBOT_POSITIONS"] = initial * scale

    if "WALL_SEGMENTS" in params:
        params["WALL_SEGMENTS"] = tuple(
            (np.asarray(start) * scale, np.asarray(end) * scale)
            for start, end in params["WALL_SEGMENTS"]
        )
        for key in ("WALL_INTERACTION_RANGE", "WALL_RECOVERY_DEPTH"):
            if key in params:
                params[key] *= scale

    # Disabled-payload placeholders (10, 10) remain placeholders. Scale only
    # active cargo's planar dimensions; its height and material stay physical.
    if params.get("PAYLOAD_RADIUS", 0.0) > 1e-9:
        for key in ("PAYLOAD_RADIUS", "PAYLOAD_SIZE", "PAYLOAD_INITIAL_POS",
                    "PAYLOAD_INITIAL_VEL", "PAYLOAD_CAPILLARY_RANGE"):
            if key in params:
                params[key] = np.asarray(params[key]) * scale
                if np.ndim(params[key]) == 0:
                    params[key] = float(params[key])

    title = source.PARAMS.get("ANIMATION_TITLE", source.__name__.split(".")[-1])
    params["ANIMATION_TITLE"] = f"{title} (Case 101, scale {scale:.4f})"
    params["TARGET_TRAJECTORY_TITLE"] = params["ANIMATION_TITLE"]
    params["GEOMETRY_SCALE_FACTOR"] = scale
    params["TARGET_RADIUS_LIMIT"] = MAX_RADIUS
    condition_name = "case_101." + source.__name__.split(".")[-1]
    params = apply_case_101_compensation(params, condition_name)
    return params, scale
