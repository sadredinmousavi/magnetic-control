"""Seven microrobots navigate a radial maze with curved and zigzag walls."""

import numpy as np


NUM_ROBOTS = 7
FORMATION_DURATION = 20.0
PATH_STEP_DURATION = 10.0

# Approximate the reference image as a continuous route: enter from the upper
# right, follow a semicircular lane counter-clockwise, descend on the left,
# then traverse the lower zigzag and leave through the lower-right opening.
_upper_arc_angles = np.linspace(np.deg2rad(10.0), np.pi, 17)
_upper_arc = 0.105 * np.column_stack((
    np.cos(_upper_arc_angles),
    np.sin(_upper_arc_angles),
))
_upper_entrance = np.array([[0.130, 0.018]])
_lower_route = np.array([
    [-0.105, 0.000],
    [-0.105, -0.038],
    [-0.080, -0.072],
    [-0.050, -0.045],
    [-0.015, -0.090],
    [0.020, -0.050],
    [0.055, -0.086],
    [0.085, -0.055],
    [0.130, -0.055],
])
PATH_POINTS = np.vstack((
    _upper_entrance,
    _upper_arc,
    _lower_route[1:],
))

# The entrance and exit are slightly wider than the curved and zigzag lanes.
CORRIDOR_HALF_WIDTHS = np.full(len(PATH_POINTS), 0.014)
CORRIDOR_HALF_WIDTHS[:2] = 0.018
CORRIDOR_HALF_WIDTHS[-2:] = 0.018


def _offset_boundaries(points, half_widths):
    """Return smooth left/right boundaries for a sampled centerline."""
    tangents = np.empty_like(points)
    tangents[0] = points[1] - points[0]
    tangents[-1] = points[-1] - points[-2]
    tangents[1:-1] = points[2:] - points[:-2]
    tangents /= np.linalg.norm(tangents, axis=1, keepdims=True)
    normals = np.column_stack((-tangents[:, 1], tangents[:, 0]))
    offsets = half_widths[:, None] * normals
    return points + offsets, points - offsets


def _one_sided_segments(points, interior_side):
    """Convert a boundary polyline into segments with inward normals."""
    walls = []
    for start, end in zip(points[:-1], points[1:]):
        tangent = end - start
        tangent /= np.linalg.norm(tangent)
        left_normal = np.array([-tangent[1], tangent[0]])
        inward_normal = (
            left_normal if interior_side == "left" else -left_normal
        )
        walls.append((start, end, inward_normal))
    return walls


LEFT_WALL_POINTS, RIGHT_WALL_POINTS = _offset_boundaries(
    PATH_POINTS,
    CORRIDOR_HALF_WIDTHS,
)
CORRIDOR_WALL_SEGMENTS = (
    _one_sided_segments(LEFT_WALL_POINTS, interior_side="right")
    + _one_sided_segments(RIGHT_WALL_POINTS, interior_side="left")
)

# A circular forbidden region reproduces the protected center in the reference.
# Its normals point outward, so robots are pushed away if they approach it.
_center_angles = np.linspace(0.0, 2.0 * np.pi, 33)
CENTER_OBSTACLE_POINTS = 0.027 * np.column_stack((
    np.cos(_center_angles),
    np.sin(_center_angles),
))
CENTER_OBSTACLE_SEGMENTS = []
for _start, _end in zip(
    CENTER_OBSTACLE_POINTS[:-1],
    CENTER_OBSTACLE_POINTS[1:],
):
    _midpoint = 0.5 * (_start + _end)
    _outward_normal = _midpoint / np.linalg.norm(_midpoint)
    CENTER_OBSTACLE_SEGMENTS.append((_start, _end, _outward_normal))

WALL_SEGMENTS = CORRIDOR_WALL_SEGMENTS + CENTER_OBSTACLE_SEGMENTS

# Keep the robots separated in a compact group at the corridor entrance.
_robot_angles = np.linspace(0.0, 2.0 * np.pi, NUM_ROBOTS, endpoint=False)
INITIAL_ROBOT_POSITIONS = PATH_POINTS[0] + np.column_stack((
    0.008 * np.cos(_robot_angles),
    0.008 * np.sin(_robot_angles),
))

# Hold the first equilibrium point while the robots form a group, then move
# through each successive point on the corridor centerline.
TARGET_SCHEDULE = [
    (0.0, PATH_POINTS[0], 1.0, np.deg2rad(0.0)),
    *[
        (
            FORMATION_DURATION + index * PATH_STEP_DURATION,
            point,
            1.0,
            np.deg2rad(0.0),
        )
        for index, point in enumerate(PATH_POINTS[1:])
    ],
]


PARAMS = {
    "TARGET_SCHEDULE": TARGET_SCHEDULE,
    "INITIAL_ROBOT_POSITIONS": INITIAL_ROBOT_POSITIONS,

    "T_SPAN": (0.0, TARGET_SCHEDULE[-1][0] + PATH_STEP_DURATION),
    "T_EVAL_POINTS": 800,
    "SOLVER_PROGRESS_INTERVAL": 0.5,
    "USE_OVERDAMPED_DYNAMICS": True,
    "DYNAMICS_SPEEDUP": 1.0,
    "SOLVER_RTOL": 1e-4,
    "SOLVER_ATOL": 1e-7,
    # Collision-aware step cap: prevents a robot from numerically tunnelling
    # across the thin wall-contact layer between solver evaluations.
    "SOLVER_MAX_STEP": 0.05,

    "ANIMATION_TITLE": "Swarm Control Through Constrained Pathways",
    "ANIMATION_DRAW_TRAJECTORIES": False,
    "ANIMATION_DRAW_TARGET_TRAJECTORY": True,

    "WALL_SEGMENTS": WALL_SEGMENTS,
    "WALL_STIFFNESS": 5e-4,
    "WALL_DAMPING": 5e-6,
    "WALL_INTERACTION_RANGE": 0.0015,
    # Only nearby crossed points are recovered. This prevents a distant wall
    # in the concave maze from acting like an infinite half-plane.
    "WALL_RECOVERY_DEPTH": 0.006,

    # Disable the payload for this condition. A payload version is reserved
    # for Condition 009.
    "PAYLOAD_RADIUS": 1e-12,
    "PAYLOAD_HEIGHT": 1.0,
    "PAYLOAD_DENSITY": 1.0,
    "PAYLOAD_DRAG_FACTOR": 0.0,
    "CONTACT_STIFFNESS": 0.0,
    "CONTACT_DAMPING": 0.0,
    "PAYLOAD_CAPILLARY_GAIN": 0.0,
    "PAYLOAD_CAPILLARY_RANGE": 1.0,
    "PAYLOAD_INITIAL_POS": np.array([10.0, 10.0]),
    "PAYLOAD_INITIAL_VEL": np.array([0.0, 0.0]),
}
