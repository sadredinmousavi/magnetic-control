"""A compact swarm traverses curved, rectangular, and zigzag corridors."""

import numpy as np


NUM_ROBOTS = 7
FORMATION_DURATION = 10.0
PATH_STEP_DURATION = 10.0

# Compact geometry (meters).
WORKSPACE_RADIUS = 0.110
HALL_HALF_WIDTH = 0.040
HALL_HALF_HEIGHT = 0.025
ENTRANCE_HALF_HEIGHT = 0.019
LEFT_JUNCTION_X = -0.085
RIGHT_JUNCTION_X = 0.085
UPPER_PATH_RADIUS = 0.080
UPPER_HALF_WIDTH = 0.015
LOWER_HALF_WIDTH = 0.013


def _wall(start, end, inward_normal):
    """Build one non-penetrable, one-sided wall segment."""
    return (
        np.asarray(start, dtype=float),
        np.asarray(end, dtype=float),
        np.asarray(inward_normal, dtype=float),
    )


def _arc_walls(radius, half_width, angles):
    """Return concentric one-sided walls enclosing a curved corridor."""
    outer_points = (radius + half_width) * np.column_stack((
        np.cos(angles),
        np.sin(angles),
    ))
    inner_points = (radius - half_width) * np.column_stack((
        np.cos(angles),
        np.sin(angles),
    ))
    walls = []
    for points, normal_sign in ((outer_points, -1.0), (inner_points, 1.0)):
        for start, end in zip(points[:-1], points[1:]):
            midpoint = 0.5 * (start + end)
            allowed_normal = normal_sign * midpoint / np.linalg.norm(midpoint)
            walls.append(_wall(start, end, allowed_normal))
    return walls


def _offset_boundaries(points, half_width):
    """Create two smooth boundaries around a polyline centerline."""
    tangents = np.empty_like(points)
    tangents[0] = points[1] - points[0]
    tangents[-1] = points[-1] - points[-2]
    tangents[1:-1] = points[2:] - points[:-2]
    tangents /= np.linalg.norm(tangents, axis=1, keepdims=True)
    normals = np.column_stack((-tangents[:, 1], tangents[:, 0]))
    return points + half_width * normals, points - half_width * normals


def _polyline_walls(points, half_width):
    """Return one-sided walls enclosing a polyline corridor."""
    left_points, right_points = _offset_boundaries(points, half_width)
    walls = []
    for start, end in zip(left_points[:-1], left_points[1:]):
        tangent = (end - start) / np.linalg.norm(end - start)
        walls.append(_wall(start, end, [tangent[1], -tangent[0]]))
    for start, end in zip(right_points[:-1], right_points[1:]):
        tangent = (end - start) / np.linalg.norm(end - start)
        walls.append(_wall(start, end, [-tangent[1], tangent[0]]))
    return walls


# Upper semicircular route, retained from the reference geometry.
_upper_angles = np.linspace(np.deg2rad(8.0), np.pi, 15)
UPPER_PATH_POINTS = UPPER_PATH_RADIUS * np.column_stack((
    np.cos(_upper_angles),
    np.sin(_upper_angles),
))
UPPER_WALL_SEGMENTS = _arc_walls(
    UPPER_PATH_RADIUS,
    UPPER_HALF_WIDTH,
    _upper_angles,
)

# Central rectangular hall. Its two openings are 0.038 m high, slightly less
# than the hall's 0.050 m height. The left entrance is straight; the right one
# tapers from the full hall height to the smaller junction opening.
HALL_WALL_SEGMENTS = [
    _wall(
        [LEFT_JUNCTION_X, ENTRANCE_HALF_HEIGHT],
        [-HALL_HALF_WIDTH, ENTRANCE_HALF_HEIGHT],
        [0.0, -1.0],
    ),
    _wall(
        [LEFT_JUNCTION_X, -ENTRANCE_HALF_HEIGHT],
        [-HALL_HALF_WIDTH, -ENTRANCE_HALF_HEIGHT],
        [0.0, 1.0],
    ),
    _wall(
        [-HALL_HALF_WIDTH, ENTRANCE_HALF_HEIGHT],
        [-HALL_HALF_WIDTH, HALL_HALF_HEIGHT],
        [1.0, 0.0],
    ),
    _wall(
        [-HALL_HALF_WIDTH, -HALL_HALF_HEIGHT],
        [-HALL_HALF_WIDTH, -ENTRANCE_HALF_HEIGHT],
        [1.0, 0.0],
    ),
    _wall(
        [-HALL_HALF_WIDTH, HALL_HALF_HEIGHT],
        [HALL_HALF_WIDTH, HALL_HALF_HEIGHT],
        [0.0, -1.0],
    ),
    _wall(
        [-HALL_HALF_WIDTH, -HALL_HALF_HEIGHT],
        [HALL_HALF_WIDTH, -HALL_HALF_HEIGHT],
        [0.0, 1.0],
    ),
    _wall(
        [HALL_HALF_WIDTH, HALL_HALF_HEIGHT],
        [RIGHT_JUNCTION_X, ENTRANCE_HALF_HEIGHT],
        [-0.006, -0.045],
    ),
    _wall(
        [HALL_HALF_WIDTH, -HALL_HALF_HEIGHT],
        [RIGHT_JUNCTION_X, -ENTRANCE_HALF_HEIGHT],
        [-0.006, 0.045],
    ),
]

# Lower /\/\/ corridor, also retained from the reference geometry.
LOWER_PATH_POINTS = np.array([
    [RIGHT_JUNCTION_X, -0.035],
    [0.060, -0.068],
    [0.030, -0.040],
    [0.000, -0.074],
    [-0.030, -0.040],
    [-0.060, -0.068],
    [LEFT_JUNCTION_X, -0.035],
])
LOWER_WALL_SEGMENTS = _polyline_walls(
    LOWER_PATH_POINTS,
    LOWER_HALF_WIDTH,
)

# Short vertical connectors join the hall junctions to the lower zigzag.
CONNECTOR_WALL_SEGMENTS = [
    _wall(
        [RIGHT_JUNCTION_X - LOWER_HALF_WIDTH, -0.035],
        [RIGHT_JUNCTION_X - LOWER_HALF_WIDTH, 0.0],
        [1.0, 0.0],
    ),
    _wall(
        [RIGHT_JUNCTION_X + LOWER_HALF_WIDTH, -0.035],
        [RIGHT_JUNCTION_X + LOWER_HALF_WIDTH, 0.0],
        [-1.0, 0.0],
    ),
    _wall(
        [LEFT_JUNCTION_X - LOWER_HALF_WIDTH, -0.035],
        [LEFT_JUNCTION_X - LOWER_HALF_WIDTH, 0.0],
        [1.0, 0.0],
    ),
    _wall(
        [LEFT_JUNCTION_X + LOWER_HALF_WIDTH, -0.035],
        [LEFT_JUNCTION_X + LOWER_HALF_WIDTH, 0.0],
        [-1.0, 0.0],
    ),
]

WALL_SEGMENTS = (
    UPPER_WALL_SEGMENTS
    + HALL_WALL_SEGMENTS
    + LOWER_WALL_SEGMENTS
    + CONNECTOR_WALL_SEGMENTS
)

# Visit all three sections: upper arc, central hall, then lower zigzag.
HALL_PATH_POINTS = np.array([
    [LEFT_JUNCTION_X, 0.0],
    [-HALL_HALF_WIDTH, 0.0],
    [0.0, 0.0],
    [HALL_HALF_WIDTH, 0.0],
    [RIGHT_JUNCTION_X, 0.0],
])
PATH_POINTS = np.vstack((
    UPPER_PATH_POINTS,
    HALL_PATH_POINTS,
    LOWER_PATH_POINTS,
))

# Case 2 uses a much tighter initial group than Condition 008 in Case 1.
_robot_angles = np.linspace(0.0, 2.0 * np.pi, NUM_ROBOTS, endpoint=False)
INITIAL_ROBOT_POSITIONS = PATH_POINTS[0] + np.column_stack((
    0.0015 * np.cos(_robot_angles),
    0.0015 * np.sin(_robot_angles),
))

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

    "DISH_RADIUS": WORKSPACE_RADIUS,
    "T_SPAN": (0.0, TARGET_SCHEDULE[-1][0] + PATH_STEP_DURATION),
    "T_EVAL_POINTS": 600,
    "SOLVER_PROGRESS_INTERVAL": 0.5,
    "SOLVER_MAX_STEP": 0.05,

    "ANIMATION_TITLE": "Swarm Control Through Compact Constrained Pathways",
    "ANIMATION_DRAW_TRAJECTORIES": False,
    "ANIMATION_DRAW_TARGET_TRAJECTORY": True,

    "WALL_SEGMENTS": WALL_SEGMENTS,
    "WALL_STIFFNESS": 5e-4,
    "WALL_DAMPING": 5e-6,
    "WALL_INTERACTION_RANGE": 0.0015,
    # Keep collision recovery local at the three-way hall junctions so one
    # branch does not repel robots following a neighboring branch.
    "WALL_RECOVERY_DEPTH": 0.004,

    # Payload remains disabled for this condition.
    "PAYLOAD_INITIAL_POS": np.array([10.0, 10.0]),
    "PAYLOAD_INITIAL_VEL": np.array([0.0, 0.0]),
}
