"""Load the Path-v3 top-view CAD geometry used by Condition 005."""

import json
from pathlib import Path

import numpy as np


ASSET_PATH = Path(__file__).resolve().parent / "assets" / "Path-v3_top_XZ_coordinates.json"
MM_TO_M = 1e-3


def _load_edge_polylines():
    data = json.loads(ASSET_PATH.read_text(encoding="utf-8"))
    if data.get("units") != "mm":
        raise ValueError(f"Expected millimetres in {ASSET_PATH}.")

    polylines = tuple(
        np.asarray(edge["points_xz_mm"], dtype=float) * MM_TO_M
        for edge in data["edges"]
    )
    if len(polylines) != data.get("visible_edge_count"):
        raise ValueError(f"Edge count does not match metadata in {ASSET_PATH}.")
    return data, polylines


CAD_DATA, CAD_EDGE_POLYLINES = _load_edge_polylines()

# Opening centers and radii are in the original CAD coordinate frame.
WALL_OPENINGS = (
    (np.array([-65.0, 0.0]) * MM_TO_M, 5.0 * MM_TO_M, None),
    (np.array([-15.0, 0.0]) * MM_TO_M, 4.0 * MM_TO_M, (3,)),
    (np.array([0.0, 15.0]) * MM_TO_M, 7.0 * MM_TO_M, (5, 19)),
    (np.array([15.0, 0.0]) * MM_TO_M, 7.0 * MM_TO_M, (18,)),
    (np.array([-57.7, -30.0]) * MM_TO_M, 5.0 * MM_TO_M, (37,)),
)


def _cut_wall_opening(segments, center, radius):
    remaining = []
    for start, end in segments:
        direction = end - start
        a = np.dot(direction, direction)
        b = 2.0 * np.dot(start - center, direction)
        c = np.dot(start - center, start - center) - radius**2
        discriminant = b**2 - 4.0 * a * c
        if discriminant <= 0:
            remaining.append((start, end))
            continue

        root = np.sqrt(discriminant)
        enter = np.clip((-b - root) / (2.0 * a), 0.0, 1.0)
        exit = np.clip((-b + root) / (2.0 * a), 0.0, 1.0)
        if enter >= exit:
            remaining.append((start, end))
            continue
        if enter > 0:
            remaining.append((start, start + enter * direction))
        if exit < 1:
            remaining.append((start + exit * direction, end))
    return tuple(remaining)


# Keep the source CAD edges intact; clip only the simulator/plot walls.
_wall_segments = []
for _edge_id, _polyline in enumerate(CAD_EDGE_POLYLINES):
    _edge_segments = tuple(
        (start.copy(), end.copy())
        for start, end in zip(_polyline[:-1], _polyline[1:])
    )
    for _center, _radius, _edge_ids in WALL_OPENINGS:
        if _edge_ids is None or _edge_id in _edge_ids:
            _edge_segments = _cut_wall_opening(_edge_segments, _center, _radius)
    _wall_segments.extend(_edge_segments)
WALL_SEGMENTS = tuple(_wall_segments)

CAD_BOUNDING_BOX = np.array([
    [CAD_DATA["bounding_box_mm"]["x_min"], CAD_DATA["bounding_box_mm"]["z_min"]],
    [CAD_DATA["bounding_box_mm"]["x_max"], CAD_DATA["bounding_box_mm"]["z_max"]],
], dtype=float) * MM_TO_M


