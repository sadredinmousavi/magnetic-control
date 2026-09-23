"""Load the Path-v4 top-view CAD geometry used by Condition 007."""

import json
from pathlib import Path

import numpy as np


ASSET_PATH = Path(__file__).resolve().parent / "assets" / "Path-v4_top_XZ_coordinates.json"
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

# Path-v4 already contains its center and outside entrances in the CAD model,
# so no synthetic wall openings are needed.
WALL_SEGMENTS = tuple(
    (start.copy(), end.copy())
    for polyline in CAD_EDGE_POLYLINES
    for start, end in zip(polyline[:-1], polyline[1:])
)

CAD_BOUNDING_BOX = np.array([
    [CAD_DATA["bounding_box_mm"]["x_min"], CAD_DATA["bounding_box_mm"]["z_min"]],
    [CAD_DATA["bounding_box_mm"]["x_max"], CAD_DATA["bounding_box_mm"]["z_max"]],
], dtype=float) * MM_TO_M


