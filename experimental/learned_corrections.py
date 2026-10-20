"""Thread-safe, context-scoped memory of successfully applied path corrections."""
import hashlib
import json
import threading
from dataclasses import asdict
from pathlib import Path

import numpy as np


def correction_context(setup, steps, hardware_signature, robot_count):
    # A moving detection ROI must not invalidate learned physical coordinates.
    geometry = {key: setup.get(key) for key in (
        'camera_source', 'video_size_px', 'corners_px', 'rectangle_width_cm',
        'rectangle_height_cm', 'axis_samples_px', 'axis_targets_m', 'coordinate_system', 'control_case')}
    if setup.get('control_target', 'Robots') != 'Robots':
        geometry['control_target'] = setup['control_target']
    value = dict(geometry=geometry, steps=[asdict(step) for step in steps],
                 hardware=hardware_signature, robots=int(robot_count))
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


class LearnedCorrections:
    def __init__(self):
        self.lock = threading.Lock()
        self.values = {}

    def snapshot(self, context, count):
        with self.lock:
            points = self.values.get(context, {})
            return [list(points.get(str(index), [0., 0.])) for index in range(count)]

    def point(self, context, index):
        with self.lock:
            value = self.values.get(context, {}).get(str(index))
            return None if value is None else list(value)

    def remember(self, context, index, correction):
        vector = np.asarray(correction, float)
        if vector.shape != (2,) or not np.isfinite(vector).all():
            raise ValueError('Learned correction must be a finite XY pair in mm.')
        with self.lock:
            self.values.setdefault(context, {})[str(index)] = vector.tolist()

    def clear(self, context):
        with self.lock:
            self.values.pop(context, None)

    def save(self, path, context, count, *, steps=None):
        value = dict(type='gui005_learned_corrections', version=1, context=context,
                     units='mm', corrections=self.snapshot(context, count))
        if steps is not None:
            value['points'] = [dict(desired_xy_mm=(np.asarray(step.target_xy_m)*1000).tolist(),
                                   u_last_mm=u, commanded_xy_mm=(np.asarray(step.target_xy_m)*1000 + u).tolist(),
                                   ratio=step.ratio, angle_deg=step.angle_deg,
                                   wait_seconds=step.wait_seconds, zero_angle_deg=step.zero_angle_deg)
                               for step,u in zip(steps, value['corrections'])]
        # Explicit Save dialog handles replacement approval; no automatic overwrite.
        Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')

    def load(self, path, context, count):
        value = json.loads(Path(path).read_text(encoding='utf-8-sig'))
        if (not isinstance(value, dict) or value.get('type') != 'gui005_learned_corrections'
                or value.get('version') != 1 or value.get('units') != 'mm' or value.get('context') != context):
            raise ValueError('Corrections do not match this sequence, calibration, hardware and robot count.')
        vectors = np.asarray(value.get('corrections'), float)
        if vectors.shape != (count, 2) or not np.isfinite(vectors).all():
            raise ValueError('Invalid learned correction values/count.')
        with self.lock:
            self.values[context] = {str(index): vector.tolist() for index, vector in enumerate(vectors)}
