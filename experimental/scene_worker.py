"""One latest-frame vision job at a time, with no Tk calls in the worker."""

import threading

import cv2
import numpy as np

from closed_loop_calibration import MovingROICalibration
from swarm_cargo import RedCargoTracker, cargo_geometry, swarm_geometry


def analyze_scene(job):
    frame, settings = job['frame'], job['settings']
    tracker = MovingROICalibration(**settings['robot'])
    tracker.references = list(settings['references'])
    tracker.workspace_circle = settings['workspace']
    cargo_tracker = RedCargoTracker()
    cargo_tracker.center = settings['cargo_center']
    cargo = None
    if settings['cargo_enabled']:
        mask, bounds = None, None
        if settings['workspace'] is not None:
            cx, cy, radius = settings['workspace']
            mask = np.zeros(frame.shape[:2], np.uint8)
            cv2.circle(mask, tuple(np.rint([cx, cy]).astype(int)), round(radius), 255, -1)
            bounds = (int(cx-radius)-2, int(cy-radius)-2, int(cx+radius)+3, int(cy+radius)+3)
        cargo = cargo_tracker.process(frame, settings['cargo_area'], settings['cargo_radius'],
                                     mask=mask, bounds=bounds)
        red_mask = cargo_tracker.detector.last_blob_mask
        if red_mask is not None:
            tracker.excluded_mask = cv2.dilate(red_mask, np.ones((5, 5), np.uint8))
        if cargo is not None:
            cargo['xy_cm'] = tuple(tracker.coordinates([cargo['center']])[0]) if tracker.axes is not None else None
            cargo = cargo_geometry(cargo, red_mask, tracker.axes, settings['cargo_shape'])
    # Both measurements use exactly this frame, and are published as one result.
    detections, center, group_xy = tracker.process(frame)
    geometry = swarm_geometry(detections, tracker.count, tracker.axes)
    if geometry is not None:
        group_xy = tuple(geometry['center_mm'] / 10)
    return dict(job, tracker=tracker, cargo_tracker=cargo_tracker, cargo=cargo,
                detections=detections, center=center, group_xy=group_xy, geometry=geometry)


class LatestSceneWorker:
    def __init__(self, analyze=analyze_scene):
        self.analyze = analyze
        self.condition = threading.Condition()
        self.pending = self.result = self.thread = None
        self.active = self.closed = False

    def submit(self, job):
        with self.condition:
            if self.closed:
                return
            self.pending = job  # Replace the waiting frame; never accumulate a backlog.
            if self.thread is None:
                self.thread = threading.Thread(target=self._run, daemon=True, name='gui005-vision')
                self.thread.start()
            self.condition.notify()

    def take_result(self):
        with self.condition:
            result, self.result = self.result, None
            return result

    @property
    def busy(self):
        with self.condition:
            return self.active or self.pending is not None or self.result is not None

    def _run(self):
        while True:
            with self.condition:
                self.condition.wait_for(lambda: self.closed or self.pending is not None)
                if self.closed:
                    return
                job, self.pending = self.pending, None
                self.active = True
            try:
                result = self.analyze(job)
            except Exception as error:
                result = dict(job, error=str(error))
            with self.condition:
                self.active = False
                if self.closed:
                    return
                self.result = result

    def close(self):
        with self.condition:
            self.closed = True
            self.pending = self.result = None
            self.condition.notify()
