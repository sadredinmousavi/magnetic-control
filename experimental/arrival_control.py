"""Fresh visual arrival confirmation followed by an explicit Next permission."""

import threading
import time

import numpy as np


class ArrivalGate:
    def __init__(self, tolerance_mm=1.0, stable_seconds=.5, *, clock=time.monotonic):
        if not np.isfinite([tolerance_mm, stable_seconds]).all() or tolerance_mm <= 0 or stable_seconds < 0:
            raise ValueError("Arrival tolerance must be positive and hold time nonnegative (finite values).")
        self.tolerance_mm, self.stable_seconds = float(tolerance_mm), float(stable_seconds)
        self.clock = clock
        self.condition = threading.Condition()
        self.target = None
        self.armed_at = None
        self.observed_at = None
        self.inside_since = None
        self.error_mm = None
        self.waiting_for_next = False
        self.next_requested = False

    def arm(self, target_xy_m):
        with self.condition:
            self.target = np.asarray(target_xy_m, dtype=float)
            self.armed_at = self.clock()
            self.observed_at = self.inside_since = self.error_mm = None
            self.waiting_for_next = self.next_requested = False
            self.condition.notify_all()

    def observe(self, positions_xy_m, observed_at=None):
        with self.condition:
            if self.target is None:
                return
            timestamp = self.clock() if observed_at is None else observed_at
            if positions_xy_m is None:
                self.inside_since = self.error_mm = None
                self.next_requested = False
                self.condition.notify_all()
                return
            if timestamp <= self.armed_at or (self.observed_at is not None and timestamp <= self.observed_at):
                return
            positions = np.asarray(positions_xy_m, dtype=float)
            if positions.ndim != 2 or positions.shape[1] != 2 or not len(positions) or not np.isfinite(positions).all():
                self.inside_since = self.error_mm = None
                self.next_requested = False
                self.condition.notify_all()
                return
            if self.observed_at is not None and timestamp - self.observed_at > 2:
                self.inside_since = None
                self.next_requested = False
            self.observed_at = timestamp
            self.error_mm = float(np.linalg.norm(positions - self.target, axis=1).max() * 1000)
            if self.error_mm <= self.tolerance_mm:
                if self.inside_since is None:
                    self.inside_since = timestamp
            else:
                self.inside_since = None
                self.next_requested = False
            self.condition.notify_all()

    def _ready(self):
        return (self.observed_at is not None and self.inside_since is not None
                and self.clock() - self.observed_at <= 2
                and self.observed_at - self.inside_since >= self.stable_seconds)

    @property
    def ready(self):
        with self.condition:
            return self._ready()

    def wait_for_arrival(self, active):
        with self.condition:
            while active():
                if self._ready():
                    self.waiting_for_next = True
                    return True
                self.condition.wait(.1)
            return False

    def request_next(self):
        with self.condition:
            if not self.waiting_for_next or self.next_requested or not self._ready():
                return False
            self.next_requested = True
            self.condition.notify_all()
            return True

    def wait_for_next(self, active):
        with self.condition:
            while active():
                if self.next_requested and self._ready():
                    self.next_requested = self.waiting_for_next = False
                    return True
                if not self._ready():
                    self.next_requested = False
                self.condition.wait(.1)
            return False
