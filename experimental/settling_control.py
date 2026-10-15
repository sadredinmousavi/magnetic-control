"""Speed-based settling from ordered, calibrated robot positions (mm)."""
from collections import deque
import threading

import numpy as np


class SettlingMonitor:
    def __init__(self, speed_mm_s=.5, window_s=.5, hold_s=.5, minimum_wait_s=.5):
        values = np.asarray([speed_mm_s, window_s, hold_s, minimum_wait_s], float)
        if not np.isfinite(values).all() or np.any(values[:3] <= 0) or values[3] < 0:
            raise ValueError('Settle speed, window and hold must be positive; minimum wait must be nonnegative (finite).')
        self.speed_limit, self.window, self.hold, self.minimum_wait = values
        self.lock = threading.RLock()
        self.reset()

    def reset(self, sent_at=None):
        with self.lock:
            self.sent_at = sent_at
            self.history, self.slow_positions = deque(), deque()
            self.low_since = self.last_sample = None
            self.speeds = None
            self.ready = False
            self.state = 'Waiting for tracking'

    def observe(self, positions_mm, timestamp):
        with self.lock:
            if positions_mm is None:
                self.reset(self.sent_at)
                return
            points = np.asarray(positions_mm, float)
            if (points.ndim != 2 or points.shape[1] != 2 or not len(points)
                    or not np.isfinite(points).all() or not np.isfinite(timestamp)):
                self.reset(self.sent_at)
                return
            if self.sent_at is not None and timestamp <= self.sent_at:
                return
            if self.last_sample is not None and timestamp <= self.last_sample:
                return
            if self.history and (points.shape != self.history[-1][1].shape or timestamp-self.last_sample > 1. + 1e-6):
                self.reset(self.sent_at)
            self.last_sample = timestamp
            self.history.append((timestamp, points.copy()))
            while len(self.history) > 2 and self.history[1][0] <= timestamp-self.window:
                self.history.popleft()
            span = timestamp-self.history[0][0]
            if len(self.history) < 3 or span + 1e-9 < self.window:
                self.ready = False
                self.low_since = self.speeds = None
                self.slow_positions.clear()
                self.state = ('Minimum wait' if self.sent_at is not None and timestamp-self.sent_at < self.minimum_wait
                              else 'Measuring speed')
                return
            times = np.asarray([sample[0] for sample in self.history])
            times -= times.mean()
            positions = np.asarray([sample[1] for sample in self.history])
            mean = positions.mean(axis=0)
            velocity = np.sum(times[:,None,None]*(positions-mean), axis=0) / np.sum(times*times)
            residual = positions - (mean + times[:,None,None]*velocity)
            # Regression suppresses frame noise; residual motion prevents an
            # oscillation returning to its start from appearing stationary.
            self.speeds = np.maximum(np.linalg.norm(velocity, axis=1),
                                     2*np.linalg.norm(residual, axis=2).max(axis=0)/span)
            waiting = self.sent_at is not None and timestamp-self.sent_at < self.minimum_wait
            if waiting or np.max(self.speeds) > self.speed_limit:
                self.low_since = None
                self.slow_positions.clear()
                self.ready = False
                self.state = 'Minimum wait' if waiting else 'Moving'
                return
            if self.low_since is None:
                self.low_since = timestamp
            self.slow_positions.append((timestamp, points.copy()))
            while len(self.slow_positions) > 1 and self.slow_positions[0][0] < timestamp-self.hold:
                self.slow_positions.popleft()
            self.ready = timestamp-self.low_since + 1e-9 >= self.hold
            self.state = 'Settled' if self.ready else 'Settling'

    def snapshot(self, now):
        with self.lock:
            fresh = self.last_sample is not None and -1e-9 <= now-self.last_sample <= 1.
            elapsed = 0. if self.low_since is None else min(self.hold, self.last_sample-self.low_since)
            return dict(ready=bool(self.ready and fresh), state=self.state if fresh else 'Waiting for tracking',
                        speed_mm_s=None if self.speeds is None or not fresh else float(np.max(self.speeds)),
                        hold_elapsed_s=float(elapsed) if fresh else 0.,
                        positions_mm=(np.mean([item[1] for item in self.slow_positions], axis=0)
                                      if self.ready and fresh and self.slow_positions else None))
