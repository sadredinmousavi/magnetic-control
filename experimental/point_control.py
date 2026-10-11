"""Reusable point synthesis and serialized motor commands for GUI005 tabs."""

from pathlib import Path
import sys
import threading
import time

import numpy as np

PROJECT_DIR = str(Path(__file__).resolve().parents[1])
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from case_loader import load_case
from control_workflow import calculate_point_control


class PointCommander:
    """Shared across tabs; call move from a worker, not the Tk UI thread."""

    def __init__(self, port_provider, zero_angle_deg=180):
        self.port_provider = port_provider
        self.zero_angle_deg = zero_angle_deg
        self.lock = threading.Lock()

    def calculate(self, case_name, target_xy_m, *, ratio=1.0, angle_deg=None):
        return calculate_point_control(load_case(case_name), target_xy_m,
                                       ratio=ratio, angle_deg=angle_deg)

    def move(self, case_name, target_xy_m, *, ratio=1.0, angle_deg=None,
             settle_seconds=0, active=lambda: True):
        """Calculate fresh motor angles, send them, then wait interruptibly.

        Cancellation skips unsent commands; it cannot undo an already sent pose.
        Returns optimization information for logging/preview in any caller tab.
        """
        from helpers import moveArray

        settle_seconds = float(settle_seconds)
        if not np.isfinite(settle_seconds) or settle_seconds < 0:
            raise ValueError("Settle time must be finite and nonnegative.")
        with self.lock:
            if not active():
                return None
            port = self.port_provider()
            if port is None or not getattr(port, "is_open", False):
                raise RuntimeError("Connect the serial controller in Connections first.")
            result = self.calculate(case_name, target_xy_m, ratio=ratio, angle_deg=angle_deg)
            if not active():
                return None
            if self.port_provider() is not port or not getattr(port, "is_open", False):
                raise RuntimeError("The controller disconnected before the move.")
            angles = np.asarray(result["angles_deg"], dtype=float)
            if angles.ndim != 1 or not angles.size or not np.isfinite(angles).all():
                raise ValueError("Calculation did not produce valid motor angles.")
            moveArray(port, angles, 0, self.zero_angle_deg)
            deadline = time.monotonic() + settle_seconds
            while active() and time.monotonic() < deadline:
                time.sleep(min(.05, max(0, deadline - time.monotonic())))
            return result
