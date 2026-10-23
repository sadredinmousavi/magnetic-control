"""Local, opt-in live exchange for one assistant-selected XY nudge at a time."""
import json
from pathlib import Path
import time
import threading
import uuid

import numpy as np

from incremental_xy_control import IncrementalXYFeedback


def atomic_json(path, payload):
    path = Path(path)
    # Unique names prevent GUI/worker publishers from sharing one temporary file.
    temporary = path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    try:
        temporary.write_text(json.dumps(payload, allow_nan=False, indent=2), encoding='utf-8')
        for attempt in range(12):
            try:
                temporary.replace(path)
                return
            except PermissionError:
                # Windows readers, antivirus and OneDrive can briefly deny replacement.
                if attempt == 11:
                    raise
                time.sleep(.01)
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass


class AssistantXYFeedback(IncrementalXYFeedback):
    control_mode = 'anchored_xy_optimizer_online_assistant'

    def __init__(self, *, bridge_directory, **kwargs):
        self.bridge_directory = Path(bridge_directory)
        self.bridge_directory.mkdir(parents=True, exist_ok=True)
        self.session_id = uuid.uuid4().hex
        self.request_id = None
        self.awaiting_command = False
        self.request_center = None
        self.request_created = None
        self.message = 'Waiting for fresh tracking and settling'
        self.history = []
        self.last_published = None
        self.observation = None
        self.bridge_publish_lock = threading.RLock()
        self.bridge_error = None
        super().__init__(**kwargs)

    def begin_point(self, *args, **kwargs):
        self.request_id = None
        self.awaiting_command = False
        return super().begin_point(*args, **kwargs)

    def _reset_slow_assist(self):
        was_ready = getattr(self, 'awaiting_command', False)
        super()._reset_slow_assist()
        self.request_id = None
        self.awaiting_command = False
        if was_ready and self.observation is not None:
            self.publish_bridge()

    def choose_goal(self, waypoint, error, measured, sample):
        now = time.monotonic()
        if (self.request_id is None or self.request_created is None or now-self.request_created > 15.
                or np.linalg.norm(measured-self.request_center) > .00025):
            self.request_id = uuid.uuid4().hex
            self.request_center = measured.copy()
            self.request_created = now
        self.awaiting_command = True
        self.publish(state='Online assistant — settled; waiting for next command', incremental_phase='awaiting_assistant')
        path = self.bridge_directory/'command.json'
        try:
            command = json.loads(path.read_text(encoding='utf-8'))
        except FileNotFoundError:
            return None
        except (ValueError, OSError) as error:
            self.message = f'Ignored unreadable command: {error}'
            return None
        if not isinstance(command, dict):
            self.message = 'Rejected command: expected a JSON object'
            return None
        if command.get('session_id') != self.session_id or command.get('request_id') != self.request_id:
            self.message = 'Ignored command for another run or expired settling request'
            return None
        try:
            issued = float(command['issued_monotonic_s'])
            delta = np.asarray(command['delta_command_mm'], float)
            if not np.isfinite(issued) or not 0 <= now-issued <= 10.:
                raise ValueError('command timestamp is stale or in the future')
            if delta.shape != (2,) or not np.isfinite(delta).all():
                raise ValueError('delta_command_mm must contain two finite values')
            if not 0 < np.linalg.norm(delta) <= self.step_m*1000+1e-9:
                raise ValueError('command exceeds the configured XY step or has zero length')
            if np.dot(delta, error) <= 0:
                raise ValueError('command must point toward the current waypoint')
        except (KeyError, TypeError, ValueError) as error:
            self.message = f'Rejected command: {error}'
            self.request_id = None
            return None
        self.awaiting_command = False
        self.message = 'Accepted one bounded assistant command'
        self.publish(assistant_command=command, assistant_session_id=self.session_id,
                     assistant_request_id=self.request_id)
        self.history.append(dict(request_id=self.request_id, delta_command_mm=delta.tolist(),
                                 center_mm=(measured*1000).tolist(), issued_monotonic_s=issued))
        self.history = self.history[-30:]
        self.request_id = None  # A command cannot be replayed at the next settled position.
        return self.applied_xy+delta*.001

    def _on_speed_brake(self):
        super()._on_speed_brake()
        self.request_id = None
        self.awaiting_command = False
        self.message = 'Speed braking; a new settled request is required before the next nudge'

    def _continuous_update(self, step, commander, case_name, gate, active, report,
                           measured, sample, loss_generation, revision):
        self.observation = dict(
            measured_xy_mm=(np.asarray(measured)*1000).tolist(),
            waypoint_xy_mm=(np.asarray(step.target_xy_m)*1000).tolist(),
            velocity_xy_mm_s=(self.pid.velocity*1000).tolist(),
            observation_monotonic_s=float(sample), ratio=float(step.ratio), angle_deg=step.angle_deg,
            arrival_tolerance_mm=gate.tolerance_mm)
        try:
            super()._continuous_update(step, commander, case_name, gate, active, report,
                                       measured, sample, loss_generation, revision)
        finally:
            if self.goal is not None or self.snapshot().get('incremental_phase') == 'arrival' or self.halt_reason:
                self.awaiting_command = False
            self.publish_bridge()

    def publish_bridge(self, *, stopped=False):
        with self.bridge_publish_lock:
            try:
                self._write_bridge(stopped=stopped)
                self.bridge_error = None
            except OSError as error:
                # A telemetry-file conflict must not crash the control thread or GUI.
                # Cancel outstanding requests and any unfinished ramp; hold the sent pose.
                self.bridge_error = str(error)
                self.awaiting_command = False
                self.request_id = None
                self.goal = None
                self.request_xy = None if self.applied_xy is None else self.applied_xy.copy()
                self.phase_started = self.settle_since = None
                self.completed_step = False
                self.message = f'Live exchange file busy: {error}'
                self.publish(state='Online assistant — live exchange busy; holding last pose')

    def _write_bridge(self, *, stopped=False):
        now = time.monotonic()
        # No live state is advertised as ready after tracking loss/control off.
        ready = self.awaiting_command and self.enabled.is_set() and not stopped and not self.halt_reason
        state = dict(schema_version=1, publisher_revision=2, session_id=self.session_id, request_id=self.request_id,
                     ready_for_command=bool(ready), published_monotonic_s=now,
                     request_created_monotonic_s=self.request_created, max_command_step_mm=self.step_m*1000,
                     applied_command_xy_mm=None if self.applied_xy is None else (self.applied_xy*1000).tolist(),
                     measured=self.observation, status='Stopped' if stopped else self.snapshot().get('state'),
                     scene=self.snapshot().get('assistant_scene'),
                     message=self.message, recent_commands=self.history,
                     command_file=str((self.bridge_directory/'command.json').resolve()))
        atomic_json(self.bridge_directory/'state.json', state)

    def set_enabled(self, enabled):
        super().set_enabled(enabled)
        self.publish_bridge(stopped=not enabled)

