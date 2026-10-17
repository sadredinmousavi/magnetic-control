"""Lightweight, asynchronous diagnostic sessions; never issues hardware commands."""

import csv
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import queue
import threading
import time

import numpy as np

from command_recording import TRACKING_FIELDS


def serializable(value):
    if is_dataclass(value):
        value = asdict(value)
    if isinstance(value, dict):
        return {str(key): serializable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [serializable(item) for item in value]
    if isinstance(value, np.ndarray):
        return serializable(value.tolist())
    if isinstance(value, np.generic):
        return serializable(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


class ControlDiagnostics:
    """One frame record per GUI-accepted frame, plus independent motor events.

    Bounded memory, no image copies/encoding. Overflow is reported, never hidden.
    Close drains queued records and writes final counts to session.json.
    """

    def __init__(self, directory, metadata, *, capacity=4096):
        root = Path(directory)
        root.mkdir(parents=True, exist_ok=True)
        name = datetime.now().strftime('%Y-%m-%d_%H-%M-%S_%f')
        self.path = root / name
        self.path.mkdir()
        self.metadata = serializable(metadata)
        self.started_at = datetime.now(timezone.utc).isoformat()
        self.started_monotonic = time.monotonic()
        self.error = None
        self.accepted = self.written = self.rejected = self.frames = 0
        self.closed = False
        self.lock = threading.Lock()
        self.pending = queue.Queue(maxsize=capacity)
        # Open synchronously so permission/path errors reach the Start button.
        self.stream = (self.path / 'records.jsonl').open('w', encoding='utf-8')
        try:
            self.csv_file = (self.path / 'frames.csv').open('w', newline='', encoding='utf-8')
        except Exception:
            self.stream.close()
            raise
        self.fields = list(dict.fromkeys(['diagnostic_frame', 'receipt_monotonic_s',
                      'elapsed_s', 'width_px', 'height_px', 'tracking_active',
                      'tracking_status', 'processing_s', 'inter_frame_s',
                      'measurement_monotonic_s', 'measurement_fresh',
                      'force_observation_monotonic_s'] + list(TRACKING_FIELDS)))
        self.csv = csv.DictWriter(self.csv_file, fieldnames=self.fields, extrasaction='ignore')
        try:
            self.csv.writeheader()
            self._manifest(False)
        except Exception:
            self.stream.close()
            self.csv_file.close()
            raise
        self.worker = threading.Thread(target=self._write, name='control-diagnostics', daemon=True)
        self.worker.start()

    def record(self, kind, payload):
        with self.lock:
            if self.closed:
                return
            if self.error:
                self.rejected += 1
                return
            record = dict(kind=kind, monotonic_s=time.monotonic(),
                          utc=datetime.now(timezone.utc).isoformat(),
                          record_number=self.accepted, data=serializable(payload))
            try:
                self.pending.put_nowait(record)
                self.accepted += 1
            except queue.Full:
                self.rejected += 1
                self.error = 'Diagnostic queue overflow; recording incomplete. Stop/save diagnostics.'

    def _write(self):
        try:
            while True:
                record = self.pending.get()
                if record is None:
                    break
                self.stream.write(json.dumps(record, allow_nan=False) + '\n')
                if record['kind'] == 'frame':
                    data = record['data']
                    common = {key: data.get(key) for key in self.fields if key in data}
                    control = data.get('control', {})
                    common.update({key: control.get(key) for key in (
                        'measurement_monotonic_s', 'measurement_fresh', 'force_observation_monotonic_s')})
                    missing = dict(group_valid=0, detected_robots=0,
                                   sequence_running=data.get('running'), control_state=control.get('state'),
                                   control_mode=control.get('mode'))
                    step = data.get('active_step') or {}
                    target = step.get('target_xy_m')
                    if target is not None:
                        missing.update(desired_x_mm=target[0]*1000, desired_y_mm=target[1]*1000)
                    for key in ('requested_force_n', 'predicted_force_n',
                                'velocity_xy_mm_s', 'pid_force_n', 'integral_force_n'):
                        missing[key] = json.dumps(control.get(key))
                    missing['calculated_motor_angles_deg'] = json.dumps(control.get('calculated_angles'))
                    missing['sent_motor_angles_deg'] = json.dumps(control.get('sent_angles'))
                    for row in data.get('tracking_rows') or [missing]:
                        self.csv.writerow(dict(row, **common))
                    self.frames += 1
                self.written += 1
                if self.written % 20 == 0:
                    self.stream.flush()
                    self.csv_file.flush()
        except Exception as error:
            self.error = f'Diagnostic writer failed: {error}'
        finally:
            for stream in (self.stream, self.csv_file):
                try:
                    stream.close()
                except OSError as error:
                    self.error = f'Diagnostic close failed: {error}'

    def _manifest(self, complete, final_metadata=None):
        data = dict(schema_version=1, started_utc=self.started_at,
                    started_monotonic_s=self.started_monotonic,
                    ended_utc=datetime.now(timezone.utc).isoformat() if self.closed else None,
                    complete=complete, error=self.error, records_accepted=self.accepted,
                    records_written=self.written, records_rejected=self.rejected,
                    frames_written=self.frames, initial=self.metadata,
                    final=serializable(final_metadata),
                    notes=['Frames are GUI-accepted camera frames, not every sensor exposure.',
                           'monotonic timestamps share one process clock; UTC is for reference.',
                           'Telemetry is the latest controller snapshot, not necessarily calculated from this frame.',
                           'Sent means serial writes completed, not motor acknowledgement or measured rotation.',
                           'Model force units N; position m internally, CSV mm; angles deg.',
                           'No images/video or credentials are recorded.'])
        (self.path / 'session.json').write_text(json.dumps(data, indent=2, allow_nan=False), encoding='utf-8')

    def close(self, final_metadata=None):
        with self.lock:
            if self.closed:
                return
            self.closed = True
        while self.worker.is_alive():
            try:
                self.pending.put(None, timeout=.1)
                break
            except queue.Full:
                continue
        self.worker.join()
        self._manifest(self.error is None and self.written == self.accepted, final_metadata)
