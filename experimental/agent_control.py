"""Local file mailbox for supervising GUI005; all GUI actions run on Tk's thread.

Opt in with ``gui005_agent.py``. No network listener or new packages.
The existing Commands worker owns synthesis, feedback and serial transmission.
"""

import argparse
import json
import math
import os
from pathlib import Path
import time
import uuid
from contextlib import contextmanager
from unittest.mock import patch

import cv2
import numpy as np


PROJECT = Path(__file__).resolve().parents[1]
DIRECTORY = PROJECT / 'outputs' / 'gui005_control'


def read_json(path):
    """Allow the GUI to replace its status file while Windows clients read it."""
    if os.name != 'nt':
        return json.loads(Path(path).read_text(encoding='utf-8'))
    import ctypes
    from ctypes import wintypes
    import msvcrt
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    create = kernel.CreateFileW
    create.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD,
                       ctypes.c_void_p, wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]
    create.restype = wintypes.HANDLE
    handle = create(str(Path(path).resolve()), 0x80000000, 7, None, 3, 0x80, None)
    if handle == ctypes.c_void_p(-1).value:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        descriptor = msvcrt.open_osfhandle(handle, os.O_RDONLY)
    except Exception:
        close = kernel.CloseHandle
        close.argtypes = [wintypes.HANDLE]
        close(handle)
        raise
    with os.fdopen(descriptor, encoding='utf-8') as stream:
        return json.load(stream)


@contextmanager
def gui_dialogs(module, load_path=None):
    """Supply a filename/report errors for one synchronous GUI action only."""
    def dialog(owner, function, title=None, message=None, **kwargs):
        if function is module.filedialog.askopenfilename and load_path is not None:
            return str(load_path)
        if function is module.messagebox.showerror:
            raise ValueError(f'{title}: {message}')
        raise ValueError('This operation requires an interactive GUI dialog.')
    with patch.object(module, 'show_view_dialog', dialog):
        yield


def plain(value):
    if isinstance(value, dict):
        return {str(key): plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [plain(item) for item in value]
    if isinstance(value, np.generic):
        return plain(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def atomic_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(plain(value), allow_nan=False), encoding='utf-8')
    for attempt in range(6):
        try:
            temporary.replace(path)
            return
        except PermissionError:
            if attempt == 5:
                raise
            time.sleep(.02)


class AgentControl:
    def __init__(self, gui, directory=DIRECTORY, *, camera_delay_s=1.):
        if not math.isfinite(camera_delay_s) or camera_delay_s < 0:
            raise ValueError('Estimated camera delay must be finite and nonnegative.')
        self.camera_delay_s = float(camera_delay_s)
        self.gui = gui
        self.directory = Path(directory)
        self.session = uuid.uuid4().hex
        self.mailbox = self.directory / self.session
        self.mailbox.mkdir(parents=True)
        self.closed = False
        self.last_io_error = None
        self.stopped_at = 0
        self.after_id = None
        self.publish()
        self.after_id = gui.root.after(250, self.poll)

    def status(self):
        gui, tab = self.gui, self.gui.commands
        frame_age = (None if tab.frame_received_at is None else
                     time.monotonic() - tab.frame_received_at)
        fresh = gui.selected_camera_running() and frame_age is not None and frame_age <= 2
        try:
            setup = tab.current_setup()
            setup_error = None
        except (ValueError, TypeError, KeyError) as error:
            setup, setup_error = None, str(error)
        steps = []
        for index, step in enumerate(tab.steps):
            steps.append(dict(point=index + 1, target_xy_mm=np.asarray(step.target_xy_m) * 1000,
                              state=tab.table.set(str(index), 'state'),
                              motor_angles_deg=tab.table.set(str(index), 'angles')))
        return dict(session=self.session, published_at=time.time(), closed=self.closed,
                    serial_connected=bool(gui.port and getattr(gui.port, 'is_open', False)),
                    camera_source=gui.camera_source_key, camera_running=gui.selected_camera_running(),
                    frame_age_s=frame_age, frame_size_px=None if tab.frame is None else tab.frame.shape[1::-1],
                    measurements_fresh=fresh, transport_fresh=fresh,
                    camera_delay_estimate_s=self.camera_delay_s,
                    measurement_age_estimate_s=None if frame_age is None else frame_age + self.camera_delay_s,
                    timestamp_basis='GUI receipt; sensor capture time unavailable',
                    calibration=setup, calibration_error=setup_error,
                    running=tab.running, precalculating=tab.precalculating,
                    control_requested=tab.control_requested,
                    stop_requested=tab.stop_event.is_set(), last_io_error=self.last_io_error,
                    active_point=None if tab.active_index is None else tab.active_index + 1,
                    sequence=str(tab.sequence_path) if tab.sequence_path else None, points=steps,
                    group_xy_mm=None if not fresh or tab.tracking_group_xy is None else np.asarray(tab.tracking_group_xy) * 10,
                    detections=tab.tracking_detections if fresh else [],
                    feedback={} if tab.feedback is None else tab.feedback.snapshot(),
                    message=tab.status.cget('text'))

    def publish(self):
        atomic_json(self.directory / 'status.json', self.status())

    def project_file(self, path):
        candidate = Path(path)
        if not candidate.is_absolute():
            candidate = PROJECT / candidate
        candidate = candidate.resolve()
        if not candidate.is_relative_to(PROJECT) or not candidate.is_file():
            raise ValueError('Choose an existing file inside this project.')
        return candidate

    def dispatch(self, request):
        if request.get('session') != self.session:
            raise ValueError('Request belongs to a different GUI session.')
        expiry = request.get('expires_at')
        if not isinstance(expiry, (float, int)) or not math.isfinite(expiry) or time.time() > expiry:
            raise ValueError('Request expired; inspect status before retrying.')
        action = request.get('action')
        args = request.get('args', {})
        tab = self.gui.commands
        if action == 'stop':
            tab.stop()
            self.stopped_at = time.time()
            return dict(message='Stop requested. Last sent motor pose remains applied.')
        if action == 'snapshot':
            if tab.frame is None or tab.frame_received_at is None or time.monotonic() - tab.frame_received_at > 2:
                raise ValueError('No fresh camera frame is available.')
            ok, encoded = cv2.imencode('.jpg', tab.frame)
            if not ok:
                raise ValueError('Camera snapshot encoding failed.')
            path = self.mailbox / (uuid.uuid4().hex + '.jpg')
            path.write_bytes(encoded.tobytes())
            return dict(path=str(path), frame_received_monotonic=tab.frame_received_at,
                        camera_delay_estimate_s=self.camera_delay_s)
        if action == 'show-camera':
            tab.open_camera_popout()
            return dict(message='Commands camera pop-out opened.')
        created = request.get('created_at', 0)
        if not isinstance(created, (int, float)) or not math.isfinite(created) or created <= self.stopped_at:
            raise ValueError('Request predates the last Stop; issue a new command.')
        if action == 'next':
            if not tab.running or tab.stop_event.is_set() or tab.arrival_gate is None:
                raise ValueError('No active run to advance.')
            self.require_observation_after_send()
            if not tab.arrival_gate.request_next():
                raise ValueError('Next is already queued or unavailable.')
            return dict(message='Next requested for current run.')
        if action == 'control':
            enabled = args.get('enabled')
            if type(enabled) is not bool:
                raise ValueError('enabled must be a boolean.')
            if tab.precalculating:
                raise ValueError('Wait for precalculation to finish.')
            if enabled != tab.control_requested:
                tab.toggle_control()
            return dict(control_requested=tab.control_requested)
        if tab.running or tab.precalculating or self.gui.calibration.axis_busy:
            raise ValueError('Stop the active operation before changing the experiment.')
        if action == 'load-sequence':
            tab.units_var.set('Auto')
            tab.load_path(self.project_file(args['path']))
            return dict(points=len(tab.steps))
        if action == 'load-setup':
            import closed_loop_calibration
            with gui_dialogs(closed_loop_calibration, self.project_file(args['path'])):
                self.gui.calibration.load_setup()
            tab.refresh_calibration()
            return dict(message='Calibration loaded and tracking refreshed.')
        if action in ('run-selected', 'run-all'):
            self.require_observation_after_send()
            if action == 'run-selected':
                point = args.get('point')
                if type(point) is not int or not 1 <= point <= len(tab.steps):
                    raise ValueError('point must be a one-based index in the loaded sequence.')
                tab.table.selection_set(str(point - 1))
            import command_tab
            tab.open_camera_popout()
            with gui_dialogs(command_tab):
                tab.run(selected_only=action == 'run-selected')
            if not tab.running:
                raise ValueError('Run did not start: ' + tab.status.cget('text'))
            return dict(message='Run started; consult status for calculation/send outcomes.')
        raise ValueError('Unknown action: ' + str(action))

    def require_observation_after_send(self):
        """Let the delayed view catch up before another supervised command.

        This is not delay compensation for the existing continuous PID worker.
        """
        tab = self.gui.commands
        if tab.feedback is None:
            return
        sent = tab.feedback.snapshot().get('sent_time_s')
        if sent is None:
            return
        minimum = sent + self.camera_delay_s + .5
        if tab.frame_received_at is None or tab.frame_received_at < minimum:
            raise ValueError('Wait for camera observations after the last send: estimated delay plus 0.5 s margin.')

    def poll(self):
        self.after_id = None
        if self.closed:
            return
        try:
            # Claim before execution: a request can never be sent twice by this process.
            requests = sorted(self.mailbox.glob('*.request.json'))
            # Stop takes precedence over pending motion requests.
            requests.sort(key=lambda path: (not path.name.startswith('stop-'), path.name))
            for path in requests[:1]:
                claimed = path.with_suffix('.claimed')
                path.replace(claimed)
                try:
                    request = json.loads(claimed.read_text(encoding='utf-8'))
                    result = dict(ok=True, result=self.dispatch(request))
                except Exception as error:
                    result = dict(ok=False, error=str(error))
                atomic_json(path.with_name(path.name.replace('.request.json', '.response.json')), result)
            self.publish()
        except (OSError, ValueError) as error:
            self.last_io_error = dict(time=time.time(), error=repr(error))
            self.gui.commands.stop()
            self.gui.commands.status.config(text=f'Local control IO failed; stop requested: {error}')
            try:
                with (self.directory / 'io_errors.log').open('a', encoding='utf-8') as stream:
                    stream.write(f'{time.time()} {error!r}\n')
            except OSError:
                pass
        finally:
            if not self.closed:
                self.after_id = self.gui.root.after(250, self.poll)

    def close(self):
        self.closed = True
        if self.after_id is not None:
            self.gui.root.after_cancel(self.after_id)
        self.publish()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('status', 'snapshot', 'show-camera', 'load-setup', 'load-sequence',
                                         'run-selected', 'run-all', 'next', 'control', 'stop'))
    parser.add_argument('--path')
    parser.add_argument('--point', type=int)
    parser.add_argument('--enabled', choices=('true', 'false'))
    options = parser.parse_args()
    try:
        status = read_json(DIRECTORY / 'status.json')
        if status['closed'] or time.time() - status['published_at'] > 3:
            raise ValueError('GUI005 control is offline. Start gui005_agent.py.')
        if options.action == 'status':
            print(json.dumps(status, indent=2))
            return
        args = dict(path=options.path, point=options.point)
        if options.enabled is not None:
            args['enabled'] = options.enabled == 'true'
        name = ('stop-' if options.action == 'stop' else '') + uuid.uuid4().hex
        mailbox = DIRECTORY / status['session']
        request_path = mailbox / (name + '.request.json')
        response_path = mailbox / (name + '.response.json')
        atomic_json(request_path, dict(session=status['session'], created_at=time.time(), expires_at=time.time() + 5,
                                       action=options.action, args=args))
        deadline = time.monotonic() + 8
        while time.monotonic() < deadline:
            if response_path.exists():
                response = json.loads(response_path.read_text(encoding='utf-8'))
                print(json.dumps(response, indent=2))
                if not response['ok']:
                    raise SystemExit(1)
                return
            time.sleep(.1)
        raise ValueError('No acknowledgement. Inspect status before retrying; do not assume no motor command was sent.')
    except (OSError, ValueError, KeyError) as error:
        parser.exit(1, str(error) + '\n')


if __name__ == '__main__':
    main()
