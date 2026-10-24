"""Stream native-resolution camera/overlay videos and timestamped tracking CSV."""

import csv
import copy
import json
import queue
import threading
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
import time

import cv2
import numpy as np


def video_crop_circle(size, *, corners=None, center=None, radius=300, fit=False):
    width, height = size
    if fit:
        points = np.asarray(corners, np.float32)
        if points.shape != (4, 2) or not np.isfinite(points).all():
            raise ValueError('Fit workspace needs four calibrated pins.')
        center, radius = cv2.minEnclosingCircle(points)
        center = tuple(np.rint(center).astype(int))
        radius = np.ceil(np.linalg.norm(points - center, axis=1).max()) + 35
    if center is None:
        center = (width / 2, height / 2)
    values = np.asarray([*center, radius], float)
    if not np.isfinite(values).all() or not 1 <= radius <= max(size):
        raise ValueError('Crop radius must be between 1 px and the largest camera dimension.')
    if not (0 <= center[0] < width and 0 <= center[1] < height):
        raise ValueError('Crop center must be inside the camera frame.')
    return (int(round(center[0])), int(round(center[1])), int(np.ceil(radius)))


class CircularVideoCrop:
    """Cached source-pixel circular crop; pad missing image area, never stretch it."""
    def __init__(self, size, circle):
        self.x, self.y, self.radius = circle
        r = self.radius
        self.size = (2 * r, 2 * r)
        yy, xx = np.ogrid[:2*r, :2*r]
        self.inside = (xx-r)**2 + (yy-r)**2 <= r*r
        width, height = size
        self.left, self.top = self.x-r, self.y-r
        self.bounds = max(0,self.left), max(0,self.top), min(width,self.x+r), min(height,self.y+r)

    def apply(self, image, background=(0,0,0)):
        result = np.empty((*self.size[::-1],3), dtype=image.dtype)
        result[:] = background
        x0,y0,x1,y1 = self.bounds
        if x1 > x0 and y1 > y0:
            result[y0-self.top:y1-self.top,x0-self.left:x1-self.left] = image[y0:y1,x0:x1]
        result[~self.inside] = background
        return result


TRACKING_FIELDS = ('cargo_shape cargo_angle_deg cargo_angle_valid cargo_length_mm cargo_width_mm '
                   'control_target control_x_mm control_y_mm cargo_valid cargo_x_px cargo_y_px cargo_x_mm cargo_y_mm spread_ratio strong_spread_mm weak_spread_mm '
                   'frame time_s robot_id robot_x_px robot_y_px robot_x_mm robot_y_mm '
                   'group_x_mm group_y_mm expected_robots detected_robots group_valid '
                   'active_point desired_x_mm desired_y_mm error_mm roi_x_px roi_y_px '
                   'roi_radius_px sequence_running arrival_ready arrival_max_error_mm '
                   'desired_ratio desired_angle_deg recording_frame recording_time_s '
                   'crop_center_x_px crop_center_y_px crop_radius_px slow_assist_active slow_assist_scale '
                   'control_enabled control_state control_mode requested_force_n predicted_force_n force_residual_n '
                   'magnet_cosine_inputs stiffness_eigenvalues_n_m control_calculation_time_s force_compensated '
                   'velocity_xy_mm_s position_error_mm pid_force_n integral_force_n inherited_integral_force_n '
                   'observation_age_s control_cycle_time_s force_kp_n_m force_ki_n_m_s force_kv_n_s_m '
                   'robot_speed_mm_s hold_speed_mm_s speed_limit_mm_s speed_braking '
                   'correction_x_mm correction_y_mm commanded_x_mm commanded_y_mm '
                   'proposed_xy_mm calculated_motor_angles_deg sent_motor_angles_deg motor_sent_time_s orientation_measured_deg '
                   'integral_xy_mm anchor_xy_mm allocation_fraction sent_optimizer_xy_mm waypoints_skipped '
                   'effective_speed_limit_mm_s command_increment_mm command_offset_xy_mm command_limits '
                   'pid_p_increment_mm pid_i_increment_mm pid_d_increment_mm pending_increment_mm').split()


def draw_command_overlay(image, setup, pixels, active_index, detections, tracker=None,
                         *, show_path=False, show_roi=False, trails=None, commanded_pixel=None, geometry=None, cargo=None,
                         show_pins=True, show_points=True, show_labels=True, robot_marker='Cross',
                         center_trail=None, show_group_center=True, show_spread_axes=True,
                         desired_path_dashed=False, desired_point_radius=2, group_center_size=12,
                         annotation_thickness=1, group_center_thickness=1):
    """Draw in source-image pixels, independently of GUI zoom or visibility."""
    if show_pins and setup is not None and len(setup['corners_px']) == 4:
        cv2.polylines(image, [np.asarray(setup['corners_px'], np.int32)], True, (153, 85, 153), 1)
    if pixels is not None:
        points = np.rint(pixels).astype(np.int32)
        if show_path and len(points) > 1:
            if desired_path_dashed:
                # Keep dash spacing continuous through corners of the path.
                distance = 0.
                for start, end in zip(points[:-1], points[1:]):
                    delta = end.astype(float) - start
                    length = float(np.linalg.norm(delta))
                    dash_start = np.floor(distance / 16.) * 16.
                    while dash_start < distance + length:
                        offset = max(0., dash_start - distance)
                        stop = min(length, dash_start + 10. - distance)
                        if stop > offset:
                            a = tuple(np.rint(start + delta * (offset / length)).astype(int))
                            b = tuple(np.rint(start + delta * (stop / length)).astype(int))
                            cv2.line(image, a, b, (136, 136, 51), annotation_thickness)
                        dash_start += 16.
                    distance += length
            else:
                cv2.polylines(image, [points], False, (136, 136, 51), annotation_thickness)
        for x, y in (points if show_points else []):
            outside = (setup is not None and cv2.pointPolygonTest(
                np.asarray(setup['corners_px'], np.float32), (float(x), float(y)), True) < -1e-3)
            cv2.circle(image, (int(x), int(y)), desired_point_radius,
                       (68, 68, 255) if outside else (139, 139, 0), annotation_thickness)
        if show_points and active_index is not None and active_index < len(points):
            point = tuple(points[active_index])
            cv2.drawMarker(image, point, (0, 153, 255), cv2.MARKER_CROSS, 6, 1, cv2.LINE_AA)
            if show_labels:
                cv2.putText(image, f'P{active_index + 1}', (point[0] + 7, point[1] - 7),
                        cv2.FONT_HERSHEY_SIMPLEX, .32, (0, 153, 255), 1, cv2.LINE_AA)
    if commanded_pixel is not None:
        point = tuple(np.rint(commanded_pixel).astype(int))
        cv2.drawMarker(image, point, (221, 85, 255), cv2.MARKER_DIAMOND, 8, 1, cv2.LINE_AA)
        if show_labels:
            cv2.putText(image, 'C', (point[0]+6, point[1]+10), cv2.FONT_HERSHEY_SIMPLEX,
                    .32, (221, 85, 255), 1, cv2.LINE_AA)
    if show_roi and tracker is not None:
        cv2.circle(image, tuple(np.rint(tracker.center).astype(int)), round(tracker.radius), (68, 153, 68), annotation_thickness)
    trail_layers = [(trail, (68, 153, 68)) for trail in (trails or {}).values()]
    if center_trail is not None:
        trail_layers.append((center_trail, (255, 255, 255)))
    for trail, color in trail_layers:
        previous = None
        for point in trail:
            current = None if point is None else tuple(np.rint(point).astype(int))
            if previous is not None and current is not None:
                cv2.line(image, previous, current, color, 1)
            previous = current
    for item in detections:
        x, y = np.rint(item['center']).astype(int)
        if robot_marker == 'Rectangle':
            bx, by, width, height = np.rint(item.get('bbox', (x-3, y-3, 6, 6))).astype(int)
            cv2.rectangle(image, (bx, by), (bx+width, by+height), (102, 255, 0), 1)
        else:
            cv2.drawMarker(image, (x, y), (102, 255, 0), cv2.MARKER_CROSS, 4, 1, cv2.LINE_AA)
        text = f"R{item['robot_id']}"
        if item.get('xy_cm') is not None:
            text += f" ({item['xy_cm'][0]*10:.1f}, {item['xy_cm'][1]*10:.1f}) mm"
        if show_labels:
            cv2.putText(image, text, (x + 4, y + 8), cv2.FONT_HERSHEY_SIMPLEX, .32, (102, 255, 0), 1, cv2.LINE_AA)
    from swarm_cargo import draw_measurements
    return draw_measurements(image, geometry, cargo, show_labels=show_labels,
                             show_group_center=show_group_center, show_spread_axes=show_spread_axes,
                             group_center_size=group_center_size, group_center_thickness=group_center_thickness)


class CommandRecording:
    def __init__(self, base_path, size, fps=15, *, clock=time.monotonic, crop_circle=None):
        if not np.isfinite(fps) or fps <= 0:
            raise ValueError('Recording FPS must be positive and finite.')
        self.size = tuple(size)
        self.crop = None if crop_circle is None else CircularVideoCrop(self.size, video_crop_circle(
            self.size, center=crop_circle[:2], radius=crop_circle[2]))
        self.output_size = self.size if self.crop is None else self.crop.size
        self.encoded_size = tuple(value + value % 2 for value in self.output_size)
        self.clock, self.started_at = clock, clock()
        self.frame_count = 0
        self.writers = []
        self.csv_file = None
        self.csv_writer = None
        base = Path(base_path).with_suffix('')
        self.paths = [base.with_name(base.name + suffix) for suffix in
                      ('_camera.mp4', '_annotated.mp4', '_overlay.mp4', '_tracking.csv')]
        if any(path.exists() for path in self.paths):
            raise ValueError('Recording files already exist. Choose a new recording name.')
        try:
            for path in self.paths[:3]:
                writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'mp4v'), fps, self.encoded_size)
                self.writers.append(writer)
                if not writer.isOpened():
                    raise ValueError(f'Cannot create video: {path}. Check the codec and destination.')
            self.csv_file = self.paths[3].open('x', newline='', encoding='utf-8')
        except Exception:
            self.close()
            raise

    def write(self, frame, annotated, overlay, rows):
        if tuple(frame.shape[1::-1]) != self.size:
            raise ValueError('Camera size changed; recording stopped to protect video dimensions.')
        for index, (writer, image) in enumerate(zip(self.writers, (frame, annotated, overlay))):
            if self.crop is not None:
                image = self.crop.apply(image, (255,0,0) if index == 2 else (0,0,0))
            padded = cv2.copyMakeBorder(image, 0, self.encoded_size[1] - self.output_size[1],
                                        0, self.encoded_size[0] - self.output_size[0], cv2.BORDER_CONSTANT)
            writer.write(padded)
        timestamp = self.clock() - self.started_at
        for row in rows or [{}]:
            measurement = dict(row, recording_frame=self.frame_count, recording_time_s=timestamp)
            measurement.update(crop_center_x_px='' if self.crop is None else self.crop.x,
                               crop_center_y_px='' if self.crop is None else self.crop.y,
                               crop_radius_px='' if self.crop is None else self.crop.radius)
            if self.csv_writer is None:
                self.csv_writer = csv.DictWriter(self.csv_file, fieldnames=TRACKING_FIELDS)
                self.csv_writer.writeheader()
            self.csv_writer.writerow(measurement)
        self.csv_file.flush()
        self.frame_count += 1

    def close(self):
        for writer in self.writers:
            writer.release()
        self.writers = []
        if self.csv_file is not None:
            self.csv_file.close()
            self.csv_file = None


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f'Cannot serialize {type(value).__name__}')


RAW_FIELDS = TRACKING_FIELDS + [
    'capture_monotonic_s', 'robot_bbox_x_px', 'robot_bbox_y_px', 'robot_bbox_width_px',
    'robot_bbox_height_px', 'group_x_px', 'group_y_px', 'waypoint_x_mm', 'waypoint_y_mm',
    'annotation_snapshot_json',
]


class RawCommandRecording:
    """One video encoder and CSV writer on a bounded background queue.

    Video FPS caps frame submission; every submitted observation is logged, even
    between video frames. CSV recording_frame indexes the exact encoded frame,
    or is blank for a CSV-only observation. Queue overflow fails explicitly.
    """
    def __init__(self, base_path, size, fps=5., *, clock=time.monotonic, metadata=None, queue_size=32):
        if not np.isfinite(fps) or fps <= 0:
            raise ValueError('Recording FPS must be positive and finite.')
        if queue_size <= 0:
            raise ValueError('Recording queue size must be positive.')
        self.size = self.output_size = tuple(size)
        self.encoded_size = tuple(v+v % 2 for v in size)
        self.crop = None  # Keep original camera pixels; crop only during offline export.
        self.fps, self.clock, self.started_at = float(fps), clock, clock()
        self.frame_count = self.observation_count = 0
        self.accepted_observations = self.rejected_observations = 0
        self.failure_reason = None
        self.last_video_sample = self.last_sample = None
        self.error, self.closed = None, False
        self.jobs = queue.Queue(maxsize=queue_size)
        self.stop_event = threading.Event()
        base = Path(base_path).with_suffix('')
        self.paths = [base.with_name(base.name+suffix) for suffix in
                      ('_camera.mp4', '_tracking.csv', '_recording.json')]
        if any(path.exists() for path in self.paths):
            raise ValueError('Recording files already exist. Choose a new recording name.')
        self.writer = self.csv_file = None
        try:
            self.writer = cv2.VideoWriter(str(self.paths[0]), cv2.VideoWriter_fourcc(*'mp4v'), self.fps, self.encoded_size)
            if not self.writer.isOpened():
                raise ValueError('Cannot create camera video. Check codec and destination.')
            self.csv_file = self.paths[1].open('x', newline='', encoding='utf-8')
            self.csv_writer = csv.DictWriter(self.csv_file, fieldnames=RAW_FIELDS)
            self.csv_writer.writeheader()
            info = dict(schema_version=1, started_utc=datetime.now(timezone.utc).isoformat(),
                        started_monotonic_s=self.started_at, video_fps=self.fps,
                        source_size_px=self.size, encoded_size_px=self.encoded_size,
                        coordinates='native camera pixels; physical positions in mm',
                        timing='recording_frame is zero-based; blank means CSV-only observation. '
                               'Use capture_monotonic_s/recording_time_s for real elapsed time; video is constant FPS.',
                        settings=metadata or {})
            with self.paths[2].open('x', encoding='utf-8') as stream:
                json.dump(info, stream, default=_json_default, indent=2)
        except Exception:
            if self.writer is not None:
                self.writer.release()
            if self.csv_file is not None:
                self.csv_file.close()
            raise
        self.thread = threading.Thread(target=self._run, daemon=True, name='gui005-recording')
        self.thread.start()

    def submit(self, frame, rows, snapshot, captured_at):
        if self.error:
            raise OSError(f'Recording worker failed: {self.error}')
        if self.closed:
            raise ValueError('Recording has already stopped.')
        if tuple(frame.shape[1::-1]) != self.size:
            raise ValueError('Camera size changed; recording stopped.')
        if not np.isfinite(captured_at):
            raise ValueError('Recording requires a finite capture timestamp.')
        if self.last_sample is not None and captured_at <= self.last_sample:
            return  # Never duplicate a camera observation.
        save_video = self.last_video_sample is None or captured_at-self.last_video_sample >= 1/self.fps-1e-9
        job = (frame.copy() if save_video else None, [dict(row) for row in rows] or [{}],
               copy.deepcopy(snapshot), float(captured_at))
        try:
            self.jobs.put_nowait(job)
        except queue.Full:
            self.rejected_observations += 1
            self.failure_reason = 'Recording queue is full; stopped to protect control. Reduce Video FPS or camera resolution.'
            raise OSError(self.failure_reason) from None
        self.accepted_observations += 1
        self.last_sample = captured_at
        if save_video:
            self.last_video_sample = captured_at

    def _run(self):
        try:
            while not self.stop_event.is_set() or not self.jobs.empty():
                try:
                    frame, rows, snapshot, stamp = self.jobs.get(timeout=.05)
                except queue.Empty:
                    continue
                frame_index = ''
                if frame is not None:
                    padded = cv2.copyMakeBorder(frame, 0, self.encoded_size[1]-self.size[1],
                                                0, self.encoded_size[0]-self.size[0], cv2.BORDER_CONSTANT)
                    self.writer.write(padded)
                    frame_index = self.frame_count
                    self.frame_count += 1
                snapshot_json = json.dumps(snapshot, default=_json_default, separators=(',', ':'))
                detections = {str(item['robot_id']): item for item in snapshot.get('detections', [])}
                geometry = snapshot.get('geometry') or {}
                center = geometry.get('center_px', ['', ''])
                waypoint = snapshot.get('waypoint_mm')
                for index, row in enumerate(rows):
                    bbox = detections.get(str(row.get('robot_id')), {}).get('bbox', ['', '', '', ''])
                    row.update(recording_frame=frame_index, recording_time_s=stamp-self.started_at,
                               capture_monotonic_s=stamp, annotation_snapshot_json=snapshot_json if index == 0 else '',
                               group_x_px=center[0], group_y_px=center[1],
                               robot_bbox_x_px=bbox[0], robot_bbox_y_px=bbox[1],
                               robot_bbox_width_px=bbox[2], robot_bbox_height_px=bbox[3],
                               waypoint_x_mm='' if waypoint is None else waypoint[0],
                               waypoint_y_mm='' if waypoint is None else waypoint[1])
                    self.csv_writer.writerow(row)
                self.observation_count += 1
                # Buffered writes avoid flushing to disk once per measurement.
                if self.observation_count % 30 == 0:
                    self.csv_file.flush()
        except Exception as error:
            self.error = str(error)
        finally:
            self.writer.release()
            self.csv_file.close()

    def close(self):
        self.closed = True
        self.stop_event.set()
        self.thread.join()
        info = json.loads(self.paths[2].read_text(encoding='utf-8'))
        info.update(finished_utc=datetime.now(timezone.utc).isoformat(),
                    encoded_frames=self.frame_count, logged_observations=self.observation_count,
                    accepted_observations=self.accepted_observations, rejected_observations=self.rejected_observations,
                    error=self.error or self.failure_reason)
        self.paths[2].write_text(json.dumps(info, indent=2), encoding='utf-8')
        if self.error:
            raise OSError(f'Recording worker failed: {self.error}')


def render_saved_annotations(frame, snapshot, trails, center_trail, options=None):
    """Shared renderer for the Export tab preview and exported video."""
    saved = dict(snapshot.get('options', {}))
    saved.update(options or {})
    saved.pop('show_time', None)
    saved.pop('time_position', None)
    saved.update(desired_path_dashed=True, desired_point_radius=5, group_center_size=24,
                 annotation_thickness=2, group_center_thickness=2)
    show_robots = saved.pop('show_robots', True)
    show_cargo = saved.pop('show_cargo', True)
    show_trails = saved.pop('show_trails', False)
    show_center_trail = saved.pop('show_center_trail', False)
    show_commanded = saved.pop('show_commanded', False)
    show_waypoint = saved.pop('show_waypoint', True)
    roi = snapshot.get('roi')
    tracker = None if roi is None else SimpleNamespace(center=np.asarray(roi[:2]), radius=roi[2])
    pixels = snapshot.get('pixels')
    draw_command_overlay(frame, snapshot.get('setup'), None if pixels is None else np.asarray(pixels),
        snapshot.get('active_index'), snapshot.get('detections', []) if show_robots else [], tracker,
        geometry=snapshot.get('geometry'), cargo=snapshot.get('cargo') if show_cargo else None,
        trails=trails if show_trails else None, center_trail=center_trail if show_center_trail else None,
        commanded_pixel=snapshot.get('commanded_pixel') if show_commanded else None, **saved)
    waypoint = snapshot.get('waypoint_pixel')
    if show_waypoint and waypoint is not None:
        cv2.circle(frame, tuple(np.rint(waypoint).astype(int)), 6, (221, 136, 255), 2)
    return frame


def render_time_annotation(frame, elapsed, options=None):
    """Draw elapsed capture time after cropping; return its draggable bounds.

    Position is the label's top-left in normalized output-image coordinates.
    Clamp the whole label inside the image, including on small exports.
    """
    options = options or {}
    if not options.get('show_time', False):
        return None
    height, width = frame.shape[:2]
    text = f'Time : {elapsed:04.1f} s'
    font = cv2.FONT_HERSHEY_SIMPLEX
    padding = min(10, max(1, min(width, height)//20))
    (unit_width, unit_height), unit_baseline = cv2.getTextSize(text, font, 1., 1)
    scale = max(.01, min(3., (width-2*padding-6)/unit_width,
                             (height-2*padding-6)/(unit_height+unit_baseline)))
    thickness = max(1, round(scale*1.5))
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    box_width, box_height = min(width, tw+2*padding), min(height, th+baseline+2*padding)
    px, py = options.get('time_position', (.03, .03))
    x = max(0, min(width-box_width, round(float(px)*width)))
    y = max(0, min(height-box_height, round(float(py)*height)))
    cv2.rectangle(frame, (x, y), (x+box_width-1, y+box_height-1), (0, 0, 0), -1)
    cv2.putText(frame, text, (x+padding, y+padding+th), font, scale,
                (255, 255, 255), thickness, cv2.LINE_AA)
    return x, y, box_width, box_height


class RecordingTrailState:
    def __init__(self):
        self.trails, self.center = {}, deque(maxlen=300)
        self.last_stamp = self.last_epoch = None

    def observe(self, snapshot, stamp):
        epoch = snapshot.get('tracking_epoch')
        if self.last_epoch is not None and epoch != self.last_epoch:
            self.trails.clear()
            self.center.clear()
        self.last_epoch = epoch
        detections = snapshot.get('detections', [])
        geometry = snapshot.get('geometry')
        gap = self.last_stamp is not None and stamp-self.last_stamp > 1.
        visible = {item['robot_id'] for item in detections}
        for robot_id, trail in self.trails.items():
            if gap or robot_id not in visible:
                trail.append(None)
        for item in detections:
            self.trails.setdefault(item['robot_id'], deque(maxlen=300)).append(item['center'])
        if geometry is None or gap:
            self.center.append(None)
        if geometry is not None:
            self.center.append(geometry['center_px'])
        self.last_stamp = stamp


class RecordingPreview:
    """Load CSV observations once; seek camera frames without running detection."""
    def __init__(self, video_path, *, cancelled=lambda: False):
        self.video_path = Path(video_path)
        if not self.video_path.stem.endswith('_camera'):
            raise ValueError('Select a *_camera.mp4 recording with its matching CSV and metadata.')
        base = self.video_path.with_name(self.video_path.stem[:-7])
        self.info = json.loads(base.with_name(base.name+'_recording.json').read_text(encoding='utf-8'))
        self.observations, self.frames = [], []
        with base.with_name(base.name+'_tracking.csv').open(newline='', encoding='utf-8') as stream:
            for row in csv.DictReader(stream):
                if cancelled():
                    raise ValueError('Loading cancelled.')
                if not row.get('annotation_snapshot_json'):
                    continue
                snapshot = json.loads(row['annotation_snapshot_json'])
                self.observations.append((snapshot, float(row['capture_monotonic_s'])))
                if row.get('recording_frame', '') != '':
                    if int(row['recording_frame']) != len(self.frames):
                        raise ValueError('CSV/video frame sequence does not match.')
                    self.frames.append(len(self.observations)-1)
        if not self.frames:
            raise ValueError('No saved frame snapshots found. This recording needs the new CSV format.')
        capture = cv2.VideoCapture(str(self.video_path))
        try:
            if not capture.isOpened():
                raise ValueError('Cannot open camera video.')
            if round(capture.get(cv2.CAP_PROP_FRAME_COUNT)) != len(self.frames):
                raise ValueError('Video frame count does not match its CSV.')
        finally:
            capture.release()

    def time_origin(self, start_frame=None):
        if start_frame is None:
            return float(self.info['started_monotonic_s'])
        if not isinstance(start_frame, int) or not 0 <= start_frame < len(self.frames):
            raise ValueError('Export start frame is outside the recording.')
        return self.observations[self.frames[start_frame]][1]

    def render(self, frame_index, *, options=None, apply_crop=True, crop_circle=None,
               start_frame=None, cancelled=lambda: False):
        frame_index = int(frame_index)
        if not 0 <= frame_index < len(self.frames):
            raise ValueError('Preview frame is outside the recording.')
        origin = self.time_origin(start_frame)
        context = RecordingTrailState()
        for snapshot, stamp in self.observations[:self.frames[frame_index]+1]:
            if cancelled():
                raise ValueError('Preview cancelled.')
            context.observe(snapshot, stamp)
        capture = cv2.VideoCapture(str(self.video_path))
        try:
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = capture.read()
            if not ok:
                raise ValueError('Cannot decode the selected video frame.')
        finally:
            capture.release()
        render_saved_annotations(frame, snapshot, context.trails, context.center, options)
        circle = crop_circle if crop_circle is not None else self.info.get('settings', {}).get('crop_circle')
        if apply_crop and circle is not None:
            frame = CircularVideoCrop(tuple(self.info['source_size_px']), circle).apply(frame)
        elapsed = stamp-origin
        render_time_annotation(frame, elapsed, options)
        return frame, elapsed


def annotate_recording(video_path, output_path, *, options=None, apply_crop=True, crop_circle=None,
                       start_frame=None, cancelled=lambda: False, progress=lambda count: None):
    """Render saved native-pixel snapshots without rerunning detection/control."""
    video_path, output_path = Path(video_path), Path(output_path)
    if not video_path.stem.endswith('_camera'):
        raise ValueError('Select a camera video saved by Camera video + CSV recording.')
    base = video_path.with_name(video_path.stem[:-7])
    csv_path, metadata_path = base.with_name(base.name+'_tracking.csv'), base.with_name(base.name+'_recording.json')
    if output_path.exists():
        raise ValueError('Output already exists. Choose a new filename.')
    if start_frame is not None and (not isinstance(start_frame, int) or start_frame < 0):
        raise ValueError('Export start frame must be a nonnegative integer.')
    info = json.loads(metadata_path.read_text(encoding='utf-8'))
    size, fps = tuple(info['encoded_size_px']), float(info['video_fps'])
    crop_circle = (crop_circle if crop_circle is not None else info.get('settings', {}).get('crop_circle')) if apply_crop else None
    crop = None if crop_circle is None else CircularVideoCrop(tuple(info['source_size_px']), crop_circle)
    out_size = size if crop is None else tuple(v+v % 2 for v in crop.size)
    capture = cv2.VideoCapture(str(video_path))
    writer = None
    count, source_count, context = 0, 0, RecordingTrailState()
    origin = float(info['started_monotonic_s']) if start_frame is None else None
    try:
        if not capture.isOpened():
            raise ValueError('Cannot open saved camera video.')
        writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*'mp4v'), fps, out_size)
        if not writer.isOpened():
            raise ValueError('Cannot create annotated video.')
        with csv_path.open(newline='', encoding='utf-8') as stream:
            for row in csv.DictReader(stream):
                if cancelled():
                    raise ValueError('Annotation export cancelled; partial output retained.')
                if not row.get('annotation_snapshot_json'):
                    continue
                snapshot = json.loads(row['annotation_snapshot_json'])
                stamp = float(row['capture_monotonic_s'])
                context.observe(snapshot, stamp)
                if row.get('recording_frame', '') == '':
                    continue
                if int(row['recording_frame']) != source_count:
                    raise ValueError('CSV/video frame sequence does not match.')
                ok, frame = capture.read()
                if not ok:
                    raise ValueError('Camera video ended before its CSV frame sequence.')
                source_index = source_count
                source_count += 1
                if start_frame is not None and source_index < start_frame:
                    continue
                if origin is None:
                    origin = stamp
                render_saved_annotations(frame, snapshot, context.trails, context.center, options)
                if crop is not None:
                    frame = crop.apply(frame)
                    frame = cv2.copyMakeBorder(frame, 0, out_size[1]-crop.size[1],
                                              0, out_size[0]-crop.size[0], cv2.BORDER_CONSTANT)
                render_time_annotation(frame, stamp-origin, options)
                writer.write(frame)
                count += 1
                progress(count)
        if count == 0:
            if start_frame is not None:
                raise ValueError('Export start frame is outside the recording; no frames exported.')
            raise ValueError('No saved video frames found in the recording CSV.')
        if capture.read()[0]:
            raise ValueError('Camera video has frames missing from its CSV.')
    finally:
        capture.release()
        if writer is not None:
            writer.release()
    return count
