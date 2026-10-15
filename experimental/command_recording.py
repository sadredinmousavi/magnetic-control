"""Stream native-resolution camera/overlay videos and timestamped tracking CSV."""

import csv
from pathlib import Path
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
        radius = np.ceil(np.linalg.norm(points - center, axis=1).max())
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


TRACKING_FIELDS = ('frame time_s robot_id robot_x_px robot_y_px robot_x_mm robot_y_mm '
                   'group_x_mm group_y_mm expected_robots detected_robots group_valid '
                   'active_point desired_x_mm desired_y_mm error_mm roi_x_px roi_y_px '
                   'roi_radius_px sequence_running arrival_ready arrival_max_error_mm '
                   'desired_ratio desired_angle_deg recording_frame recording_time_s '
                   'crop_center_x_px crop_center_y_px crop_radius_px '
                   'control_enabled control_state control_mode control_alpha control_kp adaptive_kp kp_adaptation control_limit_mm adaptive_limit limit_adaptation '
                   'robot_speed_mm_s settling_state settle_hold_elapsed_s settled_xy_mm settled '
                   'correction_x_mm correction_y_mm commanded_x_mm commanded_y_mm '
                   'proposed_xy_mm calculated_motor_angles_deg sent_motor_angles_deg motor_sent_time_s orientation_measured_deg').split()


def draw_command_overlay(image, setup, pixels, active_index, detections, tracker=None,
                         *, show_path=False, show_roi=False, trails=None, commanded_pixel=None):
    """Draw in source-image pixels, independently of GUI zoom or visibility."""
    if setup is not None and len(setup['corners_px']) == 4:
        cv2.polylines(image, [np.asarray(setup['corners_px'], np.int32)], True, (153, 85, 153), 1)
    if pixels is not None:
        points = np.rint(pixels).astype(np.int32)
        if show_path and len(points) > 1:
            cv2.polylines(image, [points], False, (136, 136, 51), 1)
        for x, y in points:
            outside = (setup is not None and cv2.pointPolygonTest(
                np.asarray(setup['corners_px'], np.float32), (float(x), float(y)), True) < -1e-3)
            cv2.circle(image, (int(x), int(y)), 2, (68, 68, 255) if outside else (204, 204, 0), 1)
        if active_index is not None and active_index < len(points):
            point = tuple(points[active_index])
            cv2.drawMarker(image, point, (0, 153, 255), cv2.MARKER_CROSS, 6, 1, cv2.LINE_AA)
            cv2.putText(image, f'P{active_index + 1}', (point[0] + 7, point[1] - 7),
                        cv2.FONT_HERSHEY_SIMPLEX, .32, (0, 153, 255), 1, cv2.LINE_AA)
    if commanded_pixel is not None:
        point = tuple(np.rint(commanded_pixel).astype(int))
        cv2.drawMarker(image, point, (221, 85, 255), cv2.MARKER_DIAMOND, 8, 1, cv2.LINE_AA)
        cv2.putText(image, 'C', (point[0]+6, point[1]+10), cv2.FONT_HERSHEY_SIMPLEX,
                    .32, (221, 85, 255), 1, cv2.LINE_AA)
    if show_roi and tracker is not None:
        cv2.circle(image, tuple(np.rint(tracker.center).astype(int)), round(tracker.radius), (68, 153, 68), 1)
    for trail in (trails or {}).values():
        previous = None
        for point in trail:
            current = None if point is None else tuple(np.rint(point).astype(int))
            if previous is not None and current is not None:
                cv2.line(image, previous, current, (68, 153, 68), 1)
            previous = current
    for item in detections:
        x, y = np.rint(item['center']).astype(int)
        cv2.drawMarker(image, (x, y), (102, 255, 0), cv2.MARKER_CROSS, 4, 1, cv2.LINE_AA)
        cv2.putText(image, f"R{item['robot_id']}", (x + 4, y + 8), cv2.FONT_HERSHEY_SIMPLEX, .32, (102, 255, 0), 1, cv2.LINE_AA)
    return image


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
