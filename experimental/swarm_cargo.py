"""Metric swarm geometry and cargo contrast tracking for GUI005."""

import cv2
import numpy as np

from robot_vision import RobotDetector
from gui004 import select_cargo_candidate
from position_control import PositionFeedback


class CargoPositionFeedback(PositionFeedback):
    """Move the equilibrium point using cargo error, without a cargo magnetic model."""

    def __init__(self, *args, preview_only=False, observation_valid=lambda: True, **kwargs):
        super().__init__(*args, **kwargs)
        self.preview_only = bool(preview_only)
        self.initial_measurement_valid = observation_valid

    def set_enabled(self, enabled):
        if enabled and not self.preview_only:
            self.reset_history()
            self.enabled.set()
        else:
            self.enabled.clear()


class CargoTracker:
    """One dark-contrast cargo in a user-selected ROI, independent of its color."""
    def __init__(self):
        self.detector = RobotDetector()
        self.center = None
        self.detection = None
        self.error = ''

    def process(self, frame, minimum_area=200., radius=100., mask=None, selection=None,
                bounds=None, roi=None, contrast_threshold=10.):
        if not np.isfinite([minimum_area, radius]).all() or minimum_area <= 0 or radius <= 0:
            raise ValueError('Cargo minimum area and tracking radius must be positive and finite.')
        height, width = frame.shape[:2]
        search_mask = np.full((height, width), 255, np.uint8) if mask is None else mask.copy()
        if roi is not None:
            cx, cy, rr = np.asarray(roi, float)
            if not np.isfinite([cx, cy, rr]).all() or rr <= 0:
                raise ValueError('Place a valid cargo ROI first.')
            roi_mask = np.zeros((height, width), np.uint8)
            cv2.circle(roi_mask, tuple(np.rint([cx, cy]).astype(int)), round(rr), 255, -1)
            search_mask = cv2.bitwise_and(search_mask, roi_mask)
            # Preserve real-image context for the same local contrast estimator.
            bounds = (int(cx-rr)-32, int(cy-rr)-32, int(cx+rr)+33, int(cy+rr)+33)
        x0, y0, x1, y1 = (0, 0, width, height) if bounds is None else bounds
        x0, y0, x1, y1 = max(0, x0), max(0, y0), min(width, x1), min(height, y1)
        self.detector.last_blob_mask = np.zeros((height, width), np.uint8)
        if x1 <= x0 or y1 <= y0:
            self.detection, self.error = None, 'Cargo ROI is outside the camera image.'
            return None
        binary = self.detector.dark_contrast_mask(frame[y0:y1, x0:x1], contrast_threshold)
        binary = cv2.bitwise_and(binary, search_mask[y0:y1, x0:x1])
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < minimum_area:
                continue
            moments = cv2.moments(contour)
            x, y, w, h = cv2.boundingRect(contour)
            candidates.append(dict(center=(round(moments['m10']/moments['m00'])+x0,
                                           round(moments['m01']/moments['m00'])+y0),
                                   bbox=(x+x0, y+y0, w, h), area=float(area)))
            # Local contrast outlines large solid objects. Fill their silhouette
            # so shape fitting and centroid use the body rather than just its rim.
            cv2.drawContours(self.detector.last_blob_mask, [contour+[x0, y0]], -1, 255, -1)
        self.detector.last_blob_mask = cv2.bitwise_and(self.detector.last_blob_mask, search_mask)
        previous = self.center if selection is None else selection
        if previous is None and len(candidates) > 1:
            self.detection, self.error = None, 'Multiple cargo objects: place the ROI around just the cargo.'
            return None
        self.detection = select_cargo_candidate(candidates, previous, radius)
        self.error = '' if self.detection is not None else 'Cargo not detected in ROI; adjust ROI, Contrast or Cargo min area. Corrections paused.'
        if self.detection is not None:
            self.center = self.detection['center']
        return self.detection


# Backwards compatibility for callers of the earlier GUI005 tracker name.
RedCargoTracker = CargoTracker


def cargo_geometry(cargo, red_mask, axes, shape='Circle'):
    """Fit a rectangle after perspective correction; its long axis is axial (0–180°)."""
    if shape not in ('Circle', 'Rectangle'):
        raise ValueError('Select Circle or Rectangle for the cargo type.')
    result = dict(cargo, shape=shape, angle_deg=None, direction_px=None,
                  outline_px=None, length_mm=None, width_mm=None)
    if shape == 'Circle' or axes is None or red_mask is None:
        return result
    x, y, width, height = cargo['bbox']
    contours, _ = cv2.findContours(red_mask[y:y+height, x:x+width], cv2.RETR_EXTERNAL,
                                  cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return result
    contour = max(contours, key=cv2.contourArea).reshape(-1, 2) + [x, y]
    if len(contour) < 3:
        return result
    metric = axes.coordinates(contour).astype(np.float32)
    rectangle = cv2.minAreaRect(metric)
    box = cv2.boxPoints(rectangle)
    edges = np.roll(box, -1, axis=0) - box
    lengths = np.linalg.norm(edges, axis=1)
    long_index = int(np.argmax(lengths))
    length, short = float(lengths[long_index]), float(np.min(lengths))
    if short < 1e-6:
        return result
    center = np.asarray(rectangle[0])
    result.update(xy_cm=tuple(center), center=tuple(axes.pixel_positions([center])[0]),
                  outline_px=axes.pixel_positions(box), length_mm=length*10, width_mm=short*10)
    # A nearly square object has no reliably distinguishable long direction.
    if length / short < 1.1:
        return result
    vector = edges[long_index] / length
    angle = float(np.degrees(np.arctan2(vector[1], vector[0])) % 180)
    direction = np.array([np.cos(np.radians(angle)), np.sin(np.radians(angle))])
    result.update(angle_deg=angle, direction_px=axes.pixel_positions(
        center + np.array([-.5, .5])[:, None] * length * direction))
    return result


def cargo_label(cargo):
    text = 'Cargo ' + cargo.get('shape', 'Circle').lower()
    xy = cargo.get('xy_cm')
    if xy is not None:
        text += f' ({xy[0]*10:.2f}, {xy[1]*10:.2f}) mm'
    if cargo.get('shape') == 'Rectangle':
        angle = cargo.get('angle_deg')
        text += '; direction n/a' if angle is None else f'; axis {angle:.1f} deg'
    return text


def swarm_geometry(detections, expected_count, axes):
    """PCA spread axes in physical coordinates, unrelated to magnetic stiffness."""
    if len(detections) != expected_count or not detections or axes is None:
        return None
    xy = axes.coordinates([item['center'] for item in detections]) * 10  # mm
    center = xy.mean(axis=0)
    result = dict(center_mm=center, center_px=axes.pixel_positions([center / 10])[0],
                  ratio=None, angle_deg=None, strong_mm=None, weak_mm=None, lines=[])
    if len(xy) < 2:
        return result
    covariance = (xy - center).T @ (xy - center) / len(xy)
    values, vectors = np.linalg.eigh(covariance)
    weak, strong = np.sqrt(np.maximum(values, 0))
    result.update(strong_mm=float(strong), weak_mm=float(weak))
    if strong < 1e-6:
        return result
    result['ratio'] = float(strong / weak) if weak > max(1e-6, strong * 1e-6) else float('inf')
    # At equal spread there is no unique orientation.
    if strong - weak <= strong * .01:
        return result
    result['angle_deg'] = float(np.degrees(np.arctan2(vectors[1, 1], vectors[0, 1])) % 180)
    for label, index, spread in (('Strong spread', 1, strong), ('Weak spread', 0, weak)):
        if spread > 1e-6:
            endpoints = (center + np.array([-1., 1.])[:, None] * vectors[:, index] * 2 * spread) / 10
            result['lines'].append((label, axes.pixel_positions(endpoints)))
    return result


def measurement_positions(target, detections, expected_count, cargo):
    """The sole feedback/arrival observation; never substitute robots for lost cargo."""
    if target == 'Cargo center':
        return None if cargo is None or cargo.get('xy_cm') is None else np.asarray([cargo['xy_cm']]) * .01
    if len(detections) != expected_count or not detections or any(d.get('xy_cm') is None for d in detections):
        return None
    return np.asarray([d['xy_cm'] for d in sorted(detections, key=lambda d: d['robot_id'])]) * .01


def draw_measurements(image, geometry=None, cargo=None, *, show_labels=True,
                      show_group_center=True, show_spread_axes=True, group_center_size=12,
                      group_center_thickness=1):
    def label(text, point, color):
        if show_labels:
            cv2.putText(image, text, tuple(np.rint(point).astype(int)), cv2.FONT_HERSHEY_SIMPLEX,
                    .4, color, 1, cv2.LINE_AA)
    if geometry is not None and show_group_center:
        center = tuple(np.rint(geometry['center_px']).astype(int))
        cv2.drawMarker(image, center, (255, 255, 255), cv2.MARKER_CROSS,
                       group_center_size, group_center_thickness)
        xy = geometry['center_mm']
        ratio = geometry['ratio']
        ratio_text = 'n/a' if ratio is None else ('infinite (line)' if np.isinf(ratio) else f'{ratio:.2f}')
        label(f'Group ({xy[0]:.2f}, {xy[1]:.2f}) mm; spread ratio {ratio_text}',
              np.asarray(center) + [8, -12], (255, 255, 255))
    if geometry is not None and show_spread_axes:
        for name, points in geometry['lines']:
            color = (0, 200, 255) if name.startswith('Strong') else (255, 180, 100)
            p1, p2 = np.rint(points).astype(int)
            cv2.line(image, tuple(p1), tuple(p2), color, 1, cv2.LINE_AA)
            label(name, p2 + [4, 4], color)
    if cargo is not None:
        x, y, w, h = cargo['bbox']
        color = (255, 255, 0)
        cv2.rectangle(image, (x, y), (x+w, y+h), color, 1)
        if cargo.get('outline_px') is not None:
            cv2.polylines(image, [np.rint(cargo['outline_px']).astype(np.int32)], True, color, 1, cv2.LINE_AA)
        if cargo.get('direction_px') is not None:
            p1, p2 = np.rint(cargo['direction_px']).astype(int)
            cv2.line(image, tuple(p1), tuple(p2), (255, 120, 255), 2, cv2.LINE_AA)
        center = tuple(np.rint(cargo['center']).astype(int))
        cv2.drawMarker(image, center, color, cv2.MARKER_CROSS, 14, 1)
        label(cargo_label(cargo), np.asarray(center) + [10, 15], color)
    return image
