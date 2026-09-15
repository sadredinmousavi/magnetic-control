"""OpenCV robot detection helpers for the experimental camera viewer."""

import math

import cv2
import numpy as np


class RobotDetector:
    """Detect colored robot bodies and optional ArUco identity markers."""

    COLOR_RANGES = {
        "Red": [((0, 100, 70), (10, 255, 255)), ((170, 100, 70), (179, 255, 255))],
        "Green": [((35, 70, 50), (85, 255, 255))],
        "Blue": [((90, 70, 50), (135, 255, 255))],
        "Dark Blue": [((90, 100, 70), (135, 255, 175))],
        "Yellow": [((20, 100, 80), (35, 255, 255))],
        "Dark": [((0, 0, 0), (179, 255, 65))],
    }

    def __init__(self):
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        parameters = cv2.aruco.DetectorParameters()
        self.aruco_detector = cv2.aruco.ArucoDetector(dictionary, parameters)

    def process(
        self,
        frame_bgr,
        mode="Off",
        color="Red",
        minimum_area=150.0,
        morphology_kernel_size=5,
        color_ranges=None,
        draw_annotations=True,
    ):
        """Return an annotated BGR frame and a list of detected robots."""
        annotated = frame_bgr.copy()
        detections = []

        if mode in ("Color blobs", "Color + ArUco"):
            detections.extend(
                self.detect_color_blobs(
                    annotated,
                    color,
                    float(minimum_area),
                    morphology_kernel_size,
                    color_ranges,
                    draw_annotations,
                )
            )

        if mode in ("ArUco markers", "Color + ArUco"):
            detections.extend(self.detect_aruco_markers(annotated))

        return annotated, detections

    def detect_color_blobs(
        self,
        frame_bgr,
        color,
        minimum_area,
        morphology_kernel_size=5,
        color_ranges=None,
        draw_annotations=True,
    ):
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        ranges = color_ranges or self.COLOR_RANGES.get(
            color, self.COLOR_RANGES["Red"]
        )
        for lower, upper in ranges:
            mask = cv2.bitwise_or(
                mask,
                cv2.inRange(
                    hsv,
                    np.array(lower, dtype=np.uint8),
                    np.array(upper, dtype=np.uint8),
                ),
            )

        kernel_size = max(1, int(morphology_kernel_size))
        if kernel_size > 1:
            kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        detections = []
        component_count, _, stats, centroids = cv2.connectedComponentsWithStats(mask)
        components = sorted(
            (
                (stats[index], centroids[index])
                for index in range(1, component_count)
                if stats[index, cv2.CC_STAT_AREA] >= minimum_area
            ),
            key=lambda component: component[0][cv2.CC_STAT_AREA],
            reverse=True,
        )
        for index, (stats_row, centroid) in enumerate(components, start=1):
            x = int(stats_row[cv2.CC_STAT_LEFT])
            y = int(stats_row[cv2.CC_STAT_TOP])
            width = int(stats_row[cv2.CC_STAT_WIDTH])
            height = int(stats_row[cv2.CC_STAT_HEIGHT])
            area = float(stats_row[cv2.CC_STAT_AREA])
            center_x = int(round(float(centroid[0])))
            center_y = int(round(float(centroid[1])))

            if draw_annotations:
                cv2.rectangle(
                    frame_bgr, (x, y), (x + width, y + height), (0, 255, 0), 2
                )
                cv2.drawMarker(
                    frame_bgr,
                    (center_x, center_y),
                    (0, 255, 0),
                    cv2.MARKER_CROSS,
                    18,
                    2,
                )
                cv2.putText(
                    frame_bgr,
                    f"{color} {index}: ({center_x}, {center_y})",
                    (x, max(y - 8, 18)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )
            detections.append(
                {
                    "kind": "color",
                    "label": f"{color} {index}",
                    "center": (center_x, center_y),
                    "area": area,
                }
            )

        return detections

    def detect_aruco_markers(self, frame_bgr):
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self.aruco_detector.detectMarkers(gray)
        if ids is None:
            return []

        cv2.aruco.drawDetectedMarkers(frame_bgr, corners, ids)
        detections = []
        for marker_corners, marker_id in zip(corners, ids.flatten()):
            points = marker_corners.reshape(4, 2)
            center = points.mean(axis=0)
            center_x, center_y = int(center[0]), int(center[1])
            edge = points[1] - points[0]
            angle = math.degrees(math.atan2(float(edge[1]), float(edge[0])))
            cv2.drawMarker(
                frame_bgr,
                (center_x, center_y),
                (255, 0, 255),
                cv2.MARKER_CROSS,
                18,
                2,
            )
            cv2.putText(
                frame_bgr,
                f"ID {marker_id}: ({center_x}, {center_y}) {angle:.1f} deg",
                (center_x + 8, center_y - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 0, 255),
                2,
                cv2.LINE_AA,
            )
            detections.append(
                {
                    "kind": "aruco",
                    "label": f"ID {int(marker_id)}",
                    "center": (center_x, center_y),
                    "angle_deg": angle,
                }
            )

        return detections

    @staticmethod
    def summarize(detections):
        if not detections:
            return "Detected robots: 0"
        details = ", ".join(
            f"{item['label']} @ {item['center']}" for item in detections[:8]
        )
        suffix = " ..." if len(detections) > 8 else ""
        return f"Detected robots: {len(detections)} — {details}{suffix}"
