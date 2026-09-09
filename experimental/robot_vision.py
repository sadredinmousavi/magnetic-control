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
        "Yellow": [((20, 100, 80), (35, 255, 255))],
        "Dark": [((0, 0, 0), (179, 255, 65))],
    }

    def __init__(self):
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        parameters = cv2.aruco.DetectorParameters()
        self.aruco_detector = cv2.aruco.ArucoDetector(dictionary, parameters)

    def process(self, frame_bgr, mode="Off", color="Red", minimum_area=150.0):
        """Return an annotated BGR frame and a list of detected robots."""
        annotated = frame_bgr.copy()
        detections = []

        if mode in ("Color blobs", "Color + ArUco"):
            detections.extend(
                self.detect_color_blobs(annotated, color, float(minimum_area))
            )

        if mode in ("ArUco markers", "Color + ArUco"):
            detections.extend(self.detect_aruco_markers(annotated))

        return annotated, detections

    def detect_color_blobs(self, frame_bgr, color, minimum_area):
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for lower, upper in self.COLOR_RANGES.get(color, self.COLOR_RANGES["Red"]):
            mask = cv2.bitwise_or(
                mask,
                cv2.inRange(
                    hsv,
                    np.array(lower, dtype=np.uint8),
                    np.array(upper, dtype=np.uint8),
                ),
            )

        kernel = np.ones((5, 5), dtype=np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        detections = []
        valid_contours = sorted(
            (
                contour
                for contour in contours
                if cv2.contourArea(contour) >= minimum_area
            ),
            key=cv2.contourArea,
            reverse=True,
        )
        for index, contour in enumerate(valid_contours, start=1):
            area = float(cv2.contourArea(contour))
            x, y, width, height = cv2.boundingRect(contour)
            moments = cv2.moments(contour)
            if moments["m00"]:
                center_x = int(moments["m10"] / moments["m00"])
                center_y = int(moments["m01"] / moments["m00"])
            else:
                center_x = x + width // 2
                center_y = y + height // 2

            label = f"{color} {index}: ({center_x}, {center_y})"
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
                label,
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
