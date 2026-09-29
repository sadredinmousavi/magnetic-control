import sys
import unittest
from pathlib import Path

import cv2
import numpy as np


EXPERIMENTAL_DIR = Path(__file__).resolve().parents[1] / "experimental"
sys.path.insert(0, str(EXPERIMENTAL_DIR))

from robot_vision import RobotDetector


class RobotVisionTests(unittest.TestCase):
    def test_dark_contrast_detects_desaturated_robot_on_bright_background(self):
        detector = RobotDetector()
        frame = np.full((100, 120, 3), (210, 190, 170), dtype=np.uint8)
        cv2.circle(frame, (40, 50), 2, (60, 30, 20), -1)
        cv2.circle(frame, (75, 55), 2, (95, 65, 55), -1)

        _, detections = detector.process(
            frame,
            mode="Color blobs",
            color="Dark contrast",
            minimum_area=1,
            morphology_kernel_size=1,
        )

        centers = [item["center"] for item in detections]
        self.assertTrue(any(np.linalg.norm(np.subtract(center, (40, 50))) <= 2
                            for center in centers))
        self.assertTrue(any(np.linalg.norm(np.subtract(center, (75, 55))) <= 2
                            for center in centers))

    def setUp(self):
        self.detector = RobotDetector()

    def test_detects_red_blob_center(self):
        frame = np.full((240, 320, 3), 255, dtype=np.uint8)
        cv2.circle(frame, (100, 120), 25, (0, 0, 255), -1)

        _, detections = self.detector.process(
            frame, mode="Color blobs", color="Red", minimum_area=150
        )

        self.assertEqual(len(detections), 1)
        self.assertEqual(detections[0]["kind"], "color")
        self.assertEqual(detections[0]["center"], (100, 120))

    def test_detects_aruco_id_and_center(self):
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        marker = cv2.aruco.generateImageMarker(dictionary, 0, 160)
        image = np.full((240, 240), 255, dtype=np.uint8)
        image[40:200, 40:200] = marker
        frame = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        _, detections = self.detector.process(frame, mode="ArUco markers")

        self.assertEqual(len(detections), 1)
        self.assertEqual(detections[0]["label"], "ID 0")
        self.assertAlmostEqual(detections[0]["center"][0], 120, delta=1)
        self.assertAlmostEqual(detections[0]["center"][1], 120, delta=1)


if __name__ == "__main__":
    unittest.main()
