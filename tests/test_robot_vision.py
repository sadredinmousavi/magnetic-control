import sys
import unittest
from pathlib import Path

import cv2
import numpy as np


EXPERIMENTAL_DIR = Path(__file__).resolve().parents[1] / "experimental"
sys.path.insert(0, str(EXPERIMENTAL_DIR))

from robot_vision import RobotDetector


class RobotVisionTests(unittest.TestCase):
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
