import csv
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np


EXPERIMENTAL_DIR = Path(__file__).resolve().parents[1] / "experimental"
sys.path.insert(0, str(EXPERIMENTAL_DIR))

import gui004
from robot_vision import RobotDetector


class OfflineDetectionTests(unittest.TestCase):
    def test_calibration_filename_includes_video_name(self):
        original_input_dir = gui004.INPUT_DIR
        gui004.INPUT_DIR = Path("inputs")
        try:
            calibration_file = gui004.calibration_file_for("experiment_07.mp4")
        finally:
            gui004.INPUT_DIR = original_input_dir

        self.assertEqual(
            calibration_file,
            Path("inputs/experiment_07_camera_calibration.json"),
        )

    def test_calibration_click_snaps_to_nearby_green_dot_center(self):
        app = gui004.OfflineDetectionGUI.__new__(gui004.OfflineDetectionGUI)
        app.detector = RobotDetector()
        app.current_frame_bgr = np.full((120, 160, 3), 255, dtype=np.uint8)
        cv2.circle(app.current_frame_bgr, (70, 50), 7, (0, 180, 0), -1)
        app.calibration_search_radius_var = SimpleNamespace(get=lambda: "35")
        app.calibration_minimum_area_var = SimpleNamespace(get=lambda: "1")

        center, snapped, distance = app.snap_calibration_point((80, 56))

        self.assertTrue(snapped)
        self.assertEqual(center, (70, 50))
        self.assertAlmostEqual(distance, np.hypot(10, 6))

    def test_calibration_click_detects_a_tiny_green_dot(self):
        app = gui004.OfflineDetectionGUI.__new__(gui004.OfflineDetectionGUI)
        app.detector = RobotDetector()
        app.current_frame_bgr = np.full((40, 40, 3), 255, dtype=np.uint8)
        cv2.rectangle(app.current_frame_bgr, (19, 19), (21, 21), (0, 180, 0), -1)
        app.calibration_search_radius_var = SimpleNamespace(get=lambda: "10")
        app.calibration_minimum_area_var = SimpleNamespace(get=lambda: "1")

        center, snapped, _ = app.snap_calibration_point((24, 24))

        self.assertTrue(snapped)
        self.assertEqual(center, (20, 20))

    def test_manual_robot_click_snaps_to_nearby_color_blob_center(self):
        app = gui004.OfflineDetectionGUI.__new__(gui004.OfflineDetectionGUI)
        app.detector = RobotDetector()
        app.current_frame_bgr = np.full((120, 160, 3), 255, dtype=np.uint8)
        cv2.circle(app.current_frame_bgr, (80, 60), 9, (0, 0, 255), -1)
        app.robot_search_radius_var = SimpleNamespace(get=lambda: "35")
        app.robot_snap_minimum_area_var = SimpleNamespace(get=lambda: "1")
        app.detection_color_var = SimpleNamespace(get=lambda: "Red")

        center, snapped, distance = app.snap_robot_point((92, 66))

        self.assertTrue(snapped)
        self.assertEqual(center, (80, 60))
        self.assertAlmostEqual(distance, np.hypot(12, 6))

    def test_manual_robot_click_detects_a_tiny_dark_blob(self):
        app = gui004.OfflineDetectionGUI.__new__(gui004.OfflineDetectionGUI)
        app.detector = RobotDetector()
        app.current_frame_bgr = np.full((40, 40, 3), 255, dtype=np.uint8)
        cv2.rectangle(app.current_frame_bgr, (19, 19), (21, 21), (20, 20, 20), -1)
        app.robot_search_radius_var = SimpleNamespace(get=lambda: "10")
        app.robot_snap_minimum_area_var = SimpleNamespace(get=lambda: "1")
        app.detection_color_var = SimpleNamespace(get=lambda: "Dark")

        center, snapped, _ = app.snap_robot_point((24, 24))

        self.assertTrue(snapped)
        self.assertEqual(center, (20, 20))

    def test_automatic_detection_can_keep_a_tiny_dark_blob(self):
        detector = RobotDetector()
        frame = np.full((40, 40, 3), 255, dtype=np.uint8)
        cv2.rectangle(frame, (19, 19), (21, 21), (20, 20, 20), -1)

        _, detections = detector.process(
            frame,
            mode="Color blobs",
            color="Dark",
            minimum_area=1,
            morphology_kernel_size=1,
        )

        self.assertEqual(len(detections), 1)
        self.assertEqual(detections[0]["center"], (20, 20))

    def test_dark_blue_profile_separates_robot_from_blue_background(self):
        detector = RobotDetector()
        frame = np.full((40, 40, 3), (209, 173, 136), dtype=np.uint8)
        cv2.rectangle(frame, (9, 9), (10, 10), (127, 93, 58), -1)
        cv2.rectangle(frame, (19, 19), (20, 20), (158, 124, 89), -1)
        cv2.rectangle(frame, (29, 29), (30, 30), (129, 95, 60), -1)

        _, detections = detector.process(
            frame,
            mode="Color blobs",
            color="Dark Blue",
            minimum_area=1,
            morphology_kernel_size=1,
        )

        self.assertEqual(len(detections), 3)

    def test_learns_robot_profile_from_clicked_samples(self):
        app = gui004.OfflineDetectionGUI.__new__(gui004.OfflineDetectionGUI)
        app.detector = RobotDetector()
        app.current_frame_bgr = np.full(
            (40, 40, 3), (209, 173, 136), dtype=np.uint8
        )
        robot_color = (127, 93, 58)
        for center in ((10, 10), (20, 20), (30, 30)):
            cv2.rectangle(
                app.current_frame_bgr,
                (center[0] - 1, center[1] - 1),
                center,
                robot_color,
                -1,
            )
        app.robot_points = [(10, 10), (20, 20), (30, 30)]
        no_op_var = SimpleNamespace(set=lambda _value: None)
        app.minimum_area_var = no_op_var
        app.robot_snap_minimum_area_var = no_op_var
        app.detection_filter_size_var = no_op_var
        app.learned_robot_color_ranges = None

        app.update_learned_robot_color_profile()
        _, detections = app.detector.process(
            app.current_frame_bgr,
            mode="Color blobs",
            color="Learned from clicks",
            minimum_area=1,
            morphology_kernel_size=1,
            color_ranges=app.learned_robot_color_ranges,
        )

        self.assertEqual(len(detections), 3)

    def test_orders_clicked_calibration_corners(self):
        points = [(150, 110), (10, 100), (145, 5), (5, 10)]

        ordered = gui004.OfflineDetectionGUI.order_corner_points(points)

        self.assertEqual(ordered, [(5, 10), (145, 5), (150, 110), (10, 100)])

    def test_processes_video_and_writes_annotated_video_and_csv(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "sample.avi"
            source_writer = cv2.VideoWriter(
                str(source), cv2.VideoWriter_fourcc(*"MJPG"), 12.0, (160, 120)
            )
            self.assertTrue(source_writer.isOpened())
            for index in range(8):
                frame = np.full((120, 160, 3), 255, dtype=np.uint8)
                cv2.circle(frame, (30 + index * 10, 60), 9, (0, 180, 0), -1)
                source_writer.write(frame)
            source_writer.release()

            original_project_dir = gui004.PROJECT_DIR
            gui004.PROJECT_DIR = root
            try:
                app = gui004.OfflineDetectionGUI.__new__(gui004.OfflineDetectionGUI)
                app.detector = RobotDetector()
                events = []
                app.put_latest_preview = events.append
                app.process_video(
                    source,
                    "Color blobs",
                    "Green",
                    20.0,
                    1.0,
                    29.0,
                    10.0,
                    10.0,
                    [(0, 0), (159, 0), (159, 119), (0, 119)],
                    threading.Event(),
                )
            finally:
                gui004.PROJECT_DIR = original_project_dir

            output_dir = root / "outputs" / "offline_detection" / "sample"
            video_output = output_dir / "sample_annotated.mp4"
            csv_output = output_dir / "sample_detections.csv"

            self.assertEqual(events[-1]["kind"], "finished")
            self.assertEqual(events[-1]["frames"], 8)
            self.assertGreater(video_output.stat().st_size, 0)
            with csv_output.open(newline="", encoding="utf-8") as csv_file:
                rows = list(csv.DictReader(csv_file))
            self.assertEqual(len(rows), 8)
            self.assertTrue(all(row["kind"] == "color" for row in rows))
            self.assertTrue(all(row["h_cm"] == "1.0" for row in rows))
            self.assertTrue(all(row["camera_height_cm"] == "29.0" for row in rows))
            self.assertTrue(
                all(row["dot_rectangle_width_cm"] == "10.0" for row in rows)
            )
            self.assertTrue(
                all(row["dot_rectangle_height_cm"] == "10.0" for row in rows)
            )


if __name__ == "__main__":
    unittest.main()
