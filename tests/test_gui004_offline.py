import csv
import json
import sys
import tempfile
import threading
import queue
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
    def test_annotation_bgra_makes_black_transparent_and_preserves_antialias_color(self):
        annotation = np.zeros((2, 2, 3), dtype=np.uint8)
        annotation[0, 0] = (0, 128, 0)
        annotation[1, 1] = (0, 0, 255)

        overlay = gui004.annotation_bgra(annotation)

        np.testing.assert_array_equal(overlay[0, 1], [0, 0, 0, 0])
        np.testing.assert_array_equal(overlay[0, 0], [0, 255, 0, 128])
        np.testing.assert_array_equal(overlay[1, 1], [0, 0, 255, 255])

    def test_chroma_composite_uses_pure_blue_only_where_overlay_is_empty(self):
        annotation = np.zeros((2, 2, 3), dtype=np.uint8)
        annotation[0, 0] = (0, 255, 0)

        chroma = gui004.annotation_on_chroma_color(annotation)

        np.testing.assert_array_equal(chroma[1, 1], [255, 0, 0])
        np.testing.assert_array_equal(chroma[0, 0], [0, 255, 0])

    def test_transparent_png_sequence_matches_source_timing_and_has_alpha(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.avi"
            destination = Path(directory) / "overlay_frames"
            writer = cv2.VideoWriter(
                str(source), cv2.VideoWriter_fourcc(*"MJPG"), 12.0, (80, 60)
            )
            self.assertTrue(writer.isOpened())
            for _ in range(3):
                writer.write(np.full((60, 80, 3), 180, dtype=np.uint8))
            writer.release()
            updates = queue.Queue()
            detections = {
                frame: [{"center": (20 + frame, 30), "area": 10}]
                for frame in range(3)
            }

            gui004.export_annotation_png_sequence(
                source, destination, detections, [], [], True, False, False,
                "cross", 1, {}, [], False, False, threading.Event(), updates,
            )

            result = updates.get_nowait()
            frames = sorted(destination.glob("annotations_*.png"))
            overlay = cv2.imread(str(frames[0]), cv2.IMREAD_UNCHANGED)
            metadata = json.loads(
                (destination / "sequence_info.json").read_text(encoding="utf-8")
            )

        self.assertEqual(result[0], "done")
        self.assertEqual(len(frames), 3)
        self.assertEqual(overlay.shape, (60, 80, 4))
        self.assertEqual(int(overlay[0, 0, 3]), 0)
        self.assertGreater(int(overlay[30, 20, 3]), 0)
        self.assertAlmostEqual(metadata["fps"], 12.0)
        self.assertEqual(metadata["frame_count"], 3)

    def test_red_cargo_selection_supports_shape_and_temporal_continuity(self):
        first = [
            {"center": (20, 20), "area": 100, "bbox": (10, 10, 20, 20)},
            {"center": (80, 80), "area": 500, "bbox": (60, 60, 40, 40)},
        ]
        cargo = gui004.select_cargo_candidate(first)
        self.assertEqual(cargo["center"], (80, 80))

        next_frame = [
            {"center": (84, 82), "area": 450, "bbox": (64, 62, 40, 40)},
            {"center": (20, 20), "area": 900, "bbox": (5, 5, 30, 30)},
        ]
        cargo = gui004.select_cargo_candidate(next_frame, (80, 80), 15)
        self.assertEqual(cargo["center"], (84, 82))
        self.assertIsNone(
            gui004.select_cargo_candidate(next_frame, (150, 150), 10)
        )

    def test_cargo_csv_reader_preserves_center_area_and_box(self):
        with tempfile.TemporaryDirectory() as directory:
            csv_path = Path(directory) / "sample_cargo.csv"
            with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=[
                    "frame", "detected", "center_x_px", "center_y_px",
                    "area_px2", "bbox_x", "bbox_y", "bbox_w", "bbox_h",
                ])
                writer.writeheader()
                writer.writerow({
                    "frame": 2, "detected": 1, "center_x_px": 40,
                    "center_y_px": 50, "area_px2": 600, "bbox_x": 20,
                    "bbox_y": 25, "bbox_w": 40, "bbox_h": 50,
                })
                writer.writerow({"frame": 3, "detected": 0})

            cargo = gui004.read_cargo_csv(csv_path)

        self.assertEqual(cargo[2]["center"], (40, 50))
        self.assertEqual(cargo[2]["area"], 600)
        self.assertEqual(cargo[2]["bbox"], (20, 25, 40, 50))
        self.assertNotIn(3, cargo)

    def test_fixed_calibration_markers_are_excluded_but_index_is_preserved(self):
        candidates = [
            {"center": (11, 10), "area": 5},
            {"center": (50, 50), "area": 6},
            {"center": (90, 90), "area": 7},
        ]

        eligible, excluded = gui004.exclude_calibration_markers(
            candidates, [(10, 10), (90, 90)], 3
        )

        self.assertEqual(excluded, {0, 2})
        self.assertEqual(len(eligible), 1)
        self.assertEqual(eligible[0]["center"], (50, 50))
        self.assertEqual(eligible[0]["candidate_index"], 1)

    def test_candidate_matching_selects_only_nearby_named_robots(self):
        candidates = [
            {"center": (11, 10), "area": 5},
            {"center": (49, 20), "area": 6},
            {"center": (90, 69), "area": 7},
            {"center": (30, 35), "area": 500},
            {"center": (100, 10), "area": 600},
        ]

        selected = gui004.match_robot_candidates(
            candidates, [(10, 10), (50, 20), (90, 70)], 8
        )

        self.assertEqual([item["robot_id"] for item in selected], [1, 2, 3])
        self.assertEqual(
            [item["center"] for item in selected],
            [(11, 10), (49, 20), (90, 69)],
        )
        self.assertEqual(
            [item["candidate_index"] for item in selected], [0, 1, 2]
        )

    def test_candidate_matching_uses_global_assignment_not_greedy_pairs(self):
        candidates = [{"center": (3, 0)}, {"center": (6, 0)}]

        selected = gui004.match_robot_candidates(
            candidates, [(0, 0), (4, 0)], 5
        )

        self.assertEqual(
            [(item["robot_id"], item["center"]) for item in selected],
            [(1, (3, 0)), (2, (6, 0))],
        )

    def test_csv_reader_ignores_rejected_color_candidates(self):
        with tempfile.TemporaryDirectory() as directory:
            csv_path = Path(directory) / "detections.csv"
            with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=[
                    "frame", "center_x_px", "center_y_px", "area_px2",
                    "accepted", "robot_id",
                ])
                writer.writeheader()
                writer.writerow({
                    "frame": 0, "center_x_px": 10, "center_y_px": 20,
                    "area_px2": 5, "accepted": 1, "robot_id": 1,
                })
                writer.writerow({
                    "frame": 0, "center_x_px": 70, "center_y_px": 80,
                    "area_px2": 50, "accepted": 0, "robot_id": "",
                })

            detections = gui004.read_detections_csv(csv_path)

        self.assertEqual(len(detections[0]), 1)
        self.assertEqual(detections[0][0]["center"], (10, 20))
        self.assertEqual(detections[0][0]["robot_id"], 1)

    def test_detection_circle_excludes_outside_blobs_without_artificial_dark_blob(self):
        frame = np.full((100, 140, 3), 240, dtype=np.uint8)
        cv2.circle(frame, (45, 50), 4, (20, 20, 20), -1)
        cv2.circle(frame, (110, 50), 6, (20, 20, 20), -1)
        mask = gui004.circle_detection_mask(frame.shape[:2], (50, 50, 30))
        _, detections = RobotDetector().process(
            frame, mode="Color blobs", color="Dark", minimum_area=1,
            morphology_kernel_size=3, detection_mask=mask)
        self.assertEqual(len(detections), 1)
        self.assertEqual(detections[0]["center"], (45, 50))

    def test_centroid_ignores_area_changes_and_keeps_fractional_position(self):
        detections = [{"center": (10, 20), "area": 1},
                      {"center": (31, 41), "area": 100}]
        self.assertEqual(gui004.robot_center_for(detections, 2), (20.5, 30.5))
        detections[0]["area"], detections[1]["area"] = 1000, 1
        self.assertEqual(gui004.robot_center_for(detections, 2), (20.5, 30.5))
        self.assertIsNone(gui004.robot_center_for(detections, 3))
        self.assertIsNone(gui004.robot_center_for(detections, 1))
        self.assertIsNone(gui004.robot_center_for([], 2))

    def test_invalid_count_hides_center_and_breaks_track(self):
        frame = np.full((80, 100, 3), 240, dtype=np.uint8)
        detections = [{"center": (20, 30)}, {"center": (60, 30)}]
        rendered = gui004.draw_detection_overlays(frame, detections, expected_count=3)
        np.testing.assert_array_equal(rendered[30, 40], frame[30, 40])
        track = [(0, (10.5, 30)), (1, None), (2, (70.5, 30))]
        np.testing.assert_array_equal(
            gui004.draw_dashed_center_track(frame, track, 2), frame)

    def test_circle_crop_preserves_inside_and_masks_outside(self):
        frame = np.full((60, 80, 3), 200, dtype=np.uint8)
        frame[30, 40] = (20, 40, 60)
        cropped = gui004.circular_video_frame(frame, (40, 30), 20)
        self.assertEqual(cropped.shape, (40, 40, 3))
        np.testing.assert_array_equal(cropped[20, 20], frame[30, 40])
        np.testing.assert_array_equal(cropped[0, 0], [0, 0, 0])
        preview = gui004.circular_video_frame(frame, (40, 30), 20, preview=True)
        self.assertEqual(preview.shape, frame.shape)
        np.testing.assert_array_equal(preview[30, 40], frame[30, 40])
        np.testing.assert_array_equal(preview[0, 0], [50, 50, 50])
        np.testing.assert_array_equal(frame[0, 0], [200, 200, 200])

    def test_circle_export_frames_and_cancel_preserve_original(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.mp4"
            destination = Path(directory) / "crop.mp4"
            writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"mp4v"),
                                     12, (80, 60))
            self.assertTrue(writer.isOpened())
            for _ in range(4):
                writer.write(np.full((60, 80, 3), 200, dtype=np.uint8))
            writer.release()
            original = source.read_bytes()
            updates = queue.Queue()
            args = (source, destination, (40, 30), 20, {}, [], [],
                    False, False, False, "cross")
            gui004.export_circular_video(*args, threading.Event(), updates)
            self.assertEqual(updates.get_nowait()[0], "done")
            capture = cv2.VideoCapture(str(destination))
            try:
                self.assertEqual(int(capture.get(cv2.CAP_PROP_FRAME_COUNT)), 4)
                self.assertAlmostEqual(capture.get(cv2.CAP_PROP_FPS), 12)
                success, frame = capture.read()
                self.assertTrue(success)
                self.assertEqual(frame.shape, (40, 40, 3))
                self.assertLess(int(frame[0, 0].max()), 15)
                self.assertGreater(int(frame[20, 20].min()), 170)
            finally:
                capture.release()
            exported = destination.read_bytes()
            stop = threading.Event()
            stop.set()
            gui004.export_circular_video(*args, stop, updates)
            self.assertEqual(updates.get_nowait()[0], "cancelled")
            self.assertEqual(destination.read_bytes(), exported)
            self.assertEqual(source.read_bytes(), original)
            self.assertFalse(list(Path(directory).glob(".circle-export-*")))

    def test_red_center_track_is_dashed_and_stops_at_current_frame(self):
        frame = np.full((80, 100, 3), 240, dtype=np.uint8)
        track = [(0, (20, 30)), (1, (60, 30)), (3, (80, 30))]

        before_motion = gui004.draw_dashed_center_track(frame, track, 0)
        after_motion = gui004.draw_dashed_center_track(frame, track, 1)
        after_gap = gui004.draw_dashed_center_track(frame, track, 3)

        self.assertTrue(np.array_equal(before_motion, frame))
        self.assertGreater(int(after_motion[30, 22, 2]), 250)
        self.assertEqual(tuple(after_motion[30, 26]), (240, 240, 240))
        self.assertEqual(tuple(after_gap[30, 70]), (240, 240, 240))

    def test_post_overlays_use_original_frame_and_any_robot_count(self):
        frame = np.full((120, 160, 3), 240, dtype=np.uint8)
        detections = [
            {"center": (20, 30), "area": 1},
            {"center": (40, 30), "area": 1},
            {"center": (60, 30), "area": 1},
            {"center": (100, 70), "area": 3},
        ]
        corners = [(5, 5), (155, 5), (155, 115), (5, 115)]
        overlaid = gui004.draw_detection_overlays(frame, detections, corners)

        self.assertEqual(tuple(overlaid[20, 15]), (240, 240, 240))
        self.assertEqual(tuple(overlaid[30, 20]), (0, 255, 0))
        self.assertEqual(tuple(overlaid[40, 55]), (0, 0, 255))
        self.assertGreater(int(overlaid[5, 80, 1]), 240)
        self.assertLess(int(overlaid[5, 80, 0]), 50)
        self.assertTrue(np.array_equal(
            gui004.draw_detection_overlays(frame, detections, corners, False, False),
            frame,
        ))

    def test_hollow_rectangle_marker_leaves_robot_center_visible(self):
        frame = np.full((80, 100, 3), 240, dtype=np.uint8)
        detections = [
            {"center": (20, 30), "area": 100},
            {"center": (60, 30), "area": 100},
        ]

        overlaid = gui004.draw_detection_overlays(
            frame,
            detections,
            show_lines=False,
            robot_marker_style="rectangle",
        )

        self.assertEqual(tuple(overlaid[30, 20]), (240, 240, 240))
        self.assertEqual(tuple(overlaid[22, 20]), (0, 255, 0))
        self.assertEqual(tuple(overlaid[30, 40]), (0, 0, 255))

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
                    # The robot crosses outside this calibration rectangle.
                    [(45, 40), (75, 40), (75, 80), (45, 80)],
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
            detections_by_frame = gui004.read_detections_csv(csv_output)
            self.assertEqual(len(detections_by_frame), 8)
            self.assertEqual(len(detections_by_frame[0]), 1)
            self.assertLess(detections_by_frame[0][0]["center"][0], 45)
            self.assertGreater(detections_by_frame[7][0]["center"][0], 75)

    def test_process_video_writes_separate_red_cargo_track(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "cargo_sample.avi"
            writer = cv2.VideoWriter(
                str(source), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (120, 90)
            )
            self.assertTrue(writer.isOpened())
            for index in range(4):
                frame = np.full((90, 120, 3), 255, dtype=np.uint8)
                cv2.circle(frame, (15, 15), 4, (0, 180, 0), -1)
                cv2.rectangle(
                    frame, (40 + index * 4, 35), (70 + index * 4, 60),
                    (0, 0, 220), -1,
                )
                writer.write(frame)
            writer.release()

            original_project_dir = gui004.PROJECT_DIR
            gui004.PROJECT_DIR = root
            try:
                app = gui004.OfflineDetectionGUI.__new__(gui004.OfflineDetectionGUI)
                app.detector = RobotDetector()
                events = []
                app.put_latest_preview = events.append
                app.process_video(
                    source, "Color blobs", "Green", 5, 1, 29, 10, 10,
                    [(0, 0), (119, 0), (119, 89), (0, 89)],
                    threading.Event(), cargo_enabled=True,
                    cargo_minimum_area=100, cargo_tracking_radius=20,
                )
            finally:
                gui004.PROJECT_DIR = original_project_dir

            cargo_path = (
                root / "outputs" / "offline_detection" / "cargo_sample"
                / "cargo_sample_cargo.csv"
            )
            cargo = gui004.read_cargo_csv(cargo_path)

        self.assertEqual(events[-1]["kind"], "finished")
        self.assertEqual(len(cargo), 4)
        self.assertLess(cargo[0]["center"][0], cargo[3]["center"][0])


if __name__ == "__main__":
    unittest.main()
