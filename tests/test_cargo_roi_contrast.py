from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'experimental'))
from swarm_cargo import CargoTracker
from scene_worker import analyze_scene
from closed_loop_calibration import MovingROICalibration
from trajectory_tab import TrajectoryTab
from command_tab import CommandTab
from unittest.mock import Mock


class CargoROIContrastTests(unittest.TestCase):
    def test_cargo_roi_action_cannot_silently_switch_robot_mode(self):
        tab = CommandTab.__new__(CommandTab)
        tab.control_target_var = Mock(get=Mock(return_value='Robots'))
        tab.tracking_status = Mock()
        tab.begin_cargo_selection()
        tab.control_target_var.set.assert_not_called()
        self.assertIn('Robots mode is active', tab.tracking_status.config.call_args.kwargs['text'])

    def frame(self, color=(16, 18, 35)):
        image = np.full((240, 240, 3), (190, 180, 160), np.uint8)
        cv2.circle(image, (100, 120), 22, color, -1)
        cv2.circle(image, (200, 120), 27, (0, 0, 0), -1)
        return image

    def test_colors_do_not_matter_and_outside_roi_is_ignored(self):
        for color in ((16,18,35), (35,35,35), (55,30,15)):
            with self.subTest(color=color):
                tracker = CargoTracker()
                found = tracker.process(self.frame(color), roi=(100,120,45), contrast_threshold=10)
                np.testing.assert_allclose(found['center'], [100,120], atol=1)
                mask = tracker.detector.last_blob_mask
                self.assertEqual(mask[120,100], 255)  # Solid interior is retained.
                self.assertEqual(mask[120,200], 0)
                missing = self.frame(color)
                cv2.circle(missing, (100,120), 25, (190,180,160), -1)
                self.assertIsNone(tracker.process(missing, roi=(100,120,45)))
                self.assertEqual(tracker.center, found['center'])

    def job(self, target):
        return dict(frame=self.frame(), settings=dict(
            robot=dict(corners=[(10,10),(230,10),(230,230),(10,230)], width_cm=10, height_cm=10,
                circle=(100,120,45), count=1, minimum_area=1,
                axis_samples={'Origin':[120,120], '+X':[142,120], '+Y':[120,98]},
                contrast_threshold=10, allow_uncalibrated=True),
            references=[], workspace=None, cargo_center=None, cargo_enabled=True,
            cargo_area=200., cargo_radius=45., cargo_shape='Circle', target=target))

    def test_async_cargo_mode_never_processes_robots(self):
        with patch.object(MovingROICalibration, 'process', side_effect=AssertionError('robot detection ran')):
            result = analyze_scene(self.job('Cargo center'))
        self.assertEqual(result['detections'], [])
        self.assertIsNone(result['geometry'])
        self.assertIsNone(result['group_xy'])
        np.testing.assert_allclose(result['cargo']['center'], [100,120], atol=1)
        self.assertIsNotNone(result['cargo']['xy_cm'])

    def test_async_robot_mode_never_processes_cargo(self):
        with patch.object(CargoTracker, 'process', side_effect=AssertionError('cargo detection ran')):
            result = analyze_scene(self.job('Robots'))
        self.assertIsNone(result['cargo'])

    def test_trajectory_cargo_measurement_does_not_require_robot_group(self):
        tab = TrajectoryTab.__new__(TrajectoryTab)
        tab.control_target_var = type('Var', (), {'get': lambda self: 'Cargo center'})()
        tab.tracking_tracker = type('Tracker', (), {'count': 6})()
        tab.tracking_group_xy, tab.tracking_detections = None, []
        tab.cargo_detection = {'xy_cm': (1.5, -.5)}
        np.testing.assert_allclose(tab.control_positions(), [[.015, -.005]])
        tab.cargo_detection = None
        self.assertIsNone(tab.control_positions())


if __name__ == '__main__':
    unittest.main()
