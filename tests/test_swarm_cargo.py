from pathlib import Path
import sys
import unittest
from unittest.mock import Mock

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'experimental'))
from axis_calibration import RobotAxes
from swarm_cargo import (RedCargoTracker, swarm_geometry, measurement_positions,
                         CargoPositionFeedback, cargo_geometry, draw_measurements)
from position_control import PositionPID
from arrival_control import ArrivalGate
from command_sequences import XYStep, execute_xy_sequence


class SwarmCargoTests(unittest.TestCase):
    def test_rectangle_direction_in_metric_xy_with_perspective_and_rotated_axes(self):
        corners = [(25, 15), (385, 50), (360, 350), (10, 380)]
        axes = RobotAxes(corners, 10, 10,
                         {'Origin': [190, 190], '+X': [210, 155], '+Y': [155, 170]})
        for angle in (25, 95, 170):
            radians = np.radians(angle)
            rotation = np.array([[np.cos(radians), -np.sin(radians)],
                                 [np.sin(radians), np.cos(radians)]])
            points = np.array([[-1.5, -.5], [1.5, -.5], [1.5, .5], [-1.5, .5]]) @ rotation.T
            frame = np.full((400, 400, 3), 240, np.uint8)
            cv2.fillConvexPoly(frame, np.rint(axes.pixel_positions(points)).astype(np.int32), (0, 0, 255))
            tracker = RedCargoTracker()
            cargo = tracker.process(frame)
            rectangle = cargo_geometry(cargo, tracker.detector.last_blob_mask, axes, 'Rectangle')
            difference = (rectangle['angle_deg'] - angle + 90) % 180 - 90
            self.assertLess(abs(difference), 2)
            self.assertAlmostEqual(rectangle['length_mm'], 30, delta=1)
            self.assertAlmostEqual(rectangle['width_mm'], 10, delta=1)
            np.testing.assert_allclose(rectangle['xy_cm'], [0, 0], atol=.05)
            self.assertEqual(rectangle['direction_px'].shape, (2, 2))
            overlay = draw_measurements(np.zeros_like(frame), cargo=rectangle)
            self.assertTrue(np.any(np.all(overlay == [255, 120, 255], axis=2)))

    def test_circle_square_and_uncalibrated_cargo_do_not_report_long_axis(self):
        axes = RobotAxes([(0, 0), (200, 0), (200, 200), (0, 200)], 10, 10,
                         {'Origin': [100, 100], '+X': [120, 100], '+Y': [100, 80]})
        frame = np.full((220, 220, 3), 240, np.uint8)
        cv2.rectangle(frame, (75, 75), (125, 125), (0, 0, 255), -1)
        tracker = RedCargoTracker()
        cargo = tracker.process(frame)
        mask = tracker.detector.last_blob_mask
        square = cargo_geometry(cargo, mask, axes, 'Rectangle')
        self.assertIsNone(square['angle_deg'])
        self.assertIsNone(square['direction_px'])
        self.assertIsNotNone(square['outline_px'])
        self.assertIsNone(cargo_geometry(cargo, mask, axes, 'Circle')['angle_deg'])
        self.assertIsNone(cargo_geometry(cargo, mask, None, 'Rectangle')['angle_deg'])

    def test_red_cargo_tracks_separately_and_holds_identity_when_lost(self):
        tracker = RedCargoTracker()
        frame = np.full((240, 320, 3), 240, np.uint8)
        cv2.circle(frame, (100, 100), 15, (0, 0, 255), -1)
        cv2.circle(frame, (80, 100), 3, (20, 20, 20), -1)
        np.testing.assert_allclose(tracker.process(frame)['center'], (100, 100), atol=1.)
        first_center = tracker.center
        missing = np.full_like(frame, 240)
        cv2.circle(missing, (260, 100), 20, (0, 0, 255), -1)
        self.assertIsNone(tracker.process(missing, radius=50))
        self.assertEqual(tracker.center, first_center)
        self.assertEqual(tracker.process(frame, radius=50)['center'], first_center)

    def test_multiple_red_objects_need_selection_and_mask_limits_search(self):
        tracker = RedCargoTracker()
        frame = np.full((240, 320, 3), 240, np.uint8)
        cv2.circle(frame, (100, 100), 15, (0, 0, 255), -1)
        cv2.circle(frame, (250, 100), 20, (0, 0, 255), -1)
        self.assertIsNone(tracker.process(frame))
        self.assertIn('Multiple cargo', tracker.error)
        self.assertEqual(tracker.process(frame, selection=(95, 100), radius=40)['center'], (100, 100))
        mask = np.zeros(frame.shape[:2], np.uint8)
        mask[:, 200:] = 255
        self.assertIsNone(tracker.process(frame, radius=40, mask=mask))

    def test_metric_geometry_ratio_and_degenerate_cases(self):
        axes = RobotAxes([(0, 0), (200, 0), (200, 200), (0, 200)], 10, 10,
                         {'Origin': [100, 100], '+X': [120, 100], '+Y': [100, 80]})
        points = axes.pixel_positions([[-2, -1], [2, -1], [2, 1], [-2, 1]])
        detections = [{'center': p} for p in points]
        geometry = swarm_geometry(detections, 4, axes)
        np.testing.assert_allclose(geometry['center_mm'], [0, 0], atol=1e-6)
        self.assertAlmostEqual(geometry['ratio'], 2)
        self.assertAlmostEqual(geometry['angle_deg'], 0)
        self.assertEqual(len(geometry['lines']), 2)
        self.assertIsNone(swarm_geometry(detections[:3], 4, axes))
        self.assertIsNone(swarm_geometry(detections[:1], 1, axes)['ratio'])
        line = swarm_geometry([{'center': [80, 100]}, {'center': [120, 100]}], 2, axes)
        self.assertTrue(np.isinf(line['ratio']))
        square = swarm_geometry([{'center': p} for p in axes.pixel_positions([[-1,-1], [1,-1], [1,1], [-1,1]])], 4, axes)
        self.assertAlmostEqual(square['ratio'], 1)
        self.assertIsNone(square['angle_deg'])

    def test_cargo_observation_never_falls_back_to_robot_center(self):
        robots = [{'robot_id': 1, 'xy_cm': (1, 2)}, {'robot_id': 2, 'xy_cm': (3, 4)}]
        cargo = {'xy_cm': (5, 6)}
        np.testing.assert_allclose(measurement_positions('Cargo center', robots, 2, cargo), [[.05, .06]])
        self.assertIsNone(measurement_positions('Cargo center', robots, 2, None))
        np.testing.assert_allclose(measurement_positions('Robots', robots, 2, cargo), [[.01, .02], [.03, .04]])
        self.assertIsNone(measurement_positions('Robots', robots[:1], 2, cargo))

    def test_cargo_preview_cannot_enable_corrections(self):
        feedback = CargoPositionFeedback(PositionPID(), preview_only=True)
        feedback.set_enabled(True)
        self.assertFalse(feedback.enabled.is_set())

    def test_cargo_error_produces_bounded_position_command_and_loss_discards_it(self):
        for lose_cargo in (False, True):
            now, active = [0.], [True]
            clock = lambda: now[0]
            gate = ArrivalGate(clock=clock)
            gate.arm((0, 0))
            now[0] = .1
            gate.observe([[-.01, 0]])
            feedback = CargoPositionFeedback(PositionPID(kp=.2, limit_mm=10, slew_mm_s=1, period_s=.5,
                                                         incremental=True), clock=clock)
            feedback.set_enabled(True)
            commander, report = Mock(), Mock()
            def move(case, target, **kwargs):
                self.assertGreater(target[0], 0)
                self.assertLessEqual(target[0], .0005)  # 1 mm/s slew, 0.5 s period.
                self.assertEqual(kwargs['ratio'], 2)
                if lose_cargo:
                    gate.observe(None)
                    self.assertFalse(kwargs['active']())
                    result = None
                else:
                    self.assertTrue(kwargs['active']())
                    result = {'angles_deg': [60, 70]}
                    kwargs['on_sent'](result)
                active[0] = False
                return result
            commander.move.side_effect = move
            feedback.wait_at_target(XYStep((0, 0), 0, 90, 2, 45), commander, 'case', gate,
                                    lambda: active[0], report)
            self.assertEqual(commander.move.call_count, 1)
            self.assertEqual(report.call_count, 0 if lose_cargo else 1)

    def test_cargo_loss_during_initial_synthesis_cancels_send(self):
        visible = [True]
        feedback = CargoPositionFeedback(PositionPID(), observation_valid=lambda: visible[0])
        commander = Mock()
        def move(case, target, **kwargs):
            self.assertTrue(kwargs['active']())
            visible[0] = False
            self.assertFalse(kwargs['active']())
            return None
        commander.move.side_effect = move
        result = execute_xy_sequence([XYStep((0, 0), 0, 180)], commander, 'case',
                                     active=lambda: True, report=Mock(), feedback=feedback)
        self.assertEqual(result, 'stopped')


if __name__ == '__main__':
    unittest.main()
