from pathlib import Path
import sys
import unittest
from unittest.mock import Mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'experimental'))
from command_sequences import XYStep
from trajectory_tab import TrajectoryTab, TrajectoryArrivalGate, trajectory_waypoints


class TrajectoryTests(unittest.TestCase):
    def test_waypoint_spacing_and_exact_destination(self):
        for start, end in (((0, 0), (.012, .009)), ((.03, -.01), (-.015, .02)),
                           ((0, 0), (.005, 0)), ((0, 0), (0, 0))):
            step = XYStep(end, 2, 180, 3, 45)
            plan = trajectory_waypoints(start, step)
            points = np.array([start, *(p.target_xy_m for p in plan)])
            self.assertLessEqual(np.linalg.norm(np.diff(points, axis=0), axis=1).max(), .005+1e-12)
            self.assertEqual(plan[-1], step)
            self.assertTrue(all(p.wait_seconds == 0 for p in plan[:-1]))
            self.assertTrue(all(p.ratio == 3 and p.angle_deg == 45 for p in plan))

    def test_auto_arrival_requires_hold_and_clears_on_tracking_loss(self):
        now = [0.]
        gate = TrajectoryArrivalGate(1, .5, clock=lambda: now[0])
        gate.arm((.005, 0))
        now[0] = .1
        gate.observe([[.005, 0]])
        self.assertFalse(gate.advance_due)
        now[0] = .7
        gate.observe([[.005, 0]])
        self.assertTrue(gate.advance_due)
        gate.observe(None)
        self.assertFalse(gate.advance_due)
        self.assertFalse(gate.ready)

    def fixture(self):
        tab = TrajectoryTab.__new__(TrajectoryTab)
        tab.trajectory_start_xy_m = (0, 0)
        tab.run_allow_extrapolation = False
        tab.setup = dict(coordinate_system='robot_axes', corners_px=[(10,10),(210,10),(210,210),(10,210)],
                         rectangle_width_cm=10, rectangle_height_cm=10, video_size_px=[220,220],
                         axis_samples_px={'Origin':[110,110], '+X':[130,110], '+Y':[110,90]})
        return tab

    def test_selected_destination_runs_all_waypoints_automatically(self):
        tab, now = self.fixture(), [0.]
        gate = TrajectoryArrivalGate(.5, .2, clock=lambda: now[0])
        commander, report = Mock(), Mock()
        targets = []
        def move(case, target, **kwargs):
            self.assertTrue(kwargs['active']())
            targets.append(target)
            result = {'angles_deg': [60, 70]}
            kwargs['on_sent'](result)
            now[0] += .1
            gate.observe([target])
            self.assertFalse(gate.advance_due)
            now[0] += .3
            gate.observe([target])
            self.assertTrue(gate.advance_due)
            return result
        commander.move.side_effect = move
        result = tab.execute_sequence([XYStep((.012, 0), 0, 180)], commander, 'case',
            active=lambda: True, report=report, arrival_gate=gate, prepared=True)
        self.assertEqual(result, 'finished')
        np.testing.assert_allclose(targets, [[.004,0], [.008,0], [.012,0]])
        self.assertEqual(sum(call.args[0] == 'completed' for call in report.call_args_list), 1)
        self.assertTrue(any(call.args[0] == 'trajectory_waypoint' for call in report.call_args_list))
        self.assertTrue(all('prepared_point' not in call.kwargs for call in commander.move.call_args_list))

    def test_invalid_later_destination_blocks_entire_route_before_motor_io(self):
        tab, commander = self.fixture(), Mock()
        with self.assertRaises(ValueError):
            tab.execute_sequence([XYStep((.01,0),0,180), XYStep((.1,0),0,180)], commander, 'case',
                                 active=lambda: True, report=Mock())
        commander.move.assert_not_called()

    def test_stop_prevents_subsequent_waypoints(self):
        tab, commander, active = self.fixture(), Mock(), [True]
        def move(case, target, **kwargs):
            active[0] = False
            return {'angles_deg': [60]}
        commander.move.side_effect = move
        result = tab.execute_sequence([XYStep((.02,0),0,180)], commander, 'case',
                                      active=lambda: active[0], report=Mock())
        self.assertEqual(result, 'stopped')
        self.assertEqual(commander.move.call_count, 1)


if __name__ == '__main__':
    unittest.main()
