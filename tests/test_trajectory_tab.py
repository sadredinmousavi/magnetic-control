from pathlib import Path
import sys
import unittest
from unittest.mock import Mock, patch
import queue
import threading

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'experimental'))
from command_sequences import XYStep
from trajectory_tab import TrajectoryTab, TrajectoryArrivalGate, trajectory_waypoints
from control_workflow import ForcePIDFeedback, OptimizerXYFeedback
from incremental_xy_control import IncrementalXYFeedback
from arrival_control import ArrivalGate
from command_tab import CommandTab


class TrajectoryTests(unittest.TestCase):
    def test_skip_bypasses_only_current_waypoint_and_next_still_requires_arrival(self):
        tab, now = self.fixture(), [0.]
        feedback = ForcePIDFeedback(incremental_angles=True, clock=lambda: now[0])
        gate = TrajectoryArrivalGate(.25, .2, velocity_tau_s=0., clock=lambda: now[0])
        commander, report = Mock(), Mock()
        commander.applied_pose.return_value = dict(angles_deg=[60.]*8, sent_time_s=0.)
        next_observations = [0]
        def advance(timeout):
            now[0] = round(now[0]+.1, 5)
            if np.allclose(gate.target, [.002, 0.]):
                gate.observe([[0., 0.]], now[0])  # Never reaches the first waypoint.
                self.assertTrue(gate.request_skip())
                self.assertFalse(gate.request_skip())
                self.assertFalse(gate.ready)  # Skipping does not pretend the robot arrived.
            else:
                self.assertFalse(gate.skip_requested)
                gate.observe([gate.target], now[0])
                next_observations[0] += 1
                if next_observations[0] < 3:
                    self.assertFalse(gate.advance_due)
        gate.condition.wait = advance
        result = tab.execute_sequence([XYStep((.004, 0.), 0., 180.)], commander, 'case',
            feedback=feedback, arrival_gate=gate, active=lambda: now[0] < 3., report=report)
        self.assertEqual(result, 'finished')
        self.assertGreaterEqual(next_observations[0], 3)
        self.assertEqual(gate.skip_count, 1)
        skipped = [call for call in report.call_args_list if call.args[0] == 'waypoint_skipped']
        self.assertEqual(len(skipped), 1)
        np.testing.assert_allclose(skipped[0].args[2]['target_xy_m'], [.002, 0.])
        commander.move.assert_not_called()

    def test_skip_is_one_shot_ignores_wait_and_does_not_survive_rearming(self):
        now = [0.]
        gate = TrajectoryArrivalGate(.25, 1., clock=lambda: now[0])
        self.assertFalse(gate.request_skip())  # No active target.
        gate.arm((.004, 0.), minimum_wait_s=60.)
        self.assertTrue(gate.request_skip())
        gate.observe([[0., 0.]], .1)
        self.assertTrue(gate.advance_due)
        self.assertFalse(gate.ready)
        gate.consume_advance()
        self.assertTrue(gate.last_advance_skipped)
        self.assertFalse(gate.request_skip())
        gate.prepare_point()
        gate.arm((.008, 0.))
        self.assertFalse(gate.advance_due)
        self.assertFalse(gate.last_advance_skipped)
        self.assertEqual(gate.skip_count, 1)

    def test_manual_first_click_and_next_use_original_rows_and_preserve_u(self):
        tab, now = self.fixture(), [0.]
        steps = [XYStep((.004, 0.), 0., 180.), XYStep((.008, 0.), 0., 180.)]
        feedback = ForcePIDFeedback(manual=True)
        gate = ArrivalGate(require_arrival=False, clock=lambda: now[0])
        commander, report, targets = Mock(), Mock(), []
        clicked = [False]
        def move(case, target, **kwargs):
            self.assertTrue(kwargs['active']())
            targets.append(np.asarray(target).copy())
            result = dict(angles_deg=[60.]*8)
            kwargs['on_sent'](result)
            return result
        commander.move.side_effect = move
        def advance(timeout):
            now[0] = round(now[0]+.1, 5)
            gate.observe([[0., 0.]], now[0])
            if feedback.current_step is steps[0] and not clicked[0]:
                feedback.request_manual(steps[0], [1., 2.])
                clicked[0] = True
            elif feedback.current_step is steps[0] and len(targets) >= 2:
                self.assertTrue(gate.request_next())
            elif feedback.current_step is steps[1]:
                self.assertTrue(gate.request_next())
        gate.condition.wait = advance
        result = tab.execute_sequence(steps, commander, 'case', feedback=feedback,
            arrival_gate=gate, active=lambda: now[0] < 3., report=report,
            initial_corrections=[[.5, 0.], [-.5, 0.]])
        self.assertEqual(result, 'finished')
        np.testing.assert_allclose(targets, [[.0045, 0.], [.005, .002], [.0075, 0.]])
        self.assertIs(feedback.current_step, steps[1])
        self.assertFalse(any(call.args[0] == 'trajectory_waypoint' for call in report.call_args_list))

    def test_manual_run_does_not_require_automatic_control_on_or_seed_pose(self):
        tab = self.fixture()
        tab.running = False
        tab.manual_mode_var = Mock(get=Mock(return_value=True))
        tab.control_requested = False
        tab.commander = Mock()
        with patch.object(CommandTab, 'run') as run:
            tab.run(selected_only=False)
        run.assert_called_once_with(selected_only=False)
        tab.commander.applied_pose.assert_not_called()

    def test_manual_gate_and_feedback_use_explicit_next_without_parsing_auto_settings(self):
        tab = self.fixture()
        tab.manual_mode_var = Mock(get=Mock(return_value=True))
        tab.arrival_tolerance_var = Mock(get=Mock(return_value='1.5'))
        tab.arrival_hold_var = Mock(get=Mock(return_value='1'))
        gate = tab.make_arrival_gate()
        feedback = tab.configure_run_feedback(ForcePIDFeedback())
        self.assertFalse(gate.require_arrival)
        self.assertTrue(feedback.manual)
        self.assertFalse(feedback.incremental_angles)
        self.assertFalse(feedback.enabled.is_set())

    def test_incremental_controller_selection_ignores_unused_pid_gains(self):
        tab = self.fixture()
        tab.controller_var = Mock(get=Mock(return_value='Incremental move-and-settle'))
        values = {'Kp': 'unused', 'Ki': 'unused', 'Kv': 'unused', 'I max µN': 'unused',
                  'Period s': '.25', 'Velocity filter s': '.15', 'Deadband deg': '.1'}
        tab.force_pid_settings = {key: Mock(get=Mock(return_value=value)) for key, value in values.items()}
        tab.incremental_settings = {key: Mock(get=Mock(return_value=value)) for key, value in {
            'Step mm': '1', 'Rate mm/s': '.5', 'Settle speed mm/s': '.2',
            'Settle hold s': '.75', 'Max offset mm': '5', 'Step timeout s': '10'}.items()}
        tab.force_preview_var = Mock(get=Mock(return_value=False))
        tab.force_hold_speed_var = Mock(get=Mock(return_value='.5'))
        tab.control_requested = True
        feedback = tab.configure_run_feedback(ForcePIDFeedback())
        self.assertIsInstance(feedback, IncrementalXYFeedback)
        self.assertEqual(feedback.step_m, .001)
        self.assertEqual(feedback.speed_limit_mm_s, .5)
        self.assertEqual(feedback.rate_m_s, .0005)
        self.assertTrue(feedback.enabled.is_set())
        for key in ('Kp', 'Ki', 'Kv', 'I max µN'):
            tab.force_pid_settings[key].get.assert_not_called()

    def test_arrival_setting_accepts_1_5_mm_and_requires_full_low_speed_hold(self):
        tab = self.fixture()
        tab.arrival_tolerance_var = Mock(get=Mock(return_value='1.5'))
        tab.arrival_hold_var = Mock(get=Mock(return_value='1'))
        tab.force_hold_speed_var = Mock(get=Mock(return_value='.5'))
        tab.force_pid_settings = {'Velocity filter s': Mock(get=Mock(return_value='.15'))}
        gate = tab.make_arrival_gate()
        now = [0.]
        gate.clock = lambda: now[0]
        gate.arm((0., 0.))
        for stamp in (.1, .2, .7):
            now[0] = stamp
            gate.observe([[.0014, 0.]], stamp)
            self.assertFalse(gate.advance_due)
        now[0] = 1.3
        gate.observe([[.0014, 0.]], now[0])
        self.assertTrue(gate.advance_due)
        now[0] = 1.4
        gate.observe([[.0016, 0.]], now[0])
        self.assertFalse(gate.advance_due)
        for invalid in ('0', '-1', 'nan', 'inf'):
            tab.arrival_tolerance_var.get.return_value = invalid
            with self.assertRaises(ValueError):
                tab.make_arrival_gate()

    def test_commanded_marker_uses_sent_optimizer_xy_instead_of_row_or_waypoint(self):
        tab = self.fixture()
        tab.steps, tab.active_index, tab.running = [XYStep((.04,0),0,180)], 0, True
        snapshot = dict(mode='anchored_xy_optimizer_pid', desired_xy_mm=[4.,0.],
                        correction_mm=[1.,2.], commanded_xy_mm=[5.,2.],
                        sent_optimizer_xy_mm=[5.,2.], proposed_xy_mm=[7.,3.])
        tab.feedback = Mock(snapshot=Mock(return_value=snapshot))
        np.testing.assert_allclose(tab.commanded_pixel(), [120.,106.], atol=1e-5)
        snapshot['mode'] = IncrementalXYFeedback.control_mode
        np.testing.assert_allclose(tab.commanded_pixel(), [120.,106.], atol=1e-5)
        tab.running, tab.active_index = False, None
        np.testing.assert_allclose(tab.commanded_pixel(), [120.,106.], atol=1e-5)
        snapshot['sent_optimizer_xy_mm'] = snapshot['commanded_xy_mm'] = None
        self.assertIsNone(tab.commanded_pixel())

    def test_xy_optimizer_configuration_uses_position_units_and_no_rate_limit(self):
        tab = self.fixture()
        values = {'Kp': '.25', 'Ki': '.01', 'Kv': '.5', 'I max µN': '2', 'Period s': '.25',
                  'Velocity filter s': '.15', 'Deadband deg': '.1', 'Rate deg/s': '0'}
        tab.force_pid_settings = {key: Mock(get=Mock(return_value=value)) for key,value in values.items()}
        tab.force_preview_var = Mock(get=Mock(return_value=False))
        tab.force_hold_speed_var = Mock(get=Mock(return_value='.5'))
        tab.control_requested = True
        feedback = tab.configure_run_feedback(ForcePIDFeedback())
        self.assertIsInstance(feedback, OptimizerXYFeedback)
        self.assertEqual(feedback.speed_limit_mm_s, .5)
        self.assertEqual(feedback.pid.kp, .25)
        self.assertEqual(feedback.pid.ki, .01)
        self.assertEqual(feedback.pid.kv, .5)
        self.assertEqual(feedback.pid.integral_limit, .002)
        self.assertEqual(feedback.angle_rate, 0.)
        self.assertTrue(feedback.enabled.is_set())
        from direction_xy_control import DirectionWeightedPID, DirectionWeightedXYFeedback
        tab.controller_var = Mock(get=Mock(return_value='Direction-weighted XY PID'))
        tab.direction_settings = {key: Mock(get=Mock(return_value=value)) for key, value in {
            'Sideways integral weight': '.2', 'Opposing decay s': '2'}.items()}
        directed = tab.configure_run_feedback(ForcePIDFeedback())
        self.assertIsInstance(directed, DirectionWeightedXYFeedback)
        self.assertIsInstance(directed.pid, DirectionWeightedPID)
        self.assertEqual(directed.pid.sideways_weight, .2)
        self.assertEqual(directed.pid.opposing_decay_s, 2.)
        self.assertEqual(directed.pid.integral_limit, feedback.pid.integral_limit)
        self.assertTrue(directed.enabled.is_set())

    def test_incremental_route_seeds_once_and_keeps_latest_angles_between_waypoints(self):
        tab, commander = self.fixture(), Mock()
        commander.applied_pose.return_value = dict(angles_deg=[60.]*8, sent_time_s=0., force_compensated=True)
        feedback = ForcePIDFeedback(incremental_angles=True)
        feedback.enabled.set()
        inputs = []
        def wait(step, ignored_commander, case, gate, active, report):
            inputs.append(feedback.last_result['angles_deg'][0])
            feedback.last_result = dict(angles_deg=[inputs[-1]+1]*8, force_compensated=True)
            feedback.pid.seed_integral([1e-8, 0.])
            return True
        feedback.wait_at_target = wait
        result = tab.execute_sequence([XYStep((.012,0),0,180)], commander, 'case_100',
            active=lambda: True, report=Mock(), arrival_gate=TrajectoryArrivalGate(), feedback=feedback)
        self.assertEqual(result, 'finished')
        self.assertEqual(inputs, [60., 61., 62., 63., 64., 65.])
        commander.applied_pose.assert_called_once_with('case_100', 180)
        commander.move.assert_not_called()
        np.testing.assert_allclose(feedback.pid.integral_force, [1e-8, 0.])

    def test_waypoint_spacing_and_exact_destination(self):
        for start, end in (((0, 0), (.012, .009)), ((.03, -.01), (-.015, .02)),
                           ((0, 0), (.005, 0)), ((0, 0), (0, 0))):
            step = XYStep(end, 2, 180, 3, 45)
            plan = trajectory_waypoints(start, step)
            points = np.array([start, *(p.target_xy_m for p in plan)])
            self.assertLessEqual(np.linalg.norm(np.diff(points, axis=0), axis=1).max(), .002+1e-12)
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
        now[0] = .2
        gate.observe([[.005, 0]])
        self.assertFalse(gate.advance_due)
        now[0] = .8
        gate.observe([[.005, 0]])
        self.assertTrue(gate.advance_due)
        gate.observe(None)
        self.assertFalse(gate.advance_due)
        self.assertFalse(gate.ready)

    def test_fast_crossing_cannot_advance_until_position_and_speed_hold_together(self):
        now = [0.]
        gate = TrajectoryArrivalGate(1., 1., max_speed_mm_s=.5, velocity_tau_s=0., clock=lambda: now[0])
        gate.arm((0.,0.))
        # Remains inside the 1 mm disk for over one second, moving at 1 mm/s.
        for stamp in np.arange(.1, 1.7, .2):
            now[0] = float(stamp)
            gate.observe([[(-.8+stamp-.1)*.001,0]], now[0])
            self.assertFalse(gate.ready)
            self.assertFalse(gate.advance_due)
        # Stationary samples start a new full hold; earlier in-range motion does not count.
        for stamp in (1.7, 1.9, 2.5):
            now[0] = stamp
            gate.observe([[.0006,0]], now[0])
            self.assertFalse(gate.advance_due)
        now[0] = 2.95
        gate.observe([[.0006,0]], now[0])
        self.assertTrue(gate.advance_due)
        now[0] = 3.15
        gate.observe([[.0008,0]], now[0])
        self.assertFalse(gate.advance_due)
        self.assertFalse(gate.ready)

    def test_low_speed_hold_restarts_after_tracking_loss_or_camera_gap(self):
        now = [0.]
        gate = TrajectoryArrivalGate(1., .5, clock=lambda: now[0])
        gate.arm((0.,0.))
        for stamp in (.1,.2,.8):
            now[0] = stamp
            gate.observe([[0.,0]], stamp)
        self.assertTrue(gate.ready)
        gate.observe(None)
        now[0] = 1.
        gate.observe([[0.,0]], now[0])
        self.assertFalse(gate.ready)
        self.assertIsNone(gate.speed_mm_s)
        for stamp in (1.2,1.8):
            now[0] = stamp
            gate.observe([[0.,0]], stamp)
        self.assertTrue(gate.ready)
        now[0] = 3.
        gate.observe([[0.,0]], now[0])
        self.assertFalse(gate.ready)
        self.assertFalse(gate.advance_due)

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
            now[0] += .1
            gate.observe([target])
            now[0] += .3
            gate.observe([target])
            self.assertTrue(gate.advance_due)
            return result
        commander.move.side_effect = move
        result = tab.execute_sequence([XYStep((.012, 0), 0, 180)], commander, 'case',
            active=lambda: True, report=report, arrival_gate=gate, prepared=True)
        self.assertEqual(result, 'finished')
        np.testing.assert_allclose(targets, [[.002,0], [.004,0], [.006,0], [.008,0], [.010,0], [.012,0]])
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

    def test_center_reset_sends_one_ratio_one_command_without_feedback_or_arrival(self):
        tab, commander, feedback, gate = self.fixture(), Mock(), Mock(), Mock()
        tab._center_state = {'reset': True}
        tab.trajectory_start_xy_m = (.05, .03)
        commander.move.return_value = {'angles_deg': [60, 70]}
        result = tab.execute_sequence([XYStep((0., 0.), 0., 180, 1., None)], commander, 'case',
            active=lambda: True, report=Mock(), feedback=feedback, arrival_gate=gate,
            prepared=True, initial_corrections=[[4., 3.]])
        self.assertEqual(result, 'finished')
        commander.move.assert_called_once()
        call = commander.move.call_args
        self.assertEqual(call.args[1], (0., 0.))
        self.assertEqual(call.kwargs['ratio'], 1.)
        self.assertIsNone(call.kwargs['angle_deg'])
        feedback.begin_point.assert_not_called()
        feedback.wait_at_target.assert_not_called()
        gate.arm.assert_not_called()

    def test_center_reset_restores_loaded_sequence_after_command(self):
        tab = self.fixture()
        original = [XYStep((.02, .01), 1, 180, 2)]
        tab.steps, tab.pixels = original, None
        tab.running, tab.precalculating = False, False
        tab.operation_allowed = lambda: True
        tab.sequence_path, tab.input_units = 'original.txt', 'mm'
        tab.display_corrections = {}
        tab.table, tab.file_label, tab.status = Mock(), Mock(), Mock()
        tab.point_ratio_var, tab.point_angle_var = Mock(), Mock()
        tab.refresh_next_button, tab.render = Mock(), Mock()
        tab.table.selection.return_value = ('0',)
        tab.table.get_children.return_value = ('0',)
        tab.table.item.return_value = {'text': '1', 'values': ['20', '10'], 'tags': []}
        observed = []
        tab.run_selected = lambda: observed.append(tab.steps[0])
        tab.move_to_center()
        self.assertEqual(observed, [XYStep((0., 0.), 0., 180, 1., None)])
        self.assertIs(tab.steps, original)
        self.assertEqual(tab.sequence_path, 'original.txt')
        self.assertIsNone(tab._center_state)

    def test_center_reset_needs_only_hardware_case_and_serial_connection(self):
        tab = self.fixture()
        tab._center_state = {'reset': True}
        tab.steps = [XYStep((0., 0.), 0., 180, 1.)]
        tab.running, tab.job = False, 0
        tab.stop_event, tab.events = threading.Event(), queue.Queue()
        tab.calibration = Mock()
        tab.calibration.control_case_var.get.return_value = 'case_100'
        port = Mock(is_open=True)
        tab.commander = Mock()
        tab.commander.port_provider.return_value = port
        tab.commander.move.return_value = {'angles_deg': [60, 70]}
        tab.frame, tab.tracking_tracker = None, None
        tab.camera_running = Mock(return_value=False)
        tab.current_setup = Mock(side_effect=AssertionError('No calibration needed'))
        tab.control_positions = Mock(side_effect=AssertionError('No detection needed'))
        for name in ('parent', 'set_running', 'load_button', 'run_button', 'run_selected_button',
                     'precalculate_button', 'stop_button', 'status', 'refresh_next_button'):
            setattr(tab, name, Mock())
        with patch('trajectory_tab.threading.Thread') as thread:
            tab.run(selected_only=True)
            thread.call_args.kwargs['target']()
        tab.commander.move.assert_called_once()
        call = tab.commander.move.call_args
        self.assertEqual(call.args, ('case_100', (0., 0.)))
        self.assertEqual(call.kwargs['ratio'], 1.)
        self.assertTrue(call.kwargs['active']())
        tab.current_setup.assert_not_called()
        tab.control_positions.assert_not_called()
        tab.camera_running.assert_not_called()
        self.assertIsNone(tab.feedback)
        self.assertIsNone(tab.arrival_gate)


if __name__ == '__main__':
    unittest.main()
