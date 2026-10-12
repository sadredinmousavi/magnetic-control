from pathlib import Path
import sys
import unittest
from unittest.mock import Mock, patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experimental"))
from point_control import PointCommander, hardware_case_name, load_hardware_case
from control_workflow import calculate_point_control


class PointControlTests(unittest.TestCase):
    def test_prepare_axes_computes_once_without_motor_io_and_reuses_results(self):
        port = Mock(is_open=True)
        commander = PointCommander(lambda: port)
        params = {'NUM_SOURCES': 8, 'RADIUS': .18}
        with patch('point_control.load_hardware_case', return_value=params), \
             patch('point_control.calculate_point_control', return_value={'angles_deg': [60] * 8}) as calculate, \
             patch('helpers.moveArray') as move:
            commander.prepare_axes('case_100')
            self.assertEqual(calculate.call_count, 3)
            move.assert_not_called()
            for target in ((0, .01), (0, 0), (.01, 0), (0, .01)):
                commander.move_prepared_axis('case_100', target)
            self.assertEqual(calculate.call_count, 3)
            self.assertEqual(move.call_count, 4)
            params['RADIUS'] = .2
            with self.assertRaisesRegex(ValueError, 'Prepare axis moves'):
                commander.move_prepared_axis('case_100', (0, 0))
            self.assertEqual(move.call_count, 4)

    def test_unprepared_axis_never_sends_and_preparation_needs_no_connected_port(self):
        commander = PointCommander(lambda: None)
        with patch('point_control.load_hardware_case', return_value={'RADIUS': .18}), \
             patch('point_control.calculate_point_control', return_value={'angles_deg': [60] * 8}), \
             patch('helpers.moveArray') as move:
            commander.prepare_axes('case_100')
            move.assert_not_called()
        port = Mock(is_open=True)
        commander = PointCommander(lambda: port)
        with patch('point_control.load_hardware_case', return_value={'RADIUS': .18}), patch('helpers.moveArray') as move:
            with self.assertRaisesRegex(ValueError, 'not prepared'):
                commander.move_prepared_axis('case_100', (0, 0))
            move.assert_not_called()
    def test_only_base_hardware_model_is_loaded_even_for_legacy_conditions(self):
        params = {"NUM_SOURCES": 8, "RADIUS": .18}
        with patch("point_control.load_case", return_value=params) as load, \
             patch("point_control.calculate_point_control") as calculate:
            for name in ("case_100", "case_100.case", "case_100.cond_013_test_extract_compensator"):
                PointCommander(lambda: None).calculate(name, (.01, 0))
                load.assert_called_with("case_100.case")
                calculate.assert_called_with(params, (.01, 0), ratio=1.0, angle_deg=None)
                self.assertEqual(hardware_case_name(name), "case_100")
        with self.assertRaises(ValueError):
            load_hardware_case("")

    def test_moves_calculate_fresh_angles_and_use_existing_motor_protocol(self):
        port = Mock(is_open=True)
        commander = PointCommander(lambda: port)
        with patch.object(commander, "calculate", return_value={"angles_deg": [60] * 8}) as calculate, \
             patch("helpers.moveArray") as move:
            for target in ((0, 0), (.01, 0), (0, .01)):
                commander.move("case_100", target)
                calculate.assert_called_with("case_100", target,
                                             ratio=1.0, angle_deg=None)
            self.assertEqual(calculate.call_count, 3)
            self.assertEqual(move.call_count, 3)
            self.assertIs(move.call_args.args[0], port)
            np.testing.assert_allclose(move.call_args.args[1], [60] * 8)
            self.assertEqual(move.call_args.args[2:], (0, 180))

    def test_calculation_failure_or_cancel_never_sends_motor_command(self):
        port = Mock(is_open=True)
        commander = PointCommander(lambda: port)
        with patch.object(commander, "calculate", side_effect=RuntimeError("solver failed")), \
             patch("helpers.moveArray") as move:
            with self.assertRaises(RuntimeError):
                commander.move("case", (0, 0))
            move.assert_not_called()
        with patch.object(commander, "calculate", return_value={"angles_deg": [60] * 8}), \
             patch("helpers.moveArray") as move:
            self.assertIsNone(commander.move("case", (0, 0), active=Mock(side_effect=[True, False])))
            move.assert_not_called()

    def test_disconnected_port_never_runs_solver(self):
        commander = PointCommander(lambda: None)
        with patch.object(commander, "calculate") as calculate:
            with self.assertRaises(RuntimeError):
                commander.move("case", (0, 0))
            calculate.assert_not_called()

    def test_optional_angle_is_converted_to_radians_for_future_commands(self):
        with patch("control_workflow.build_common_config") as cfg, \
             patch("control_workflow.run_control_workflow") as run:
            run.return_value.opt_infos = [{"angles_deg": [60] * 8}]
            calculate_point_control({}, (.02, .03), ratio=2, angle_deg=90)
            schedule = cfg.call_args.args[0]["TARGET_SCHEDULE"]
            self.assertEqual(schedule[0][2], 2)
            self.assertAlmostEqual(schedule[0][3], np.pi / 2)

    def test_invalid_targets_rejected_before_solver(self):
        for target in ([0], [0, np.nan], [0, np.inf]):
            with self.assertRaises(ValueError):
                calculate_point_control({}, target)


if __name__ == "__main__":
    unittest.main()
