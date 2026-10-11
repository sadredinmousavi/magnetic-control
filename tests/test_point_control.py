from pathlib import Path
import sys
import unittest
from unittest.mock import Mock, patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experimental"))
from point_control import PointCommander
from control_workflow import calculate_point_control


class PointControlTests(unittest.TestCase):
    def test_moves_calculate_fresh_angles_and_use_existing_motor_protocol(self):
        port = Mock(is_open=True)
        commander = PointCommander(lambda: port)
        with patch.object(commander, "calculate", return_value={"angles_deg": [60] * 8}) as calculate, \
             patch("helpers.moveArray") as move:
            for target in ((0, 0), (.01, 0), (0, .01)):
                commander.move("case_100.cond_013_test_extract_compensator", target)
                calculate.assert_called_with("case_100.cond_013_test_extract_compensator", target,
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
