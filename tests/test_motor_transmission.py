from pathlib import Path
import sys
import unittest
from unittest.mock import Mock, patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'experimental'))
from helpers import moveArray
from point_control import PointCommander


class MotorTransmissionTests(unittest.TestCase):
    def test_exact_duplicate_packet_count_and_spacing(self):
        for count in (1, 2, 3):
            with self.subTest(count=count):
                port = Mock()
                with patch('helpers.sleep') as sleep:
                    moveArray(port, [10., 20.], .2, 180., repetitions=count)
                packets = [call.args[0] for call in port.write.call_args_list]
                self.assertEqual(len(packets), 2*count)
                self.assertEqual([packet[2] for packet in packets], [1]*count+[2]*count)
                self.assertTrue(all(packet == packets[0] for packet in packets[:count]))
                self.assertTrue(all(packet == packets[count] for packet in packets[count:]))
                self.assertEqual(sum(call.args == (.01,) for call in sleep.call_args_list), 2*(count-1))
                sleep.assert_called_with(.2)

    def test_legacy_default_and_invalid_counts(self):
        port = Mock()
        with patch('helpers.sleep'):
            moveArray(port, [0.], 0, 180.)
        self.assertEqual(port.write.call_count, 3)
        for count in (0, 4, 2.5):
            with self.assertRaises(ValueError):
                moveArray(port, [0.], 0, 180., repetitions=count)

    def test_shared_commander_repeats_packets_without_repeating_optimization(self):
        port = Mock(is_open=True)
        commander = PointCommander(lambda: port)
        result = {'angles_deg': [60.]*8}
        for count in (1, 2, 3):
            commander.set_motor_repetitions(count)
            with patch.object(commander, 'calculate', return_value=result) as calculate, \
                 patch('point_control.load_hardware_case', return_value={}), \
                 patch('point_control.calculate_force_pid_control', return_value=result) as force, \
                 patch('helpers.moveArray') as send:
                commander.move('case', (0, 0))
                calculate.assert_called_once()
                self.assertEqual(send.call_args.kwargs, {'repetitions': count})
                np.testing.assert_allclose(send.call_args.args[1], [60.]*8)
                commander.move_force_pid('case', (0, 0), (.004, 0), [61.]*8, [.5]*8, [1e-8, 0])
                force.assert_called_once()
                self.assertEqual(send.call_args.kwargs, {'repetitions': count})
                self.assertEqual(send.call_count, 2)
        for invalid in (0, 4, 2.5):
            with self.assertRaises(ValueError):
                commander.set_motor_repetitions(invalid)


if __name__ == '__main__':
    unittest.main()
