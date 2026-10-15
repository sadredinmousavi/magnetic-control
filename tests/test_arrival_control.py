from pathlib import Path
import sys
import threading
import unittest
from unittest.mock import Mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experimental"))
from arrival_control import ArrivalGate
from command_sequences import XYStep, execute_xy_sequence


class ArrivalControlTests(unittest.TestCase):
    def test_manual_next_does_not_require_arrival_and_survives_detection_loss(self):
        gate = ArrivalGate(require_arrival=False)
        self.assertTrue(gate.request_next())
        gate.prepare_point(preserve_next=True)
        gate.arm((0,0), preserve_next=True)
        self.assertTrue(gate.next_requested)  # Queued during calculation; retained on send.
        self.assertFalse(gate.request_next())  # No duplicate permissions.
        gate.observe(None)
        self.assertTrue(gate.next_requested)
        self.assertTrue(gate.wait_for_next(lambda: True))
        gate.arm((.01,0))
        self.assertFalse(gate.next_requested)
        self.assertTrue(gate.wait_for_arrival(lambda: True))

    def gate(self, hold=.5):
        self.now = 10.
        return ArrivalGate(1, hold, clock=lambda: self.now)

    def observe(self, gate, positions, advance=.1):
        self.now += advance
        gate.observe(positions, self.now)

    def test_fresh_frames_and_continuous_hold_required(self):
        gate = self.gate()
        gate.arm((.01, 0))
        gate.observe([[.01, 0]], 9)
        self.assertFalse(gate.ready)
        self.observe(gate, [[.0105, 0]])
        self.assertFalse(gate.ready)
        self.observe(gate, [[.0105, 0]], .6)
        self.assertTrue(gate.ready)
        self.assertFalse(gate.request_next())
        self.assertTrue(gate.wait_for_arrival(lambda: True))
        self.assertTrue(gate.request_next())
        self.assertTrue(gate.wait_for_next(lambda: True))

    def test_missing_stale_and_out_of_range_observations_revoke_arrival(self):
        gate = self.gate(0)
        gate.arm((0, 0))
        self.observe(gate, [[0, 0]])
        self.assertTrue(gate.wait_for_arrival(lambda: True))
        gate.observe(None)
        self.assertFalse(gate.ready)
        self.assertFalse(gate.request_next())
        self.observe(gate, [[.002, 0]])
        self.assertFalse(gate.ready)
        self.observe(gate, [[0, 0]])
        self.assertTrue(gate.ready)
        self.now += 3
        self.assertFalse(gate.ready)
        self.assertFalse(gate.request_next())

    def test_all_robots_must_arrive_not_only_their_centroid(self):
        gate = self.gate(0)
        gate.arm((0, 0))
        self.observe(gate, [[-.002, 0], [.002, 0]])
        self.assertFalse(gate.ready)
        self.observe(gate, [[-.0005, 0], [.0005, 0]])
        self.assertTrue(gate.ready)

    def test_leaving_tolerance_restarts_hold_and_clears_queued_permission(self):
        gate = self.gate()
        gate.arm((0, 0))
        self.observe(gate, [[0, 0]])
        self.observe(gate, [[0, 0]], .6)
        self.assertTrue(gate.wait_for_arrival(lambda: True))
        self.assertTrue(gate.request_next())
        self.observe(gate, [[.005, 0]])
        self.assertFalse(gate.next_requested)
        self.observe(gate, [[0, 0]])
        self.assertFalse(gate.ready)
        self.observe(gate, [[0, 0]], .6)
        self.assertTrue(gate.ready)
        self.assertFalse(gate.next_requested)

    def test_cancel_breaks_both_waits_without_confirmation(self):
        gate = self.gate()
        gate.arm((0, 0))
        self.assertFalse(gate.wait_for_arrival(lambda: False))
        self.assertFalse(gate.wait_for_next(lambda: False))

    def test_frame_gap_clears_old_next_permission(self):
        gate = self.gate(0)
        gate.arm((0, 0))
        self.observe(gate, [[0, 0]])
        gate.wait_for_arrival(lambda: True)
        self.assertTrue(gate.request_next())
        self.observe(gate, [[0, 0]], 3)
        self.assertFalse(gate.next_requested)

    def test_no_automatic_advance_and_no_permission_carries_to_next_point(self):
        gate = self.gate(0)
        stop = threading.Event()
        sent, reached, finished = threading.Event(), threading.Event(), threading.Event()
        sent_indices, reached_indices = [], []
        result = []
        commander = Mock()

        def move(case, target, **kwargs):
            info = {"angles_deg": [60, 70]}
            kwargs["on_sent"](info)
            return info

        commander.move.side_effect = move

        def report(kind, index, info):
            if kind == "sent":
                sent_indices.append(index)
                sent.set()
            if kind == "arrived":
                reached_indices.append(index)
                reached.set()

        def worker():
            try:
                result.append(execute_xy_sequence([XYStep((0, 0), 0, 180), XYStep((.01, 0), 0, 180)],
                              commander, "case_100", active=lambda: not stop.is_set(), report=report,
                              arrival_gate=gate))
            finally:
                finished.set()

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()
        try:
            self.assertTrue(sent.wait(1))
            self.assertFalse(gate.request_next())
            self.observe(gate, [[0, 0]])
            self.assertTrue(reached.wait(1))
            self.assertEqual(sent_indices, [0])
            sent.clear()
            self.assertTrue(gate.request_next())
            self.assertTrue(sent.wait(1))
            self.assertEqual(sent_indices, [0, 1])
            self.assertFalse(gate.ready)
            self.assertFalse(gate.request_next())
            self.assertEqual(reached_indices, [0])
            stop.set()
            self.assertTrue(finished.wait(1))
            self.assertEqual(result, ["stopped"])
        finally:
            stop.set()
            thread.join(1)


if __name__ == "__main__":
    unittest.main()
