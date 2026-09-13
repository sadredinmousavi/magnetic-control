import unittest

import numpy as np

from usage4 import _PartialSolutionRecorder


class Usage4PartialStopTests(unittest.TestCase):
    def test_partial_recorder_retains_ordered_animation_samples(self):
        recorder = _PartialSolutionRecorder(
            t_eval=np.linspace(0.0, 2.0, 5),
            initial_state=np.array([1.0, 2.0]),
        )

        recorder.update(0.2, [2.0, 3.0])
        recorder.update(0.6, [3.0, 4.0])
        recorder.update(0.55, [99.0, 99.0])
        recorder.update(1.1, [4.0, 5.0])
        solution = recorder.build_solution()

        self.assertTrue(np.allclose(solution.t, [0.0, 0.6, 1.1]))
        self.assertEqual(solution.y.shape, (2, 3))
        self.assertTrue(np.all(np.diff(solution.t) > 0.0))
        self.assertFalse(solution.success)
        self.assertIn("stopped by user", solution.message)


if __name__ == "__main__":
    unittest.main()
