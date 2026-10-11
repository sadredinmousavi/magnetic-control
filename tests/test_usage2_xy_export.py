import ast
from contextlib import redirect_stdout
import io
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

import usage2


class Usage2XYExportTests(unittest.TestCase):
    def test_each_export_pairs_xy_with_angle_rows_and_same_filename(self):
        result = SimpleNamespace(
            opt_infos=[{"angles_deg": [60, 70]}, {"angles_deg": [80, 90]}],
            scheduled_controls=[
                SimpleNamespace(target_positions=(np.array([.0123456789, -.02]),)),
                SimpleNamespace(target_positions=(np.array([.03, .04]), np.array([-.03, -.04]))),
            ],
        )
        original_dir = Path.cwd()
        with tempfile.TemporaryDirectory() as directory:
            try:
                os.chdir(directory)
                with patch("usage2.load_case", return_value={"SEQUENCE_WAIT": 3}), \
                     patch("usage2.require_keys"), patch("usage2.build_common_config"), \
                     patch("usage2.run_control_workflow", return_value=result), \
                     redirect_stdout(io.StringIO()):
                    output = usage2.main("case_001.cond_001")
                    sequence = Path("experimental/sequences") / output.name
                    xy = Path("experimental/xy_sequences") / sequence.name
                    angle_rows = [line for line in sequence.read_text().splitlines() if not line.startswith("#")]
                    xy_rows = [line for line in xy.read_text().splitlines() if not line.startswith("#")]
                    self.assertEqual(len(xy_rows), len(angle_rows))
                    self.assertEqual(ast.literal_eval(xy_rows[0].split("|")[0]), [.0123456789, -.02])
                    self.assertEqual(ast.literal_eval(xy_rows[1].split("|")[0]), [[.03, .04], [-.03, -.04]])
                    for angle_row, xy_row in zip(angle_rows, xy_rows):
                        self.assertEqual(angle_row.split("|")[1:], xy_row.split("|")[1:])
                    result.scheduled_controls[0].target_positions = (np.array([.05, .06]),)
                    usage2.main("case_001.cond_001")
                    first_xy_row = next(line for line in xy.read_text().splitlines() if not line.startswith("#"))
                    self.assertEqual(ast.literal_eval(first_xy_row.split("|")[0]), [.05, .06])
            finally:
                os.chdir(original_dir)


if __name__ == "__main__":
    unittest.main()
