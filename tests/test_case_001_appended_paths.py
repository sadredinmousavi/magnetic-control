import importlib
import unittest

import numpy as np

from case_loader import load_case, validate_target_schedule


class Case001AppendedPathTests(unittest.TestCase):
    def test_condition_011_reuses_case_002_condition_005_path(self):
        appended = importlib.import_module("cases.case_001.cond_011")
        source = importlib.import_module("cases.case_002.cond_005")

        np.testing.assert_allclose(appended.PATH_POINTS, source.PATH_POINTS)
        self.assertIs(appended.PATH_V3_EDGE_POLYLINES, source.PATH_V3_EDGE_POLYLINES)
        validate_target_schedule(load_case("case_001.cond_011")["TARGET_SCHEDULE"])

    def test_condition_012_reuses_case_002_condition_007_path(self):
        appended = importlib.import_module("cases.case_001.cond_012")
        source = importlib.import_module("cases.case_002.cond_007")

        np.testing.assert_allclose(appended.PATH_POINTS, source.PATH_POINTS)
        self.assertIs(appended.PATH_V4_EDGE_POLYLINES, source.PATH_V4_EDGE_POLYLINES)
        validate_target_schedule(load_case("case_001.cond_012")["TARGET_SCHEDULE"])

    def test_appended_conditions_keep_case_001_base_properties(self):
        case_001_params = load_case("case_001.cond_011")
        case_002_params = load_case("case_002.cond_005")

        self.assertEqual(case_001_params["ROBOT_MAGNETIZATION"], 868e3)
        self.assertEqual(case_002_params["ROBOT_MAGNETIZATION"], 1.56e5)


if __name__ == "__main__":
    unittest.main()
