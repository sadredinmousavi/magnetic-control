import unittest

from case_loader import load_case


class Case002AnimationTests(unittest.TestCase):
    def test_each_case_explicitly_uses_smaller_plain_cross(self):
        conditions = (
            "case_001.cond_001",
            "case_002.cond_005",
            "case_003.cond_001",
            "case_004.cond_001",
            "case_005.cond_001",
        )
        for condition in conditions:
            params = load_case(condition)
            self.assertEqual(params["ANIMATION_ACTIVE_TARGET_MARKER"], "x")
            self.assertEqual(params["ANIMATION_ACTIVE_TARGET_MARKER_SIZE"], 60)
            self.assertEqual(params["ANIMATION_ACTIVE_TARGET_ALPHA"], 0.7)


if __name__ == "__main__":
    unittest.main()
