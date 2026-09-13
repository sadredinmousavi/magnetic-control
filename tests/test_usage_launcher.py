import tempfile
import unittest
from pathlib import Path

from usage_launcher import discover_cases, resolve_usage0_input, _resolve_typed_choice


class UsageLauncherTests(unittest.TestCase):
    def test_discovers_case_folders_and_conditions_in_name_order(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            for case_name, conditions in (
                ("case_010", ("cond_002.py", "cond_001.py")),
                ("case_002", ("cond_003.py",)),
            ):
                folder = root / "cases" / case_name
                folder.mkdir(parents=True)
                (folder / "case.py").touch()
                for condition in conditions:
                    (folder / condition).touch()

            self.assertEqual(discover_cases(root), [
                ("case_002", ["cond_003"]),
                ("case_010", ["cond_001", "cond_002"]),
            ])

    def test_typed_choice_accepts_order_or_name(self):
        choices = ["case_001", "case_002", "case_003"]
        self.assertEqual(_resolve_typed_choice("2", choices), 1)
        self.assertEqual(_resolve_typed_choice("case_003", choices), 2)
        self.assertIsNone(_resolve_typed_choice("20", choices))

    def test_usage0_prefers_matching_output_sequence(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            output = root / "outputs" / "case_002_cond_005.txt"
            experimental = (
                root / "experimental" / "sequences" / "case_002_cond_005.txt"
            )
            output.parent.mkdir(parents=True)
            experimental.parent.mkdir(parents=True)
            output.touch()
            experimental.touch()

            self.assertEqual(
                resolve_usage0_input("case_002.cond_005", root), output
            )


if __name__ == "__main__":
    unittest.main()
