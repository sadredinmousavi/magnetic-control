import ast
import re
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHEATSHEET = PROJECT_ROOT / "CASE_OPTIONS_CHEATSHEET.md"
OPTION_ROW = re.compile(r"^\| `([A-Z][A-Z0-9_]+)` \|", re.MULTILINE)


def params_keys_used_by_source(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    keys = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            owner = node.func.value
            if (
                isinstance(owner, ast.Name)
                and owner.id == "params"
                and node.func.attr == "get"
                and node.args
                and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str)
            ):
                keys.add(node.args[0].value)
        elif isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name):
            if node.value.id == "params":
                key_node = node.slice
                if isinstance(key_node, ast.Constant) and isinstance(key_node.value, str):
                    keys.add(key_node.value)
    return keys


def literal_params_keys(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    keys = set()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        if not any(isinstance(target, ast.Name) and target.id == "PARAMS" for target in targets):
            continue
        value = node.value
        if isinstance(value, ast.Dict):
            keys.update(
                key.value
                for key in value.keys
                if isinstance(key, ast.Constant) and isinstance(key.value, str)
            )
    return keys


class CaseOptionsCheatsheetTests(unittest.TestCase):
    def test_all_implemented_and_current_case_options_are_documented(self):
        documented = set(OPTION_ROW.findall(CHEATSHEET.read_text(encoding="utf-8")))
        implemented = set()
        for filename in (
            "case_loader.py",
            "control_workflow.py",
            "usage0.py",
            "usage1.py",
            "usage2.py",
            "usage3.py",
            "usage4.py",
        ):
            implemented.update(params_keys_used_by_source(PROJECT_ROOT / filename))

        case_keys = set()
        for path in (PROJECT_ROOT / "cases").glob("case_*/*.py"):
            case_keys.update(literal_params_keys(path))

        missing = sorted((implemented | case_keys) - documented)
        self.assertEqual(missing, [], f"Undocumented case PARAMS options: {missing}")


if __name__ == "__main__":
    unittest.main()
