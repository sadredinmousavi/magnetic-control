"""Interactive Windows launcher used by usage.bat."""

import importlib
import os
from pathlib import Path
import sys


PROJECT_DIR = Path(__file__).resolve().parent
USAGES = (
    ("0", "Usage 0 - plot a saved angle sequence"),
    ("1", "Usage 1 - plot the scheduled target trajectory"),
    ("2", "Usage 2 - optimize and save a sequence"),
    ("3", "Usage 3 - optimize and create static plots"),
    ("4", "Usage 4 - optimize, simulate, and create animation"),
)


def discover_cases(project_dir=PROJECT_DIR):
    """Return sorted case folders and their runnable condition stems."""
    cases_dir = Path(project_dir) / "cases"
    discovered = []
    for case_dir in sorted(cases_dir.glob("case_*"), key=lambda path: path.name.lower()):
        if not case_dir.is_dir() or not (case_dir / "case.py").is_file():
            continue
        conditions = sorted(
            (path.stem for path in case_dir.glob("cond_*.py") if path.is_file()),
            key=str.lower,
        )
        if conditions:
            discovered.append((case_dir.name, conditions))
    return discovered


def _resolve_typed_choice(text, choices):
    text = text.strip()
    if text.isdigit():
        index = int(text) - 1
        return index if 0 <= index < len(choices) else None

    normalized = text.lower().removesuffix(".py")
    exact = [
        index for index, choice in enumerate(choices)
        if choice.lower().removesuffix(".py") == normalized
    ]
    if exact:
        return exact[0]

    prefix = [
        index for index, choice in enumerate(choices)
        if choice.lower().startswith(normalized)
    ]
    return prefix[0] if len(prefix) == 1 else None


def _basic_menu(title, choices):
    """Fallback menu for a redirected/non-Windows terminal."""
    while True:
        print(f"\n{title}")
        print("=" * len(title))
        for index, choice in enumerate(choices, start=1):
            print(f"{index}. {choice}")
        answer = input("Number or name (Q to quit/back): ").strip()
        if answer.lower() == "q":
            return None
        selected = _resolve_typed_choice(answer, choices)
        if selected is not None:
            return selected
        print("No matching selection. Please try again.")


def select_menu(title, choices):
    """Select with arrows or by typing a one-based number/name."""
    if not choices:
        raise ValueError(f"No choices are available for {title}.")
    if os.name != "nt" or not sys.stdin.isatty():
        return _basic_menu(title, choices)

    import msvcrt

    selected = 0
    typed = ""
    message = ""
    while True:
        os.system("cls")
        print("Magnetic Control Launcher")
        print("=========================\n")
        print(title)
        print("Use Up/Down + Enter, or type a list number/name + Enter.")
        print("Esc goes back; Q quits when the input line is empty.\n")
        for index, choice in enumerate(choices, start=1):
            marker = ">" if index - 1 == selected else " "
            print(f"{marker} {index:>2}. {choice}")
        print(f"\nInput: {typed}")
        if message:
            print(message)

        key = msvcrt.getwch()
        if key in ("\x00", "\xe0"):
            special = msvcrt.getwch()
            if special == "H":
                selected = (selected - 1) % len(choices)
                typed = ""
            elif special == "P":
                selected = (selected + 1) % len(choices)
                typed = ""
            message = ""
        elif key == "\r":
            if not typed:
                return selected
            resolved = _resolve_typed_choice(typed, choices)
            if resolved is not None:
                return resolved
            message = f"No item matches '{typed}'."
            typed = ""
        elif key == "\x1b":
            return None
        elif key == "\b":
            typed = typed[:-1]
            message = ""
        elif key.lower() == "q" and not typed:
            raise SystemExit(0)
        elif key.isprintable():
            typed += key
            message = ""


def resolve_usage0_input(case_name, project_dir=PROJECT_DIR):
    """Find the sequence generated for the selected case/condition."""
    flat_name = case_name.replace(".", "_") + ".txt"
    candidates = (
        Path(project_dir) / "outputs" / flat_name,
        Path(project_dir) / "experimental" / "sequences" / flat_name,
    )
    return next((path for path in candidates if path.is_file()), None)


def run_usage(usage_number, case_name):
    module = importlib.import_module(f"usage{usage_number}")
    if usage_number == "0":
        input_file = resolve_usage0_input(case_name)
        if input_file is None:
            flat_name = case_name.replace(".", "_") + ".txt"
            raise FileNotFoundError(
                f"No saved sequence named '{flat_name}' was found in outputs or "
                "experimental/sequences. Run Usage 2 for this condition first."
            )
        print(f"Using saved sequence: {input_file}")
        return module.main(case_name=case_name, input_filename=input_file)
    return module.main(case_name=case_name)


def main():
    discovered = discover_cases()
    if not discovered:
        raise FileNotFoundError("No cases/case_*/cond_*.py files were found.")

    usage_index = select_menu("Choose a usage", [label for _, label in USAGES])
    if usage_index is None:
        return 0
    usage_number = USAGES[usage_index][0]

    while True:
        case_index = select_menu("Choose a case", [name for name, _ in discovered])
        if case_index is None:
            return main()
        case_name, conditions = discovered[case_index]

        condition_index = select_menu(
            f"Choose a condition for {case_name}", conditions
        )
        if condition_index is None:
            continue

        condition_name = conditions[condition_index]
        selected_case = f"{case_name}.{condition_name}"
        os.system("cls" if os.name == "nt" else "clear")
        print(f"Starting Usage {usage_number} with {selected_case}...\n")
        run_usage(usage_number, selected_case)
        return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, ValueError) as exc:
        print(f"\nLauncher error: {exc}", file=sys.stderr)
        raise SystemExit(1)
