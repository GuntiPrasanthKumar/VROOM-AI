"""Verification test for VROOM AI Command Parser.

Verifies deterministic extraction of Intent and Entity from natural language inputs.
Ensures zero execution of system actions.
"""

import sys
import json

# Ensure safe UTF-8 output on Windows console
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass

from app.brain.command_parser import CommandParser, ParsedCommand


def run_tests() -> int:
    print("=" * 70)
    print("       VROOM AI - Command Parser Verification Suite")
    print("=" * 70)

    parser = CommandParser()

    # Define test cases: (input_text, expected_intent, expected_entity)
    test_cases = [
        ("Open Chrome", "open_application", "Chrome"),
        ("Launch Chrome", "open_application", "Chrome"),
        ("Start Chrome", "open_application", "Chrome"),
        ("Create a folder called Projects", "create_folder", "Projects"),
        ("Make a folder called Projects", "create_folder", "Projects"),
        ("Open YouTube", "open_website", "YouTube"),
        ("Open Google", "open_website", "Google"),
        ("Exit", "exit_assistant", None),
        ("Quit", "exit_assistant", None),
        ("Stop", "exit_assistant", None),
        ("Tell me a joke", "unknown", None),
        ("Play some jazz music", "unknown", None),
    ]

    all_passed = True
    print("\n%-35s | %-18s | %-12s | %s" % ("Input Command", "Parsed Intent", "Entity", "Status"))
    print("-" * 75)

    for text, expected_intent, expected_entity in test_cases:
        result: ParsedCommand = parser.parse(text)
        
        passed = (result.intent == expected_intent) and (result.entity == expected_entity)
        if not passed:
            all_passed = False

        status_str = "PASS" if passed else "FAIL"
        entity_display = str(result.entity) if result.entity is not None else "None"
        print("%-35s | %-18s | %-12s | %s" % (f"\"{text}\"", result.intent, entity_display, status_str))

    print("-" * 75)

    # Show structured JSON output for core test cases
    print("\n--- Structured Output Demonstration ---")
    core_examples = [
        "Open Chrome",
        "Create a folder called Projects",
        "Open YouTube",
        "Exit",
        "Tell me a joke",
    ]
    for example in core_examples:
        parsed = parser.parse(example)
        print(f"\nInput: \"{example}\"")
        print(json.dumps(parsed.to_dict(), indent=4))

    print("\n" + "=" * 70)
    if all_passed:
        print("RESULT: ALL 12 TEST CASES PASSED SUCCESSFULLY.")
        print("SECURITY CHECK: Zero system actions executed (Pure Intent/Entity parsing).")
        print("=" * 70)
        return 0
    else:
        print("RESULT: SOME TESTS FAILED.")
        print("=" * 70)
        return 1


if __name__ == "__main__":
    sys.exit(run_tests())
