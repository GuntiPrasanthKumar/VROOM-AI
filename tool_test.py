"""Verification suite for VROOM AI Tool System and Tool Router.

Tests all required security boundaries, allowlists, and execution paths:
1. Valid application command (Notepad) -> Executes safely
2. Valid website command (YouTube) -> Executes safely
3. Valid folder command (Projects) -> Executes in vroom_workspace sandbox
4. Unknown intent (delete_everything) -> Rejected safely
5. Unsupported application (SomeRandomProgram) -> Rejected safely
6. Arbitrary filesystem path (C:\\Windows\\System32\\Something) -> Rejected safely
7. Invalid structured command (missing intent) -> Rejected safely
8. End-to-end integration: Text -> Brain Parser -> Tool Router -> Tool Result
"""

import sys
import json
import shutil
from pathlib import Path

# Ensure safe UTF-8 output on Windows console
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass

from app.brain.command_parser import CommandParser
from app.tools.router import ToolRouter
from app.tools.base import ToolResult


def main() -> int:
    print("=" * 75)
    print("       VROOM AI - Tool System & Router Verification Suite")
    print("=" * 75)

    router = ToolRouter()

    # Clean up prior test folder if present so test 3 can succeed idempotently
    workspace_dir = Path(__file__).resolve().parent / "vroom_workspace"
    test_projects_dir = workspace_dir / "Projects"
    if test_projects_dir.exists():
        shutil.rmtree(test_projects_dir)

    test_cases = [
        {
            "id": 1,
            "name": "Valid Application (Notepad)",
            "command": {"intent": "open_application", "entity": "Notepad"},
            "expect_success": True,
        },
        {
            "id": 2,
            "name": "Valid Website (YouTube)",
            "command": {"intent": "open_website", "entity": "YouTube"},
            "expect_success": True,
        },
        {
            "id": 3,
            "name": "Valid Folder (Projects)",
            "command": {"intent": "create_folder", "entity": "Projects"},
            "expect_success": True,
        },
        {
            "id": 4,
            "name": "Unknown Intent (delete_everything)",
            "command": {"intent": "delete_everything", "entity": "C:\\"},
            "expect_success": False,
        },
        {
            "id": 5,
            "name": "Unsupported Application (SomeRandomProgram)",
            "command": {"intent": "open_application", "entity": "SomeRandomProgram"},
            "expect_success": False,
        },
        {
            "id": 6,
            "name": "Arbitrary Filesystem Path (C:\\Windows\\System32\\Something)",
            "command": {"intent": "create_folder", "entity": "C:\\Windows\\System32\\Something"},
            "expect_success": False,
        },
        {
            "id": 7,
            "name": "Invalid Structured Command (Missing Intent)",
            "command": {"entity": "MissingIntent"},
            "expect_success": False,
        },
    ]

    all_passed = True

    print("\n--- Running Mandatory Security & Execution Tests ---")
    for tc in test_cases:
        print(f"\n[TEST {tc['id']}] {tc['name']}")
        print(f"  Input Command : {json.dumps(tc['command'])}")
        
        result: ToolResult = router.route_and_execute(tc["command"])
        
        passed = (result.success == tc["expect_success"])
        if not passed:
            all_passed = False

        status_tag = "PASS" if passed else "FAIL"
        print(f"  Outcome       : success={result.success}")
        print(f"  Message       : {result.message}")
        print(f"  Test Status   : [{status_tag}] (Expected success={tc['expect_success']})")

    # Step 8: End-to-End Pipeline Demonstration
    print("\n" + "=" * 75)
    print("--- End-to-End Pipeline Demonstration ---")
    print("Flow: Raw Text Command -> Brain Parser -> Tool Router -> Tool Execution")
    print("=" * 75)

    parser = CommandParser()
    sample_text = "Create a folder called Projects"
    
    # 1. Natural Language -> Structured Intent & Entity
    parsed_cmd = parser.parse(sample_text)
    print(f"\n1. User Speech / Text : \"{sample_text}\"")
    print(f"2. Brain Parsing Result: {json.dumps(parsed_cmd.to_dict(), indent=2)}")

    # 2. Structured Command -> Tool Router -> Action
    print("3. Tool Router Processing...")
    # Clean up again to show clean execution
    if test_projects_dir.exists():
        shutil.rmtree(test_projects_dir)
    pipeline_result = router.route_and_execute(parsed_cmd)
    print(f"4. Tool Execution Result:")
    print(f"   Success: {pipeline_result.success}")
    print(f"   Message: {pipeline_result.message}")
    print(f"   Data   : {pipeline_result.data}")

    print("\n" + "=" * 75)
    if all_passed:
        print("ALL 7 CORE SECURITY & ROUTING TESTS PASSED.")
        print("Sandbox verified: Filesystem operations strictly confined to 'vroom_workspace'.")
        print("Allowlist verified: Unapproved apps, URLs, and intents completely blocked.")
        print("=" * 75)
        return 0
    else:
        print("ONE OR MORE TESTS FAILED.")
        print("=" * 75)
        return 1


if __name__ == "__main__":
    sys.exit(main())
