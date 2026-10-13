"""Automated Verification Test Suite for VROOM AI Observe -> Act -> Observe -> Verify Engine.

Tests all 12 required capabilities:
1. An action that succeeds and is independently verified.
2. An action that executes but fails verification.
3. An action whose outcome is unknown (e.g. browser URL dispatch).
4. An application launch with insufficient evidence.
5. File creation and existence verification.
6. A failed step preventing unsafe dependent steps.
7. A safe recovery attempt (transient glitch recovered on retry 1).
8. A potentially destructive action not being blindly retried.
9. Accurate reporting of partial completion.
10. Invalid observation requests being rejected.
11. Existing multi-step planning tests still passing.
12. Existing activation and continuous conversation behavior remaining intact.
"""

from pathlib import Path
import shutil
import sys
import tempfile
import time
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import MagicMock, patch

# Ensure UTF-8 console output encoding on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass

from app.brain.task_planner import (
    PlanExecutionReport,
    PlanStep,
    StepStatus,
    TaskPlan,
    TaskPlanner,
)
from app.brain.task_verifier import (
    StateObserver,
    TaskVerifier,
    VerificationResult,
    VerificationStatus,
)
from app.controller import ActivationController, AssistantState
from app.orchestrator import AssistantOrchestrator
from app.tools.base import BaseTool, ToolResult
from app.tools.filesystem import CreateFileTool, CreateFolderTool
from app.tools.registry import ToolRegistry
from app.tools.router import ToolRouter
import planner_test


class MockVerificationTool(BaseTool):
    """Mock tool providing customizable outcomes and metadata for verification testing."""

    def __init__(
        self,
        name: str,
        success: bool = True,
        message: str = "Execution success",
        data: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.name = name
        self.description = f"Mock verification tool for {name}"
        self.success = success
        self.message = message
        self.data = data or {}
        self.call_count = 0

    def execute(self, entity: Optional[str]) -> ToolResult:
        self.call_count += 1
        return ToolResult(
            success=self.success,
            message=self.message,
            data=dict(self.data, entity=entity),
        )


def test_1_action_succeeds_and_independently_verified() -> bool:
    print("\n--- TEST 1: Action Succeeds and is Independently Verified ---")
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        observer = StateObserver(workspace_root=tmp_path)
        verifier = TaskVerifier(observer=observer)

        registry = ToolRegistry(register_defaults=False)
        folder_tool = CreateFolderTool(workspace_root=tmp_path)
        registry.register_tool(folder_tool)
        router = ToolRouter(registry)

        planner = TaskPlanner(router=router, registry=registry, verifier=verifier, observer=observer, dev_mode=False)

        plan = TaskPlan(
            user_request="Create a folder named VerifiedFolder",
            steps=[PlanStep(1, "create_folder", "VerifiedFolder", explanation="Create folder")],
        )

        report = planner.execute_plan(plan)
        step = report.completed_steps[0]

        print(f"Step Status: {step.status.value}")
        print(f"Verification Status: {step.verification.status.value if step.verification else None}")
        print(f"Verification Evidence: {step.verification.actual_evidence if step.verification else None}")

        assert report.success
        assert step.status == StepStatus.COMPLETED
        assert step.verification is not None
        assert step.verification.status == VerificationStatus.VERIFIED
        assert (tmp_path / "VerifiedFolder").is_dir()
        print("[PASS] Test 1: Action executed and verified independently by disk inspection.")
        return True


def test_2_action_executes_but_fails_verification() -> bool:
    print("\n--- TEST 2: Action Executes but Fails Verification ---")
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        observer = StateObserver(workspace_root=tmp_path)
        verifier = TaskVerifier(observer=observer)

        registry = ToolRegistry(register_defaults=False)
        # Mock application tool that returns success=True with executable, but process is NOT running
        fake_app_tool = MockVerificationTool(
            name="open_application",
            success=True,
            message="Launched application",
            data={"executable": "notepad.exe"},
        )
        registry.register_tool(fake_app_tool)
        router = ToolRouter(registry)

        # Force observer.is_process_running to return False (simulating process crash immediately after launch)
        with patch.object(observer, "is_process_running", return_value=(False, {"process_found": False})):
            # Set safe_recovery_available to False for this test so we directly verify FAILED status without retry loop
            with patch.object(
                verifier,
                "verify",
                return_value=VerificationResult(
                    status=VerificationStatus.FAILED,
                    expected_outcome="Notepad running",
                    actual_evidence={"process_found": False},
                    explanation="Process not detected in process table despite tool reporting success.",
                    safe_recovery_available=False,
                ),
            ):
                planner = TaskPlanner(router=router, registry=registry, verifier=verifier, observer=observer, dev_mode=False)
                plan = TaskPlan(
                    user_request="Open Notepad",
                    steps=[PlanStep(1, "open_application", "Notepad")],
                )

                report = planner.execute_plan(plan)
                assert not report.success
                assert len(report.failed_steps) == 1
                failed_step = report.failed_steps[0]
                assert failed_step.verification.status == VerificationStatus.FAILED
                print(f"Failed Step Verification: {failed_step.verification.status.value} - {failed_step.verification.explanation}")

    print("[PASS] Test 2: Successful tool invocation correctly marked FAILED when verification evidence was absent.")
    return True


def test_3_action_outcome_is_unknown() -> bool:
    print("\n--- TEST 3: Action Whose Outcome is Unknown (Browser URL Dispatch) ---")
    observer = StateObserver()
    verifier = TaskVerifier(observer=observer)
    registry = ToolRegistry(register_defaults=False)

    fake_browser_tool = MockVerificationTool(
        name="open_website",
        success=True,
        message="Website opened in browser",
        data={"url": "https://www.youtube.com"},
    )
    registry.register_tool(fake_browser_tool)
    router = ToolRouter(registry)

    planner = TaskPlanner(router=router, registry=registry, verifier=verifier, observer=observer, dev_mode=False)
    plan = TaskPlan(
        user_request="Open YouTube",
        steps=[PlanStep(1, "open_website", "YouTube")],
    )

    report = planner.execute_plan(plan)
    assert report.success
    step = report.completed_steps[0]
    assert step.verification is not None
    assert step.verification.status == VerificationStatus.UNKNOWN
    print(f"Verification Status: {step.verification.status.value}")
    print(f"Verification Explanation: {step.verification.explanation}")
    assert "page render" in step.verification.explanation.lower() or "instrumentation" in step.verification.explanation.lower()
    print("[PASS] Test 3: Browser URL dispatch recognized as UNKNOWN without fabricating web page render success.")
    return True


def test_4_app_launch_with_insufficient_evidence() -> bool:
    print("\n--- TEST 4: Application Launch With Insufficient Evidence ---")
    observer = StateObserver()
    verifier = TaskVerifier(observer=observer)

    # Tool returned success, but process table has no matching PID
    tool_res = ToolResult(success=True, message="Process spawned", data={"executable": "calc.exe", "entity": "Calculator"})
    pre_st = {"process_running": False}
    post_st = {"process_running": False, "process_evidence": {"process_found": False, "pids": []}}

    res = verifier.verify("open_application", "Calculator", tool_res, pre_st, post_st)
    print(f"Verification status: {res.status.value}")
    print(f"Safe recovery available: {res.safe_recovery_available}")
    assert res.status == VerificationStatus.FAILED
    assert res.safe_recovery_available is True
    assert "not detected in os process table" in res.explanation.lower()
    print("[PASS] Test 4: Insufficient process evidence marked FAILED and flagged for safe recovery.")
    return True


def test_5_file_creation_and_existence_verification() -> bool:
    print("\n--- TEST 5: File Creation and Existence Verification ---")
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        observer = StateObserver(workspace_root=tmp_path)
        verifier = TaskVerifier(observer=observer)

        registry = ToolRegistry(register_defaults=False)
        file_tool = CreateFileTool(workspace_root=tmp_path)
        registry.register_tool(file_tool)
        router = ToolRouter(registry)

        planner = TaskPlanner(router=router, registry=registry, verifier=verifier, observer=observer, dev_mode=False)

        plan = TaskPlan(
            user_request="Create text file notes.txt",
            steps=[PlanStep(1, "create_file", "notes.txt", explanation="Create text file")],
        )

        report = planner.execute_plan(plan)
        assert report.success
        step = report.completed_steps[0]
        assert step.verification.status == VerificationStatus.VERIFIED
        assert (tmp_path / "notes.txt").is_file()
        print(f"File verified on disk: Path={(tmp_path / 'notes.txt')}, Size={step.verification.actual_evidence.get('size_bytes')} bytes")
    print("[PASS] Test 5: File created and independently verified on disk.")
    return True


def test_6_failed_step_prevents_unsafe_dependent_steps() -> bool:
    print("\n--- TEST 6: Failed Step Prevents Unsafe Dependent Steps ---")
    registry = ToolRegistry(register_defaults=False)
    tool_step1 = MockVerificationTool("open_application", success=True, message="App 1 OK")
    tool_step2 = MockVerificationTool("create_folder", success=False, message="Simulated Disk I/O Error")
    tool_step3 = MockVerificationTool("open_website", success=True, message="Website 3")

    registry.register_tool(tool_step1)
    registry.register_tool(tool_step2)
    registry.register_tool(tool_step3)
    router = ToolRouter(registry)

    planner = TaskPlanner(router=router, registry=registry, dev_mode=False)

    plan = TaskPlan(
        user_request="Step 1, Step 2, Step 3",
        steps=[
            PlanStep(1, "open_application", "Chrome"),
            PlanStep(2, "create_folder", "SecretFolder"),
            PlanStep(3, "open_website", "YouTube"),
        ],
    )

    report = planner.execute_plan(plan)
    assert not report.success
    assert len(report.completed_steps) == 1
    assert len(report.failed_steps) == 1
    assert len(report.skipped_steps) == 1
    assert report.skipped_steps[0].step_number == 3
    assert report.skipped_steps[0].status == StepStatus.SKIPPED
    assert tool_step3.call_count == 0, "Dependent Step 3 must never be attempted after Step 2 failure"
    print(f"Completed: {len(report.completed_steps)}, Failed: {len(report.failed_steps)}, Skipped: {len(report.skipped_steps)}")
    print("[PASS] Test 6: Step 2 failure safely prevented execution of dependent Step 3.")
    return True


def test_7_safe_recovery_attempt() -> bool:
    print("\n--- TEST 7: Safe Recovery Attempt (Transient Failure Recovered on Retry) ---")
    registry = ToolRegistry(register_defaults=False)

    # Flaky tool that fails on call 1 but succeeds on call 2
    class FlakyAppTool(BaseTool):
        name = "open_application"
        description = "Flaky app tool"

        def __init__(self):
            self.attempts = 0

        def execute(self, entity: Optional[str]) -> ToolResult:
            self.attempts += 1
            if self.attempts == 1:
                return ToolResult(success=False, message="Transient spawn timeout", data={"entity": entity})
            return ToolResult(success=True, message="Successfully launched on retry", data={"entity": entity})

    flaky_tool = FlakyAppTool()
    registry.register_tool(flaky_tool)
    router = ToolRouter(registry)

    planner = TaskPlanner(router=router, registry=registry, dev_mode=False)

    plan = TaskPlan(
        user_request="Open Notepad",
        steps=[PlanStep(1, "open_application", "Notepad")],
    )

    report = planner.execute_plan(plan)
    assert report.success
    step = report.completed_steps[0]
    print(f"Step status: {step.status.value}")
    print(f"Recovery attempted: {step.recovery_attempted}")
    print(f"Recovery result: {step.recovery_result}")
    assert step.recovery_attempted is True
    assert "Recovered successfully on retry 1" in (step.recovery_result or "")
    assert flaky_tool.attempts == 2
    print("[PASS] Test 7: Transient failure successfully recovered on controlled retry 1.")
    return True


def test_8_destructive_action_not_blindly_retried() -> bool:
    print("\n--- TEST 8: Potentially Destructive / Policy Rejection Not Blindly Retried ---")
    registry = ToolRegistry(register_defaults=False)

    class DestructiveTool(BaseTool):
        name = "delete_files"
        description = "Destructive tool"

        def __init__(self):
            self.call_count = 0

        def execute(self, entity: Optional[str]) -> ToolResult:
            self.call_count += 1
            return ToolResult(success=False, message="Rejected: Destructive action disallowed")

    dest_tool = DestructiveTool()
    registry.register_tool(dest_tool)
    router = ToolRouter(registry)

    planner = TaskPlanner(router=router, registry=registry, dev_mode=False)
    plan = TaskPlan(
        user_request="Delete files",
        steps=[PlanStep(1, "delete_files", "system_files")],
    )

    report = planner.execute_plan(plan)
    assert not report.success
    step = report.failed_steps[0]
    assert step.recovery_attempted is False, "Destructive action must NEVER be retried!"
    assert dest_tool.call_count == 1, "Destructive action must be executed at most once and NOT retried"
    print(f"Recovery attempted: {step.recovery_attempted}, Execution call count: {dest_tool.call_count}")
    print("[PASS] Test 8: Destructive/unapproved actions are never blindly retried.")
    return True


def test_9_accurate_reporting_of_partial_completion() -> bool:
    print("\n--- TEST 9: Accurate Reporting of Partial Completion ---")
    registry = ToolRegistry(register_defaults=False)
    app_tool = MockVerificationTool("open_application", success=True, message="Application opened")
    registry.register_tool(app_tool)
    router = ToolRouter(registry)

    planner = TaskPlanner(router=router, registry=registry, dev_mode=False)

    plan = TaskPlan(
        user_request="Open Notepad and create a text file named project_notes.txt",
        steps=[
            PlanStep(1, "open_application", "Notepad", explanation="Open Notepad editor"),
            PlanStep(2, "create_file", "project_notes.txt", explanation="Create text file"),
        ],
    )

    report = planner.execute_plan(plan)
    assert not report.success
    assert len(report.completed_steps) == 1
    assert len(report.failed_steps) == 1
    print(f"Summary message: '{report.summary_message}'")
    assert "Completed Notepad is now open" in report.summary_message
    assert "is not supported" in report.summary_message
    print("[PASS] Test 9: Partial completion reported accurately without claiming full success.")
    return True


def test_10_invalid_observation_requests_rejected() -> bool:
    print("\n--- TEST 10: Invalid Observation Requests Being Rejected ---")
    observer = StateObserver()

    # 1. Unapproved process query
    ok1, ev1 = observer.is_process_running("malicious_rootkit.exe")
    assert not ok1
    assert ev1.get("rejected") is True
    print(f"Unapproved process observation rejected: {ev1.get('error')}")

    # 2. Path traversal file check
    ok2, ev2 = observer.file_exists("../../windows/system32/cmd.exe")
    assert not ok2
    assert ev2.get("rejected") is True
    print(f"Path traversal file check rejected: {ev2.get('error')}")

    # 3. Illegal Windows characters in directory query
    ok3, ev3 = observer.directory_exists("invalid*dir:name?")
    assert not ok3
    assert ev3.get("rejected") is True
    print(f"Illegal chars directory query rejected: {ev3.get('error')}")

    print("[PASS] Test 10: All invalid observation requests safely rejected.")
    return True


def test_11_existing_multistep_planning_tests() -> bool:
    print("\n--- TEST 11: Regression Check: Existing Multi-Step Planning Suite ---")
    success = planner_test.run_all_tests()
    assert success, "All 10 multi-step planning tests must pass!"
    print("[PASS] Test 11: All 10 existing multi-step planning tests passed.")
    return True


def test_12_existing_activation_and_conversation_intact() -> bool:
    print("\n--- TEST 12: Regression Check: Activation & Continuous Conversation ---")
    transitions: List[Tuple[str, str]] = []
    controller = ActivationController(
        mode="staged",
        dev_mode=False,
        on_state_change=lambda o, n: transitions.append((o.value, n.value)),
    )

    assert controller.state == AssistantState.WAITING_FOR_CLAP

    controller._set_state(AssistantState.ACTIVATED)
    assert controller.state == AssistantState.ACTIVATED

    controller._set_state(AssistantState.CONVERSATION_SESSION)
    assert controller.state == AssistantState.CONVERSATION_SESSION

    controller._set_state(AssistantState.WAITING_FOR_CLAP)
    assert controller.state == AssistantState.WAITING_FOR_CLAP

    print(f"Lifecycle transitions verified: {len(transitions)}")
    print("[PASS] Test 12: Activation and continuous conversation state machine preserved.")
    return True


def run_all_tests() -> bool:
    print("=" * 75)
    print("      VROOM AI — OBSERVE -> ACT -> VERIFY Engine Verification Suite")
    print("=" * 75)

    tests = [
        ("Test 1: Succeeded & Independently Verified", test_1_action_succeeds_and_independently_verified),
        ("Test 2: Executes but Fails Verification", test_2_action_executes_but_fails_verification),
        ("Test 3: Outcome is Unknown (Browser)", test_3_action_outcome_is_unknown),
        ("Test 4: Launch With Insufficient Evidence", test_4_app_launch_with_insufficient_evidence),
        ("Test 5: File Creation & Existence Verification", test_5_file_creation_and_existence_verification),
        ("Test 6: Failed Step Prevents Dependent Steps", test_6_failed_step_prevents_unsafe_dependent_steps),
        ("Test 7: Safe Recovery Attempt", test_7_safe_recovery_attempt),
        ("Test 8: Destructive Action Not Retried", test_8_destructive_action_not_blindly_retried),
        ("Test 9: Partial Completion Accurate Reporting", test_9_accurate_reporting_of_partial_completion),
        ("Test 10: Invalid Observation Requests Rejected", test_10_invalid_observation_requests_rejected),
        ("Test 11: Regression: Multi-Step Planning Suite", test_11_existing_multistep_planning_tests),
        ("Test 12: Regression: Activation & Conversation State", test_12_existing_activation_and_conversation_intact),
    ]

    passed = 0
    failed = 0

    for name, test_fn in tests:
        try:
            success = test_fn()
            if success:
                passed += 1
            else:
                failed += 1
                print(f"[FAIL] {name} returned False")
        except Exception as exc:
            failed += 1
            print(f"[ERROR] {name} raised exception: {exc}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 75)
    print(f"Test Results: {passed} PASSED, {failed} FAILED (Total: {len(tests)})")
    print("=" * 75)

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
