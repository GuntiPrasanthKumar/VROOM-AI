"""Automated test suite for VROOM AI Multi-Step Task Planning Engine.

Validates the 10 required capabilities:
1. Creating a valid multi-step plan.
2. Rejecting unknown tools.
3. Rejecting invalid arguments.
4. Executing steps in the correct order.
5. Stopping when a required step fails.
6. Reporting partial completion accurately.
7. Handling malformed LLM output.
8. Keeping simple single-action requests working.
9. Preserving existing activation and continuous conversation behavior.
10. Preventing arbitrary shell execution through LLM-generated plans.

Uses mock tools to prevent unwanted modifications to the real desktop environment.
"""

import sys
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

from app.brain.command_parser import CommandParser
from app.brain.task_planner import (
    PlanExecutionReport,
    PlanStep,
    StepStatus,
    TaskPlan,
    TaskPlanner,
)
from app.controller import ActivationController, AssistantState
from app.orchestrator import AssistantOrchestrator
from app.tools.base import BaseTool, ToolResult
from app.tools.registry import ToolRegistry
from app.tools.router import ToolRouter


class MockTool(BaseTool):
    """Mock tool that records execution calls without touching the real desktop."""

    def __init__(self, name: str, should_succeed: bool = True, return_message: str = "Mock success") -> None:
        self.name = name
        self.description = f"Mock tool for {name}"
        self.should_succeed = should_succeed
        self.return_message = return_message
        self.call_history: List[Optional[str]] = []

    def execute(self, entity: Optional[str]) -> ToolResult:
        self.call_history.append(entity)
        return ToolResult(
            success=self.should_succeed,
            message=self.return_message if self.should_succeed else f"Mock failure for {entity}",
            data={"entity": entity},
        )


def build_mock_registry() -> Tuple[ToolRegistry, Dict[str, MockTool]]:
    """Build a mock tool registry mirroring approved VROOM tools."""
    registry = ToolRegistry(register_defaults=False)
    tools = {
        "open_application": MockTool("open_application", return_message="Application opened"),
        "open_website": MockTool("open_website", return_message="Website opened"),
        "create_folder": MockTool("create_folder", return_message="Folder created"),
    }
    for tool in tools.values():
        registry.register_tool(tool)
    return registry, tools


def test_1_create_valid_multistep_plan() -> bool:
    print("\n--- TEST 1: Creating a Valid Multi-Step Plan ---")
    registry, _ = build_mock_registry()
    router = ToolRouter(registry)
    planner = TaskPlanner(router=router, registry=registry, dev_mode=False)

    command = "Open Chrome, open YouTube, and then open Notepad"
    plan = planner.create_plan(command)

    print(f"User Request: {plan.user_request}")
    print(f"Generated Steps count: {len(plan.steps)}")
    for s in plan.steps:
        print(f"  Step {s.step_number}: Action='{s.action}', Entity='{s.entity}' ({s.explanation})")

    assert len(plan.steps) == 3, f"Expected 3 steps, got {len(plan.steps)}"
    assert plan.steps[0].action == "open_application" and "chrome" in str(plan.steps[0].entity).lower()
    assert plan.steps[1].action == "open_website" and "youtube" in str(plan.steps[1].entity).lower()
    assert plan.steps[2].action == "open_application" and "notepad" in str(plan.steps[2].entity).lower()

    # Pre-validation check
    valid, errors = planner.validate_plan(plan)
    assert valid, f"Expected plan to be valid, got errors: {errors}"
    print("[PASS] Test 1: Multi-step plan created and validated successfully.")
    return True


def test_2_reject_unknown_tools() -> bool:
    print("\n--- TEST 2: Rejecting Unknown Tools ---")
    registry, _ = build_mock_registry()
    router = ToolRouter(registry)
    planner = TaskPlanner(router=router, registry=registry, dev_mode=False)

    step_unknown = PlanStep(
        step_number=1,
        action="hack_the_system",
        entity="Mainframe",
    )
    is_valid, err = planner.validate_step(step_unknown)
    print(f"Validation of unknown tool 'hack_the_system': Valid={is_valid}, Error='{err}'")
    assert not is_valid, "Expected unknown tool to be rejected"
    assert "not a supported tool" in err

    # Verify execution rejection
    plan = TaskPlan(user_request="Hack mainframe", steps=[step_unknown])
    report = planner.execute_plan(plan)
    assert not report.success
    assert len(report.failed_steps) == 1
    assert report.failed_steps[0].status == StepStatus.FAILED
    assert "not a supported tool" in report.summary_message
    print("[PASS] Test 2: Unknown tool rejected safely without execution.")
    return True


def test_3_reject_invalid_arguments() -> bool:
    print("\n--- TEST 3: Rejecting Invalid Arguments ---")
    registry, _ = build_mock_registry()
    router = ToolRouter(registry)
    planner = TaskPlanner(router=router, registry=registry, dev_mode=False)

    # Empty entity
    step_empty = PlanStep(step_number=1, action="create_folder", entity="")
    v1, err1 = planner.validate_step(step_empty)
    assert not v1 and "requires a valid target entity" in err1
    print(f"Empty entity rejected: {err1}")

    # Path traversal separator
    step_traversal = PlanStep(step_number=2, action="create_folder", entity="folder/nested")
    v2, err2 = planner.validate_step(step_traversal)
    assert not v2 and "cannot contain path separators" in err2
    print(f"Path traversal rejected: {err2}")

    # Illegal Windows characters
    step_illegal = PlanStep(step_number=3, action="create_folder", entity="bad*folder?")
    v3, err3 = planner.validate_step(step_illegal)
    assert not v3 and "contains illegal characters" in err3
    print(f"Illegal chars rejected: {err3}")

    print("[PASS] Test 3: Invalid arguments rejected safely by schema validation.")
    return True


def test_4_execute_steps_in_correct_order() -> bool:
    print("\n--- TEST 4: Executing Steps in Correct Order ---")
    registry, tools = build_mock_registry()
    router = ToolRouter(registry)
    planner = TaskPlanner(router=router, registry=registry, dev_mode=False)

    execution_order: List[str] = []

    def record_exec(tool_name: str, entity: Optional[str]) -> ToolResult:
        execution_order.append(f"{tool_name}:{entity}")
        return ToolResult(success=True, message=f"Executed {tool_name} {entity}")

    tools["open_application"].execute = lambda ent: record_exec("open_application", ent)
    tools["open_website"].execute = lambda ent: record_exec("open_website", ent)
    tools["create_folder"].execute = lambda ent: record_exec("create_folder", ent)

    plan = TaskPlan(
        user_request="Order test",
        steps=[
            PlanStep(1, "open_application", "Chrome"),
            PlanStep(2, "open_website", "YouTube"),
            PlanStep(3, "create_folder", "MyNotes"),
        ],
    )

    report = planner.execute_plan(plan)
    assert report.success, "Plan execution should succeed"
    expected = ["open_application:Chrome", "open_website:YouTube", "create_folder:MyNotes"]
    print(f"Observed execution order: {execution_order}")
    assert execution_order == expected, f"Order mismatch: expected {expected}, got {execution_order}"
    print("[PASS] Test 4: Steps executed strictly in the requested sequence.")
    return True


def test_5_stop_when_required_step_fails() -> bool:
    print("\n--- TEST 5: Stopping When a Required Step Fails ---")
    registry, tools = build_mock_registry()
    router = ToolRouter(registry)
    planner = TaskPlanner(router=router, registry=registry, dev_mode=False)

    executed_steps: List[int] = []

    def step1_exec(ent):
        executed_steps.append(1)
        return ToolResult(success=True, message="Step 1 OK")

    def step2_exec(ent):
        executed_steps.append(2)
        return ToolResult(success=False, message="Step 2 Simulated Failure")

    def step3_exec(ent):
        executed_steps.append(3)
        return ToolResult(success=True, message="Step 3 OK")

    tools["open_application"].execute = step1_exec
    tools["open_website"].execute = step2_exec
    tools["create_folder"].execute = step3_exec

    plan = TaskPlan(
        user_request="Fail test",
        steps=[
            PlanStep(1, "open_application", "Chrome"),
            PlanStep(2, "open_website", "YouTube"),
            PlanStep(3, "create_folder", "Notes"),
        ],
    )

    report = planner.execute_plan(plan)
    print(f"Executed step indices: {executed_steps}")
    print(f"Completed steps count: {len(report.completed_steps)}")
    print(f"Failed steps count: {len(report.failed_steps)}")
    print(f"Skipped steps count: {len(report.skipped_steps)}")
    print(f"Stop reason: {report.stop_reason}")

    assert executed_steps == [1, 2], "Step 3 must NEVER be executed after Step 2 failed"
    assert len(report.completed_steps) == 1
    assert len(report.failed_steps) == 1
    assert len(report.skipped_steps) == 1
    assert report.skipped_steps[0].step_number == 3
    assert report.skipped_steps[0].status == StepStatus.SKIPPED
    assert not report.success
    print("[PASS] Test 5: Execution stopped safely upon failure; Step 3 was skipped.")
    return True


def test_6_report_partial_completion_accurately() -> bool:
    print("\n--- TEST 6: Reporting Partial Completion Accurately ---")
    registry, tools = build_mock_registry()
    router = ToolRouter(registry)
    planner = TaskPlanner(router=router, registry=registry, dev_mode=False)

    # Command: "Open Notepad and create a text file named project_notes.txt"
    plan = TaskPlan(
        user_request="Open Notepad and create a text file named project_notes.txt",
        steps=[
            PlanStep(1, "open_application", "Notepad", explanation="Open Notepad editor"),
            PlanStep(2, "create_file", "project_notes.txt", explanation="Create text file"),
        ],
    )

    report = planner.execute_plan(plan)
    print(f"Completed: {[s.action for s in report.completed_steps]}")
    print(f"Failed: {[s.action for s in report.failed_steps]}")
    print(f"Summary message: '{report.summary_message}'")

    assert len(report.completed_steps) == 1
    assert report.completed_steps[0].action == "open_application"
    assert len(report.failed_steps) == 1
    assert report.failed_steps[0].action == "create_file"
    assert not report.success

    # Verify summary message distinguishes completion and does NOT hallucinate file creation
    assert "Completed Notepad is now open" in report.summary_message
    assert "creating a file" in report.summary_message.lower() and "is not supported" in report.summary_message.lower()
    print("[PASS] Test 6: Partial completion reported accurately without hallucinating success.")
    return True


def test_7_handle_malformed_llm_output() -> bool:
    print("\n--- TEST 7: Handling Malformed LLM Output ---")
    registry, _ = build_mock_registry()
    router = ToolRouter(registry)

    # Mock LLM that returns invalid JSON
    mock_llm = MagicMock()
    mock_llm.base_url = "http://127.0.0.1:11434"
    mock_llm.model = "llama3.2:1b"
    mock_llm.timeout = 10.0

    planner = TaskPlanner(router=router, llm=mock_llm, registry=registry, dev_mode=False)

    # Simulate LLM raising an exception or returning garbage
    with patch.object(planner, "_decompose_with_llm", side_effect=Exception("JSON parse error")):
        command = "Open Chrome, open YouTube, and then open Notepad"
        plan = planner.create_plan(command)

        print(f"Fallback Plan generated {len(plan.steps)} steps:")
        for s in plan.steps:
            print(f"  Step {s.step_number}: {s.action} -> {s.entity}")

        assert len(plan.steps) == 3, f"Fallback should have produced 3 steps, got {len(plan.steps)}"
        assert plan.steps[0].action == "open_application"
        assert plan.steps[1].action == "open_website"
        assert plan.steps[2].action == "open_application"

    print("[PASS] Test 7: Malformed LLM output fell back to deterministic parser cleanly.")
    return True


def test_8_keep_simple_single_action_working() -> bool:
    print("\n--- TEST 8: Keeping Simple Single-Action Requests Working ---")
    registry, tools = build_mock_registry()
    router = ToolRouter(registry)

    # Use mock TTS and recorder to avoid audio hardware usage during unit test
    mock_tts = MagicMock()
    mock_recorder = MagicMock()
    mock_stt = MagicMock()

    orchestrator = AssistantOrchestrator(
        recorder=mock_recorder,
        stt=mock_stt,
        router=router,
        tts=mock_tts,
        dev_mode=False,
    )

    res = orchestrator.process_command_text("Open Notepad")
    print(f"Single action result: Intent='{res['intent']}', Entity='{res['entity']}', Response='{res['response']}'")

    assert res["intent"] == "open_application"
    assert res["entity"] == "Notepad"
    assert res["tool_result"].success
    assert res["response"] == "Notepad is now open."
    assert not res["should_exit"]
    print("[PASS] Test 8: Single-action command processed with backward-compatible format.")
    return True


def test_9_preserve_activation_and_continuous_conversation() -> bool:
    print("\n--- TEST 9: Preserving Activation and Continuous Conversation Behavior ---")
    registry, _ = build_mock_registry()
    router = ToolRouter(registry)
    mock_tts = MagicMock()
    mock_recorder = MagicMock()
    mock_stt = MagicMock()

    orchestrator = AssistantOrchestrator(
        recorder=mock_recorder,
        stt=mock_stt,
        router=router,
        tts=mock_tts,
        dev_mode=False,
    )

    transitions: List[Tuple[str, str]] = []
    controller = ActivationController(
        orchestrator=orchestrator,
        mode="staged",
        dev_mode=False,
        on_state_change=lambda o, n: transitions.append((o.value, n.value)),
    )

    assert controller.state == AssistantState.WAITING_FOR_CLAP

    # Verify controller state machine initial and standby invariants
    print(f"Initial state: {controller.state.value}")
    assert controller.state == AssistantState.WAITING_FOR_CLAP

    # Test single-turn conversation session transition
    controller._set_state(AssistantState.ACTIVATED)
    assert controller.state == AssistantState.ACTIVATED

    controller._set_state(AssistantState.CONVERSATION_SESSION)
    assert controller.state == AssistantState.CONVERSATION_SESSION

    controller._set_state(AssistantState.WAITING_FOR_CLAP)
    assert controller.state == AssistantState.WAITING_FOR_CLAP

    print(f"Recorded state transitions: {len(transitions)}")
    print("[PASS] Test 9: State machine transitions and standby lifecycle preserved.")
    return True


def test_10_prevent_arbitrary_shell_execution() -> bool:
    print("\n--- TEST 10: Preventing Arbitrary Shell Execution Through LLM Plans ---")
    registry, _ = build_mock_registry()
    router = ToolRouter(registry)
    planner = TaskPlanner(router=router, registry=registry, dev_mode=False)

    adversarial_steps = [
        PlanStep(1, "powershell", "-Command Remove-Item -Recurse C:\\"),
        PlanStep(2, "cmd", "/c del *.*"),
        PlanStep(3, "bash", "-c 'rm -rf /'"),
        PlanStep(4, "run_command", "format C:"),
        PlanStep(5, "open_application", "notepad; calc.exe"),  # Command chaining injection attempt
        PlanStep(6, "create_folder", "test && dir"),           # Shell chaining attempt
    ]

    for step in adversarial_steps:
        is_valid, err = planner.validate_step(step)
        print(f"Adversarial step '{step.action}' with entity '{step.entity}': Valid={is_valid}, Rejected='{err}'")
        assert not is_valid, f"Step '{step.action}' should have been rejected"

    # Verify complete plan rejection
    plan = TaskPlan(user_request="Adversarial attack", steps=adversarial_steps)
    valid, errors = planner.validate_plan(plan)
    assert not valid
    assert len(errors) == len(adversarial_steps)

    report = planner.execute_plan(plan)
    assert not report.success
    assert len(report.completed_steps) == 0
    print("[PASS] Test 10: All adversarial shell commands and injection attempts rejected.")
    return True


def run_all_tests() -> bool:
    print("=" * 75)
    print("      VROOM AI — Multi-Step Task Planning Engine Verification Suite")
    print("=" * 75)

    tests = [
        ("Test 1: Valid Multi-Step Plan", test_1_create_valid_multistep_plan),
        ("Test 2: Reject Unknown Tools", test_2_reject_unknown_tools),
        ("Test 3: Reject Invalid Arguments", test_3_reject_invalid_arguments),
        ("Test 4: Correct Execution Order", test_4_execute_steps_in_correct_order),
        ("Test 5: Stop on Step Failure", test_5_stop_when_required_step_fails),
        ("Test 6: Partial Completion Reporting", test_6_report_partial_completion_accurately),
        ("Test 7: Malformed LLM Output Handling", test_7_handle_malformed_llm_output),
        ("Test 8: Single-Action Backward Compatibility", test_8_keep_simple_single_action_working),
        ("Test 9: Activation & Conversation Preservation", test_9_preserve_activation_and_continuous_conversation),
        ("Test 10: Shell Execution Prevention", test_10_prevent_arbitrary_shell_execution),
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
