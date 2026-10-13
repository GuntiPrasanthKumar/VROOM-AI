"""Multi-Step Task Planning Engine for VROOM AI.

Decomposes complex, multi-action natural language commands into structured,
sequential task plans. Validates each step against approved tool registries
and safety boundaries, executes steps sequentially, monitors progress, safely
halts on failures, and provides accurate outcome reporting.

Architecture:
    User Request
    -> LLM Interpretation (Ollama / Deterministic Fallback)
    -> Structured Task Plan (TaskPlan)
    -> Safety & Tool Schema Validation (validate_step / validate_plan)
    -> Sequential Dispatch via ToolRouter
    -> Tool Execution & Verification
    -> Execution Report (PlanExecutionReport)
    -> Final Response Generation for Piper TTS
"""

from dataclasses import dataclass, field
from enum import Enum
import json
import re
import time
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from app.brain.command_parser import CommandParser, ParsedCommand
from app.brain.llm_client import OllamaClient
from app.brain.task_verifier import (
    StateObserver,
    TaskVerifier,
    VerificationResult,
    VerificationStatus,
)
from app.tools.base import ToolResult
from app.tools.registry import ToolRegistry
from app.tools.router import ToolRouter


class StepStatus(str, Enum):
    """Lifecycle status of an individual plan step."""

    PENDING = "PENDING"          # Created, awaiting validation
    VALIDATED = "VALIDATED"      # Checked against registry and schema
    IN_PROGRESS = "IN_PROGRESS"  # Currently executing
    COMPLETED = "COMPLETED"      # Executed successfully
    FAILED = "FAILED"            # Execution or validation failed
    SKIPPED = "SKIPPED"          # Aborted because a prior step failed
    REJECTED = "REJECTED"        # Security or schema rejection


@dataclass
class PlanStep:
    """Represents a discrete executable step in a multi-step task plan.

    Attributes:
        step_number: 1-indexed sequential step identifier.
        action: Target tool intent (e.g. 'open_application', 'open_website').
        entity: Parameter or target object of the action (e.g. 'Chrome', 'YouTube').
        explanation: Short human-readable description of the intended action.
        status: Current execution status.
        result: ToolResult returned after execution (if attempted).
        error: Detailed error message if step failed or was rejected.
        verification: Independent outcome verification result.
        recovery_attempted: Whether a safe recovery was attempted.
        recovery_result: Description of recovery outcome.
    """

    step_number: int
    action: str
    entity: Optional[str] = None
    explanation: str = ""
    status: StepStatus = StepStatus.PENDING
    result: Optional[ToolResult] = None
    error: Optional[str] = None
    verification: Optional[VerificationResult] = None
    recovery_attempted: bool = False
    recovery_result: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize plan step to a standard dictionary."""
        return {
            "step_number": self.step_number,
            "action": self.action,
            "entity": self.entity,
            "explanation": self.explanation,
            "status": self.status.value,
            "result": self.result.to_dict() if self.result else None,
            "error": self.error,
            "verification": self.verification.to_dict() if self.verification else None,
            "recovery_attempted": self.recovery_attempted,
            "recovery_result": self.recovery_result,
        }


@dataclass
class TaskPlan:
    """Represents an ordered sequence of plan steps generated for a request.

    Attributes:
        user_request: Original user command text.
        steps: Ordered list of PlanStep instances.
        created_at: Unix timestamp of plan generation.
        is_single_action: True if the request consists of only one action.
    """

    user_request: str
    steps: List[PlanStep] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    is_single_action: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Serialize task plan to a standard dictionary."""
        return {
            "user_request": self.user_request,
            "steps": [s.to_dict() for s in self.steps],
            "created_at": self.created_at,
            "is_single_action": self.is_single_action,
        }


@dataclass
class PlanExecutionReport:
    """Comprehensive report summarizing plan execution outcome.

    Attributes:
        plan: The TaskPlan that was executed.
        success: True if all steps executed successfully.
        completed_steps: List of steps that succeeded.
        failed_steps: List of steps that failed.
        skipped_steps: List of steps that were skipped after a failure.
        stop_reason: Explanation why execution stopped if unsuccessful.
        summary_message: Conversational summary message suitable for TTS.
    """

    plan: TaskPlan
    success: bool = False
    completed_steps: List[PlanStep] = field(default_factory=list)
    failed_steps: List[PlanStep] = field(default_factory=list)
    skipped_steps: List[PlanStep] = field(default_factory=list)
    stop_reason: Optional[str] = None
    summary_message: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Serialize execution report to a standard dictionary."""
        return {
            "user_request": self.plan.user_request,
            "success": self.success,
            "completed_steps": [s.to_dict() for s in self.completed_steps],
            "failed_steps": [s.to_dict() for s in self.failed_steps],
            "skipped_steps": [s.to_dict() for s in self.skipped_steps],
            "stop_reason": self.stop_reason,
            "summary_message": self.summary_message,
        }


class TaskPlanner:
    """Multi-Step Task Planning Engine for VROOM AI.

    Decomposes commands, validates against safe tool allowlists, executes sequentially,
    and reports clear outcomes without hallucinating unsupported actions.
    """

    # Forbidden shell command keywords to block arbitrary code injection
    FORBIDDEN_SHELL_KEYWORDS: Set[str] = {
        "powershell",
        "cmd",
        "bash",
        "sh",
        "exec",
        "execute",
        "shell",
        "python",
        "eval",
        "system",
        "subprocess",
        "rmdir",
        "del",
        "format",
    }

    # Common multi-step conjunction delimiters for deterministic fallback splitting
    CONJUNCTION_PATTERN = re.compile(
        r"(?:,\s*(?:and\s+then|then|after\s+that|and|also)\s+)"
        r"|(?:\s+(?:and\s+then|then|after\s+that)\s+)"
        r"|(?:\s+and\s+(?:then\s+)?)"
        r"|(?:,\s+)",
        re.IGNORECASE,
    )

    def __init__(
        self,
        router: Optional[ToolRouter] = None,
        llm: Optional[OllamaClient] = None,
        parser: Optional[CommandParser] = None,
        registry: Optional[ToolRegistry] = None,
        verifier: Optional[TaskVerifier] = None,
        observer: Optional[StateObserver] = None,
        dev_mode: bool = True,
    ) -> None:
        """Initialize the task planner.

        Args:
            router: ToolRouter instance for executing approved actions.
            llm: OllamaClient for LLM-based natural language decomposition.
            parser: CommandParser for deterministic clause understanding.
            registry: ToolRegistry for schema and intent inspection.
            verifier: TaskVerifier for independent outcome validation.
            observer: StateObserver for read-only desktop observation.
            dev_mode: Whether to print diagnostic development output.
        """
        self.dev_mode = dev_mode
        self.registry = registry or ToolRegistry()
        self.router = router or ToolRouter(self.registry)
        self.llm = llm or OllamaClient()
        self.parser = parser or CommandParser()
        self.observer = observer or StateObserver(dev_mode=dev_mode)
        self.verifier = verifier or TaskVerifier(observer=self.observer, dev_mode=dev_mode)

    def is_multi_step_request(self, text: str) -> bool:
        """Determine if a text command contains compound multi-step actions.

        Args:
            text: Raw or transcribed natural language string.

        Returns:
            True if text contains multiple sequential action clauses.
        """
        if not text or not text.strip():
            return False

        clean = text.strip().lower()

        # Check for explicit sequential markers
        sequential_markers = (
            " and then ",
            ", and then ",
            " then ",
            ", then ",
            " after that ",
            ", after that ",
            ", and ",
            " and also ",
        )
        for marker in sequential_markers:
            if marker in clean:
                return True

        # Check for multiple action verbs like "open X and open Y" or "open X and create Y"
        action_verbs = ("open", "launch", "start", "create", "make")
        count_verbs = sum(len(re.findall(rf"\b{verb}\b", clean)) for verb in action_verbs)
        if count_verbs >= 2 and (" and " in clean or "," in clean):
            return True

        return False

    def create_plan(self, text: str) -> TaskPlan:
        """Convert a user command into an ordered TaskPlan.

        Uses local LLM inference with deterministic fallback to ensure reliable operation.

        Args:
            text: Raw natural language command string.

        Returns:
            A structured TaskPlan containing an ordered list of PlanStep items.
        """
        clean_text = text.strip() if text else ""
        if not clean_text:
            return TaskPlan(user_request="", steps=[], is_single_action=True)

        # Quick check: If clearly single-action, parse directly to minimize latency
        if not self.is_multi_step_request(clean_text):
            return self._create_single_step_plan(clean_text)

        # Multi-step command: Attempt LLM decomposition first
        try:
            plan = self._decompose_with_llm(clean_text)
            if plan and len(plan.steps) > 0:
                return plan
        except Exception as exc:
            if self.dev_mode:
                print(f"[PLANNER WARN] LLM decomposition failed ({exc}). Falling back to deterministic planner.")

        # Fallback: Deterministic regex clause decomposition
        return self._decompose_deterministic(clean_text)

    def _create_single_step_plan(self, text: str) -> TaskPlan:
        """Create a single-step TaskPlan for a simple command."""
        intent = "unknown"
        entity = None

        try:
            llm_res = self.llm.parse_command(text)
            intent = llm_res.get("intent", "unknown")
            entity = llm_res.get("entity")
        except Exception:
            parsed = self.parser.parse(text)
            intent = parsed.intent
            entity = parsed.entity

        step = PlanStep(
            step_number=1,
            action=intent,
            entity=entity,
            explanation=f"Execute {intent} for {entity or 'requested item'}",
            status=StepStatus.PENDING,
        )
        return TaskPlan(
            user_request=text,
            steps=[step],
            is_single_action=True,
        )

    def _decompose_with_llm(self, text: str) -> TaskPlan:
        """Use Ollama LLM to decompose a compound request into structured steps."""
        system_prompt = (
            "You are a task planning engine for the VROOM AI desktop assistant. "
            "Your job is to break down the user's multi-step request into an ordered sequence of discrete actions. "
            "DO NOT execute anything. DO NOT invent arbitrary shell or system commands. "
            "You MUST respond ONLY with a valid JSON object with a single key 'steps'. "
            "'steps' must be a JSON array of step objects, ordered chronologically. "
            "Each step object MUST contain:\n"
            "  - 'step_number': integer starting from 1\n"
            "  - 'action': string identifier of the intent (e.g. 'open_application', 'open_website', 'create_folder', 'create_file', 'unknown')\n"
            "  - 'entity': string name of target parameter (e.g. 'Chrome', 'YouTube', 'project_notes.txt'), or null\n"
            "  - 'explanation': brief 3-6 word summary of what this step does\n\n"
            "Approved tools in VROOM are: 'open_application', 'open_website', 'create_folder'. "
            "If an action is unsupported (e.g. creating a file, deleting files), output the most accurate descriptive action name (e.g. 'create_file', 'delete_files').\n\n"
            "Examples:\n"
            "Input: 'Open Chrome, open YouTube, and then open Notepad'\n"
            "Output: {\"steps\": [\n"
            "  {\"step_number\": 1, \"action\": \"open_application\", \"entity\": \"Chrome\", \"explanation\": \"Open Chrome browser\"},\n"
            "  {\"step_number\": 2, \"action\": \"open_website\", \"entity\": \"YouTube\", \"explanation\": \"Open YouTube website\"},\n"
            "  {\"step_number\": 3, \"action\": \"open_application\", \"entity\": \"Notepad\", \"explanation\": \"Open Notepad editor\"}\n"
            "]}\n\n"
            "Input: 'Open Notepad and create a text file named notes.txt'\n"
            "Output: {\"steps\": [\n"
            "  {\"step_number\": 1, \"action\": \"open_application\", \"entity\": \"Notepad\", \"explanation\": \"Open Notepad editor\"},\n"
            "  {\"step_number\": 2, \"action\": \"create_file\", \"entity\": \"notes.txt\", \"explanation\": \"Create text file\"}\n"
            "]}"
        )

        prompt = f"User Request: \"{text}\""

        # Use generate API with JSON formatting
        endpoint = f"{self.llm.base_url}/api/generate"
        payload = {
            "model": self.llm.model,
            "prompt": prompt,
            "system": system_prompt,
            "format": "json",
            "stream": False,
            "options": {"temperature": 0.0},
        }

        import urllib.request
        data_bytes = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            endpoint,
            data=data_bytes,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        with urllib.request.urlopen(req, timeout=self.llm.timeout) as resp:
            body = json.loads(resp.read().decode("utf-8"))
            raw_response = body.get("response", "").strip()

        data = json.loads(raw_response)
        raw_steps = data.get("steps", [])
        if not isinstance(raw_steps, list) or len(raw_steps) == 0:
            raise ValueError(f"LLM returned no steps in 'steps' list: {data}")

        steps: List[PlanStep] = []
        for idx, item in enumerate(raw_steps, start=1):
            if not isinstance(item, dict):
                continue
            act = str(item.get("action", "unknown")).strip().lower()
            ent = item.get("entity")
            if ent is not None:
                ent = str(ent).strip()
            exp = str(item.get("explanation", f"Step {idx}")).strip()

            steps.append(
                PlanStep(
                    step_number=idx,
                    action=act,
                    entity=ent,
                    explanation=exp,
                    status=StepStatus.PENDING,
                )
            )

        return TaskPlan(
            user_request=text,
            steps=steps,
            is_single_action=(len(steps) == 1),
        )

    def _decompose_deterministic(self, text: str) -> TaskPlan:
        """Decompose text into steps using deterministic regex splitting and parsing."""
        clauses = self.CONJUNCTION_PATTERN.split(text)
        clean_clauses = [c.strip().rstrip(".!?,;:") for c in clauses if c and c.strip()]

        if not clean_clauses:
            clean_clauses = [text.strip()]

        steps: List[PlanStep] = []
        for idx, clause in enumerate(clean_clauses, start=1):
            action, entity = self._parse_single_clause(clause)
            steps.append(
                PlanStep(
                    step_number=idx,
                    action=action,
                    entity=entity,
                    explanation=f"Execute {action} for {entity or 'item'}",
                    status=StepStatus.PENDING,
                )
            )

        return TaskPlan(
            user_request=text,
            steps=steps,
            is_single_action=(len(steps) == 1),
        )

    def _parse_single_clause(self, clause: str) -> Tuple[str, Optional[str]]:
        """Parse an individual sub-clause into an action and entity pair."""
        lower_clause = clause.lower().strip()

        # Check for file creation (unsupported tool)
        file_match = re.search(
            r"(?:create|make)\s+(?:a\s+)?(?:text\s+)?file(?:\s+(?:named|called))?\s+(.+)",
            lower_clause,
        )
        if file_match:
            return "create_file", file_match.group(1).strip()

        # Check for delete (unsupported / destructive)
        if lower_clause.startswith(("delete", "remove", "erase", "destroy")):
            return "delete_files", lower_clause

        # Check for folder creation
        folder_match = re.search(
            r"(?:create|make)\s+(?:a\s+)?folder(?:\s+(?:called|named))?\s+(.+)",
            lower_clause,
        )
        if folder_match:
            return "create_folder", folder_match.group(1).strip()

        # Check for open/launch
        open_match = re.search(r"^(?:open|launch|start)\s+(.+)$", lower_clause)
        if open_match:
            target = open_match.group(1).strip()
            if target in CommandParser.KNOWN_WEBSITES:
                return "open_website", target.capitalize()
            return "open_application", target.capitalize()

        # Fallback to general command parser
        parsed = self.parser.parse(clause)
        return parsed.intent, parsed.entity

    def validate_step(self, step: PlanStep) -> Tuple[bool, Optional[str]]:
        """Validate an individual PlanStep against security rules and tool schemas.

        Args:
            step: PlanStep to validate.

        Returns:
            Tuple of (is_valid, error_reason_if_invalid).
        """
        # Security Check 1: Deny arbitrary shell commands
        if not step.action or not step.action.strip():
            return False, "Action is empty or unspecified."

        clean_action = step.action.strip().lower()
        if clean_action in self.FORBIDDEN_SHELL_KEYWORDS:
            return False, f"Action '{clean_action}' is a forbidden shell keyword."

        # Security Check 2: Verify action exists in ToolRegistry
        tool = self.registry.get_tool(clean_action)
        if tool is None:
            approved = self.registry.list_approved_intents()
            return False, f"Action '{step.action}' is not a supported tool. Supported tools: {approved}."

        # Security Check 3: Validate entity schema
        if step.entity is None or not str(step.entity).strip():
            return False, f"Action '{step.action}' requires a valid target entity, but none was provided."

        clean_entity = str(step.entity).strip()

        # Security Check 4: Check for shell command injection characters in entity
        dangerous_shell_chars = (";", "&&", "||", "|", "`", "$", "\n", "\r")
        for char in dangerous_shell_chars:
            if char in clean_entity:
                return False, f"Entity contains disallowed shell character '{char}'."

        # Action-specific parameter validation
        if clean_action == "create_folder":
            # Disallow path separators (path traversal prevention)
            if "/" in clean_entity or "\\" in clean_entity:
                return False, f"Folder name '{clean_entity}' cannot contain path separators."
            # Disallow Windows reserved characters
            if re.search(r'[<>:"/\\|?*]', clean_entity):
                return False, f"Folder name '{clean_entity}' contains illegal characters."

        step.status = StepStatus.VALIDATED
        return True, None

    def validate_plan(self, plan: TaskPlan) -> Tuple[bool, List[str]]:
        """Validate all steps in a TaskPlan before execution.

        Args:
            plan: TaskPlan to validate.

        Returns:
            Tuple of (all_valid, list_of_error_messages).
        """
        errors: List[str] = []
        for step in plan.steps:
            valid, err = self.validate_step(step)
            if not valid and err:
                errors.append(f"Step {step.step_number}: {err}")

        return (len(errors) == 0), errors

    def execute_plan(self, plan: TaskPlan) -> PlanExecutionReport:
        """Sequentially execute a validated TaskPlan using OBSERVE -> ACT -> OBSERVE -> VERIFY -> RECOVER.

        Halts immediately on the first failure or unsupported step, marks remaining
        steps as SKIPPED, and generates an accurate outcome report.

        Args:
            plan: The TaskPlan to execute.

        Returns:
            PlanExecutionReport detailing completed, failed, and skipped steps.
        """
        report = PlanExecutionReport(plan=plan)

        if not plan.steps:
            report.success = True
            report.summary_message = "No actions to execute."
            return report

        for i, step in enumerate(plan.steps):
            step.status = StepStatus.IN_PROGRESS
            if self.dev_mode:
                print(f"\n[PLANNER] Executing Step {step.step_number}/{len(plan.steps)}: {step.action} -> {step.entity}")

            # 1. Step Validation (check tool registry, schema, shell safety)
            is_valid, validation_error = self.validate_step(step)
            if not is_valid:
                step.status = StepStatus.FAILED
                step.error = validation_error
                step.verification = VerificationResult(
                    status=VerificationStatus.FAILED,
                    expected_outcome=f"Validate action '{step.action}'",
                    actual_evidence={"validation_error": validation_error},
                    explanation=validation_error or "Validation failed",
                    safe_recovery_available=False,
                )
                report.failed_steps.append(step)
                report.stop_reason = f"Step {step.step_number} failed validation: {validation_error}"
                if self.dev_mode:
                    print(f"  [REJECTED] {report.stop_reason}")

                # Mark all subsequent steps as SKIPPED
                for remaining_step in plan.steps[i + 1 :]:
                    remaining_step.status = StepStatus.SKIPPED
                    remaining_step.error = "Aborted due to prior step failure."
                    report.skipped_steps.append(remaining_step)
                break

            # 2. Pre-Action Observation (Capture environment state before action)
            pre_state = self.observer.observe_pre_state(step.action, step.entity)

            # 3. Sequential Execution via ToolRouter
            step_cmd = {"intent": step.action, "entity": step.entity}
            try:
                tool_result = self.router.route_and_execute(step_cmd)
            except Exception as exc:
                tool_result = ToolResult(
                    success=False,
                    message=f"Execution error: {exc}",
                )

            step.result = tool_result

            # 4. Post-Action Observation (Capture environment state after action)
            post_state = self.observer.observe_post_state(step.action, step.entity)

            # 5. Independent Verification
            verification = self.verifier.verify(
                step.action,
                step.entity,
                tool_result,
                pre_state,
                post_state,
            )
            step.verification = verification

            # 6. Evaluate Outcome and Safe Recovery
            # Succeeded if tool succeeded AND verification is VERIFIED or UNKNOWN (e.g. browser URL dispatch)
            step_succeeded = tool_result.success and verification.status in (
                VerificationStatus.VERIFIED,
                VerificationStatus.UNKNOWN,
            )

            if step_succeeded:
                step.status = StepStatus.COMPLETED
                report.completed_steps.append(step)
                if self.dev_mode:
                    status_lbl = "VERIFIED" if verification.status == VerificationStatus.VERIFIED else "UNKNOWN"
                    print(f"  [{status_lbl}] {verification.explanation}")
            else:
                # Execution or verification failed — Evaluate controlled recovery
                if verification.safe_recovery_available and not step.recovery_attempted:
                    step.recovery_attempted = True
                    step.recovery_result = "Recovery attempt 1 in progress..."
                    if self.dev_mode:
                        print(f"  [RECOVERY] Attempting 1 safe recovery for step {step.step_number} ({step.action} -> {step.entity})...")

                    # Re-observe pre-state
                    pre_retry = self.observer.observe_pre_state(step.action, step.entity)
                    try:
                        retry_result = self.router.route_and_execute(step_cmd)
                    except Exception as exc:
                        retry_result = ToolResult(success=False, message=f"Recovery error: {exc}")
                    step.result = retry_result

                    # Re-observe post-state
                    post_retry = self.observer.observe_post_state(step.action, step.entity)
                    retry_verification = self.verifier.verify(
                        step.action,
                        step.entity,
                        retry_result,
                        pre_retry,
                        post_retry,
                    )
                    step.verification = retry_verification

                    if retry_result.success and retry_verification.status in (
                        VerificationStatus.VERIFIED,
                        VerificationStatus.UNKNOWN,
                    ):
                        step.recovery_result = "Recovered successfully on retry 1."
                        step.status = StepStatus.COMPLETED
                        report.completed_steps.append(step)
                        if self.dev_mode:
                            print(f"  [RECOVERY SUCCESS] Step {step.step_number} recovered and verified.")
                        continue
                    else:
                        step.recovery_result = f"Recovery attempt 1 failed: {retry_verification.explanation}"
                        step.status = StepStatus.FAILED
                        step.error = retry_verification.explanation or retry_result.message
                        report.failed_steps.append(step)
                        report.stop_reason = f"Step {step.step_number} failed after recovery attempt: {step.error}"
                        if self.dev_mode:
                            print(f"  [RECOVERY FAILED] {report.stop_reason}")
                else:
                    step.status = StepStatus.FAILED
                    step.error = verification.explanation or tool_result.message
                    report.failed_steps.append(step)
                    report.stop_reason = f"Step {step.step_number} failed verification: {step.error}"
                    if self.dev_mode:
                        print(f"  [VERIFICATION FAILED] {report.stop_reason}")

                # Mark all subsequent steps as SKIPPED
                for remaining_step in plan.steps[i + 1 :]:
                    remaining_step.status = StepStatus.SKIPPED
                    remaining_step.error = "Aborted due to prior step failure."
                    report.skipped_steps.append(remaining_step)
                break

        # Check overall success
        report.success = len(report.failed_steps) == 0 and len(report.completed_steps) == len(plan.steps)
        report.summary_message = self.format_report_message(report)
        return report

    def plan_and_execute(self, text: str) -> PlanExecutionReport:
        """Create a plan from text and execute it in a single entry point.

        Args:
            text: Raw user request string.

        Returns:
            PlanExecutionReport with final execution status and TTS summary.
        """
        plan = self.create_plan(text)
        return self.execute_plan(plan)

    def _format_step_outcome(self, step: PlanStep) -> str:
        """Generate a natural phrase describing an individual step's success."""
        entity = step.entity or "the requested item"
        if step.action == "open_application":
            return f"{entity} is now open"
        elif step.action == "open_website":
            return f"opened {entity}"
        elif step.action == "create_folder":
            return f"folder {entity} was created"
        elif step.result:
            return step.result.message
        return f"{step.action} succeeded"

    def format_report_message(self, report: PlanExecutionReport) -> str:
        """Format an accurate, concise natural language summary for Piper TTS.

        Distinguishes completed, failed, and skipped steps without hallucinating success.

        Args:
            report: PlanExecutionReport to summarize.

        Returns:
            Natural spoken response string.
        """
        total_steps = len(report.plan.steps)

        # Case 0: Empty plan
        if total_steps == 0:
            return "I didn't detect any actions to perform."

        # Case 1: Single-step command (preserves existing conversational feel)
        if total_steps == 1:
            step = report.plan.steps[0]
            if step.status == StepStatus.COMPLETED:
                entity = step.entity or "the requested item"
                if step.action == "open_application":
                    return f"{entity} is now open."
                elif step.action == "open_website":
                    return f"Opening {entity} now."
                elif step.action == "create_folder":
                    return f"Folder {entity} has been created in your workspace."
                elif step.result:
                    return step.result.message
                return "Action completed successfully."
            else:
                # Failure / Rejection
                err_msg = step.error or (step.result.message if step.result else "Action was rejected.")
                if "not in the approved allowlist" in err_msg:
                    return f"I cannot open {step.entity}. That application is not in the approved allowlist."
                if "not a supported tool" in err_msg or "does not map" in err_msg or "not supported" in err_msg:
                    return f"I cannot perform that action. {step.action} is not a supported tool."
                if step.verification and step.verification.status == VerificationStatus.FAILED:
                    return f"Action '{step.action}' failed verification: {step.verification.explanation}"
                return f"I was unable to complete {step.action}: {err_msg}"

        # Case 2: Multi-step command — All steps succeeded
        if report.success:
            step_phrases = [self._format_step_outcome(s) for s in report.completed_steps]
            if len(step_phrases) == 2:
                joined = f"{step_phrases[0]}, and {step_phrases[1]}."
            elif len(step_phrases) > 2:
                joined = ", ".join(step_phrases[:-1]) + f", and {step_phrases[-1]}."
            else:
                joined = step_phrases[0] + "."
            return f"Completed all {total_steps} actions: {joined}"

        # Case 3: Partial completion (at least 1 step completed, followed by failure)
        if len(report.completed_steps) > 0 and len(report.failed_steps) > 0:
            failed_step = report.failed_steps[0]
            completed_str = ", ".join(self._format_step_outcome(s) for s in report.completed_steps)

            # Check if failure was due to unsupported tool
            if "not a supported tool" in (failed_step.error or ""):
                ent_text = f" named {failed_step.entity}" if failed_step.entity else ""
                fail_explanation = f"creating a file{ent_text} is not supported" if "create_file" in failed_step.action else f"{failed_step.action} is not supported"
            else:
                fail_explanation = failed_step.error or "the action failed"

            skipped_count = len(report.skipped_steps)
            skipped_text = f" Skipping remaining {skipped_count} actions." if skipped_count > 0 else ""
            return f"Completed {completed_str}. However, {fail_explanation}. Stopping plan.{skipped_text}"

        # Case 4: First step failed immediately
        if len(report.failed_steps) > 0:
            failed_step = report.failed_steps[0]
            err = failed_step.error or (failed_step.result.message if failed_step.result else "action failed")

            if "not a supported tool" in err:
                fail_explanation = f"Action '{failed_step.action}' is not supported"
            elif "not in the approved allowlist" in err:
                fail_explanation = f"'{failed_step.entity}' is not in the approved allowlist"
            else:
                fail_explanation = err

            skipped_count = len(report.skipped_steps)
            skipped_text = f" {skipped_count} remaining steps were not attempted." if skipped_count > 0 else ""
            return f"Failed at step {failed_step.step_number}: {fail_explanation}. Stopping plan.{skipped_text}"

        return "Task plan could not be executed."
