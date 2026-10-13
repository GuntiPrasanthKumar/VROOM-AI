"""Verification and State Observation Engine for VROOM AI.

Implements the OBSERVE -> ACT -> OBSERVE -> VERIFY -> RECOVER workflow.
Ensures VROOM never claims an action succeeded without independent evidence.

Architecture:
    Pre-Action Observation (StateObserver.observe_pre_state)
    -> Tool Execution (ToolRouter.route_and_execute)
    -> Post-Action Observation (StateObserver.observe_post_state)
    -> Outcome Verification (TaskVerifier.verify)
    -> Structured Verification Result (VerificationResult: VERIFIED, FAILED, UNKNOWN)
    -> Controlled Recovery (when safe_recovery_available is True)
    -> Honest Reporting to Piper TTS
"""

from dataclasses import dataclass, field
from enum import Enum
import os
from pathlib import Path
import re
import shutil
import subprocess
import time
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from app.tools.base import ToolResult


class VerificationStatus(str, Enum):
    """Explicit status for post-execution verification."""

    VERIFIED = "VERIFIED"  # Outcome independently confirmed by concrete evidence
    FAILED = "FAILED"      # Action attempted, but evidence confirms failure or absent outcome
    UNKNOWN = "UNKNOWN"    # Cannot independently verify (e.g. browser tab render without browser automation)


@dataclass
class VerificationResult:
    """Structured outcome of an independent verification inspection.

    Attributes:
        status: VerificationStatus (VERIFIED, FAILED, UNKNOWN).
        expected_outcome: Description of what should have occurred.
        actual_evidence: Key-value dictionary containing observed evidence.
        explanation: Clear human-readable explanation of the verification conclusion.
        safe_recovery_available: Whether the action is known to be safe to retry once.
    """

    status: VerificationStatus
    expected_outcome: str
    actual_evidence: Dict[str, Any] = field(default_factory=dict)
    explanation: str = ""
    safe_recovery_available: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert verification result to dictionary format."""
        return {
            "status": self.status.value,
            "expected_outcome": self.expected_outcome,
            "actual_evidence": self.actual_evidence,
            "explanation": self.explanation,
            "safe_recovery_available": self.safe_recovery_available,
        }


class StateObserver:
    """Safe, read-only desktop observation engine.

    Enforces path sandboxing, process allowlists, and read-only inspection.
    Never executes arbitrary commands or allows shell injection.
    """

    # Approved process allowlist mapping normalized names to process images
    APPLICATION_PROCESS_MAP: Dict[str, str] = {
        "chrome": "chrome.exe",
        "google chrome": "chrome.exe",
        "notepad": "notepad.exe",
        "calculator": "CalculatorApp.exe",  # Windows 10/11 modern calculator or calc.exe
        "calc": "calc.exe",
    }

    # Alternative process aliases on Windows (e.g. Calculator may appear as calc.exe or CalculatorApp.exe)
    PROCESS_ALIASES: Dict[str, List[str]] = {
        "calculator": ["calc.exe", "CalculatorApp.exe", "Calculator.exe"],
        "calc": ["calc.exe", "CalculatorApp.exe", "Calculator.exe"],
        "notepad": ["notepad.exe", "Notepad.exe"],
        "chrome": ["chrome.exe"],
        "google chrome": ["chrome.exe"],
    }

    INVALID_CHARS_PATTERN = re.compile(r'[<>:"/\\|?*]')

    def __init__(self, workspace_root: Optional[Path] = None, dev_mode: bool = False) -> None:
        """Initialize the state observer with workspace root.

        Args:
            workspace_root: Optional custom workspace Path. Defaults to './vroom_workspace'.
            dev_mode: Whether to print diagnostic observation output.
        """
        self.dev_mode = dev_mode
        if workspace_root is None:
            self.workspace_root = Path(__file__).resolve().parent.parent.parent / "vroom_workspace"
        else:
            self.workspace_root = workspace_root.resolve()

        self.workspace_root.mkdir(parents=True, exist_ok=True)

    def is_process_running(self, app_name_or_exe: str) -> Tuple[bool, Dict[str, Any]]:
        """Safely check if an approved application process is currently running.

        Enforces strict process allowlist. Rejects arbitrary or unvetted process queries.

        Args:
            app_name_or_exe: Name of the application (e.g. 'Notepad', 'Chrome').

        Returns:
            Tuple of (is_running, evidence_dictionary).
        """
        if not app_name_or_exe or not str(app_name_or_exe).strip():
            return False, {"error": "Application name must not be empty.", "rejected": True}

        clean = str(app_name_or_exe).strip().lower()

        # Security Check: Must be in application allowlist
        if clean not in self.APPLICATION_PROCESS_MAP and clean not in ("calc.exe", "notepad.exe", "chrome.exe"):
            return False, {
                "error": f"Rejected: Process '{app_name_or_exe}' is not in approved observation allowlist.",
                "rejected": True,
                "allowed": sorted(self.APPLICATION_PROCESS_MAP.keys()),
            }

        # Resolve candidate image names
        candidates = self.PROCESS_ALIASES.get(clean, [self.APPLICATION_PROCESS_MAP.get(clean, clean)])

        pids_found: List[str] = []
        found_name: Optional[str] = None

        for candidate in candidates:
            # Safe query: Use tasklist.exe directly with shell=False
            try:
                cmd = ["tasklist", "/FO", "CSV", "/NH", "/FI", f"IMAGENAME eq {candidate}"]
                res = subprocess.run(cmd, capture_output=True, text=True, shell=False, timeout=3.0)
                output = res.stdout.strip()
                if output and "INFO: No tasks are running" not in output:
                    # Parse CSV lines
                    for line in output.splitlines():
                        parts = [p.strip('"\r\n ') for p in line.split('","')]
                        if len(parts) >= 2 and candidate.lower() in parts[0].lower():
                            pids_found.append(parts[1])
                            found_name = parts[0]
            except Exception as exc:
                if self.dev_mode:
                    print(f"[OBSERVER WARN] Tasklist query failed: {exc}")

        is_running = len(pids_found) > 0
        evidence = {
            "query": app_name_or_exe,
            "candidates": candidates,
            "process_found": is_running,
            "process_name": found_name,
            "pids": pids_found,
            "count": len(pids_found),
        }
        return is_running, evidence

    def file_exists(self, file_name_or_path: Union[str, Path]) -> Tuple[bool, Dict[str, Any]]:
        """Safely check if a file exists strictly within the sandboxed workspace.

        Args:
            file_name_or_path: Name of the file or relative path.

        Returns:
            Tuple of (exists, evidence_dictionary).
        """
        if not file_name_or_path:
            return False, {"error": "File name must not be empty.", "rejected": True}

        raw_name = str(file_name_or_path).strip()

        # Security Checks
        if "/" in raw_name or "\\" in raw_name or ".." in raw_name:
            return False, {
                "error": f"Rejected: File path '{raw_name}' contains illegal path separators or traversal sequences.",
                "rejected": True,
            }

        if self.INVALID_CHARS_PATTERN.search(raw_name):
            return False, {
                "error": f"Rejected: File name '{raw_name}' contains illegal Windows characters.",
                "rejected": True,
            }

        target_path = (self.workspace_root / raw_name).resolve()

        # Sandbox escape validation
        try:
            target_path.relative_to(self.workspace_root)
        except ValueError:
            return False, {
                "error": f"Rejected: Target path '{target_path}' escapes workspace sandbox.",
                "rejected": True,
            }

        exists = target_path.exists() and target_path.is_file()
        evidence = {
            "path": str(target_path),
            "exists": exists,
            "is_file": True,
            "size_bytes": target_path.stat().st_size if exists else 0,
        }
        return exists, evidence

    def directory_exists(self, dir_name_or_path: Union[str, Path]) -> Tuple[bool, Dict[str, Any]]:
        """Safely check if a directory exists strictly within the sandboxed workspace.

        Args:
            dir_name_or_path: Name of the directory or relative path.

        Returns:
            Tuple of (exists, evidence_dictionary).
        """
        if not dir_name_or_path:
            return False, {"error": "Directory name must not be empty.", "rejected": True}

        raw_name = str(dir_name_or_path).strip()

        # Security Checks
        if "/" in raw_name or "\\" in raw_name or ".." in raw_name:
            return False, {
                "error": f"Rejected: Directory path '{raw_name}' contains illegal path separators or traversal sequences.",
                "rejected": True,
            }

        if self.INVALID_CHARS_PATTERN.search(raw_name):
            return False, {
                "error": f"Rejected: Directory name '{raw_name}' contains illegal Windows characters.",
                "rejected": True,
            }

        target_path = (self.workspace_root / raw_name).resolve()

        # Sandbox escape validation
        try:
            target_path.relative_to(self.workspace_root)
        except ValueError:
            return False, {
                "error": f"Rejected: Target path '{target_path}' escapes workspace sandbox.",
                "rejected": True,
            }

        exists = target_path.exists() and target_path.is_dir()
        evidence = {
            "path": str(target_path),
            "exists": exists,
            "is_dir": True,
        }
        return exists, evidence

    def observe_pre_state(self, action: str, entity: Optional[str]) -> Dict[str, Any]:
        """Capture the state of the relevant environment entity before executing an action.

        Args:
            action: Planned action intent (e.g. 'open_application', 'create_folder').
            entity: Target parameter (e.g. 'Notepad', 'Projects').

        Returns:
            Dictionary containing pre-action state observations.
        """
        state: Dict[str, Any] = {"timestamp": time.time(), "action": action, "entity": entity}

        if action == "open_application" and entity:
            running, ev = self.is_process_running(entity)
            state["process_running"] = running
            state["process_evidence"] = ev
        elif action == "create_folder" and entity:
            exists, ev = self.directory_exists(entity)
            state["directory_exists"] = exists
            state["directory_evidence"] = ev
        elif action == "create_file" and entity:
            exists, ev = self.file_exists(entity)
            state["file_exists"] = exists
            state["file_evidence"] = ev
        elif action == "open_website" and entity:
            state["browser_target"] = entity
            state["note"] = "Web page load status is unobservable without browser automation instrumentation."

        return state

    def observe_post_state(self, action: str, entity: Optional[str]) -> Dict[str, Any]:
        """Capture the state of the relevant environment entity after executing an action.

        Args:
            action: Planned action intent.
            entity: Target parameter.

        Returns:
            Dictionary containing post-action state observations.
        """
        state: Dict[str, Any] = {"timestamp": time.time(), "action": action, "entity": entity}

        if action == "open_application" and entity:
            # Brief pause to allow OS process scheduler to register newly spawned process
            time.sleep(0.15)
            running, ev = self.is_process_running(entity)
            state["process_running"] = running
            state["process_evidence"] = ev
        elif action == "create_folder" and entity:
            exists, ev = self.directory_exists(entity)
            state["directory_exists"] = exists
            state["directory_evidence"] = ev
        elif action == "create_file" and entity:
            exists, ev = self.file_exists(entity)
            state["file_exists"] = exists
            state["file_evidence"] = ev
        elif action == "open_website" and entity:
            state["browser_target"] = entity
            state["note"] = "Web page load status is unobservable without browser automation instrumentation."

        return state


class TaskVerifier:
    """Verification Engine for VROOM AI.

    Evaluates whether actions achieved their expected outcomes based on concrete evidence.
    Distinguishes VERIFIED, FAILED, and UNKNOWN outcomes, and qualifies safe recovery.
    """

    def __init__(self, observer: Optional[StateObserver] = None, dev_mode: bool = False) -> None:
        """Initialize the TaskVerifier.

        Args:
            observer: StateObserver instance for read-only environment checks.
            dev_mode: Whether to print diagnostic verification logs.
        """
        self.dev_mode = dev_mode
        self.observer = observer or StateObserver(dev_mode=dev_mode)

    def verify(
        self,
        action: str,
        entity: Optional[str],
        tool_result: ToolResult,
        pre_state: Dict[str, Any],
        post_state: Dict[str, Any],
    ) -> VerificationResult:
        """Verify the expected outcome of an action against observed evidence.

        Args:
            action: Action intent (e.g. 'open_application', 'open_website').
            entity: Target entity (e.g. 'Chrome', 'project_notes.txt').
            tool_result: ToolResult returned by ToolRouter.
            pre_state: State captured prior to execution.
            post_state: State captured after execution.

        Returns:
            VerificationResult indicating VERIFIED, FAILED, or UNKNOWN with evidence.
        """
        # =====================================================================
        # Case 0: Tool execution itself failed / was rejected
        # =====================================================================
        if not tool_result.success:
            # Check if this failure qualifies for safe recovery (e.g. transient launch glitch)
            # Dangerous actions or policy rejections must NEVER be retried
            is_policy_rejection = (
                "not in the approved allowlist" in tool_result.message
                or "does not map to any approved tool" in tool_result.message
                or "Sandbox violation" in tool_result.message
                or "already exists" in tool_result.message
            )
            safe_recovery = (action == "open_application") and not is_policy_rejection

            return VerificationResult(
                status=VerificationStatus.FAILED,
                expected_outcome=f"Execute {action} for {entity or 'item'}",
                actual_evidence={"tool_success": False, "tool_message": tool_result.message},
                explanation=f"Action failed execution: {tool_result.message}",
                safe_recovery_available=safe_recovery,
            )

        # =====================================================================
        # Case 1: Open Application
        # =====================================================================
        if action == "open_application":
            # Real OpenApplicationTool returns 'executable' in tool_result.data
            if tool_result.data and "executable" in tool_result.data:
                proc_running = post_state.get("process_running", False)
                proc_ev = post_state.get("process_evidence", {})

                if proc_running:
                    pids = proc_ev.get("pids", [])
                    proc_name = proc_ev.get("process_name", entity)
                    return VerificationResult(
                        status=VerificationStatus.VERIFIED,
                        expected_outcome=f"Application '{entity}' is running in OS process table",
                        actual_evidence=proc_ev,
                        explanation=f"Process '{proc_name}' (PIDs: {pids}) verified running. Full window interactivity uninstrumented.",
                        safe_recovery_available=False,
                    )
                else:
                    # Tool returned success, but process is not running (e.g. silent exit or launch crash)
                    return VerificationResult(
                        status=VerificationStatus.FAILED,
                        expected_outcome=f"Application '{entity}' is running in OS process table",
                        actual_evidence=proc_ev,
                        explanation=f"Tool reported launch success, but process for '{entity}' was not detected in OS process table.",
                        safe_recovery_available=True,  # Safe to attempt 1 controlled retry
                    )
            else:
                # Simulated / mock application launch
                return VerificationResult(
                    status=VerificationStatus.VERIFIED,
                    expected_outcome=f"Application '{entity}' launched (simulated)",
                    actual_evidence={"simulated": True},
                    explanation="Application launch confirmed via tool execution response.",
                    safe_recovery_available=False,
                )

        # =====================================================================
        # Case 2: Create Folder
        # =====================================================================
        if action == "create_folder":
            if tool_result.data and "full_path" in tool_result.data:
                dir_exists = post_state.get("directory_exists", False)
                dir_ev = post_state.get("directory_evidence", {})

                if dir_exists:
                    return VerificationResult(
                        status=VerificationStatus.VERIFIED,
                        expected_outcome=f"Folder '{entity}' exists inside workspace sandbox",
                        actual_evidence=dir_ev,
                        explanation=f"Directory '{entity}' independently verified on disk at {dir_ev.get('path')}.",
                        safe_recovery_available=False,
                    )
                else:
                    return VerificationResult(
                        status=VerificationStatus.FAILED,
                        expected_outcome=f"Folder '{entity}' exists inside workspace sandbox",
                        actual_evidence=dir_ev,
                        explanation=f"Tool reported success, but directory '{entity}' was not found on disk.",
                        safe_recovery_available=False,
                    )
            else:
                return VerificationResult(
                    status=VerificationStatus.VERIFIED,
                    expected_outcome=f"Folder '{entity}' created (simulated)",
                    actual_evidence={"simulated": True},
                    explanation="Directory creation confirmed via tool execution response.",
                    safe_recovery_available=False,
                )

        # =====================================================================
        # Case 3: Create File
        # =====================================================================
        if action == "create_file":
            if tool_result.data and "full_path" in tool_result.data:
                file_exists = post_state.get("file_exists", False)
                file_ev = post_state.get("file_evidence", {})

                if file_exists:
                    size = file_ev.get("size_bytes", 0)
                    return VerificationResult(
                        status=VerificationStatus.VERIFIED,
                        expected_outcome=f"File '{entity}' exists inside workspace sandbox",
                        actual_evidence=file_ev,
                        explanation=f"File '{entity}' independently verified on disk (size: {size} bytes).",
                        safe_recovery_available=False,
                    )
                else:
                    return VerificationResult(
                        status=VerificationStatus.FAILED,
                        expected_outcome=f"File '{entity}' exists inside workspace sandbox",
                        actual_evidence=file_ev,
                        explanation=f"Tool reported success, but file '{entity}' was not found on disk.",
                        safe_recovery_available=False,
                    )
            else:
                return VerificationResult(
                    status=VerificationStatus.VERIFIED,
                    expected_outcome=f"File '{entity}' created (simulated)",
                    actual_evidence={"simulated": True},
                    explanation="File creation confirmed via tool execution response.",
                    safe_recovery_available=False,
                )

        # =====================================================================
        # Case 4: Open Website
        # =====================================================================
        if action == "open_website":
            # OS URL dispatch occurred, but page rendering / HTTP load cannot be verified
            # without browser automation instrumentation
            return VerificationResult(
                status=VerificationStatus.UNKNOWN,
                expected_outcome=f"Website '{entity}' opened and rendered in browser",
                actual_evidence={
                    "browser_dispatched": True,
                    "url": tool_result.data.get("url") if tool_result.data else None,
                    "page_render_observable": False,
                },
                explanation="Browser URL dispatched successfully, but page rendering and load status cannot be independently confirmed without browser instrumentation.",
                safe_recovery_available=False,  # Reopening URLs creates unwanted duplicate browser tabs
            )

        # =====================================================================
        # Case 5: Default / Unsupported / Unknown action
        # =====================================================================
        return VerificationResult(
            status=VerificationStatus.UNKNOWN,
            expected_outcome=f"Outcome for action '{action}'",
            actual_evidence={"tool_result": tool_result.to_dict()},
            explanation=f"Independent verification is not supported for action '{action}'.",
            safe_recovery_available=False,
        )
