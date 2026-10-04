"""Desktop application automation tool for VROOM AI.

Safely opens approved desktop applications using an explicit allowlist.
Arbitrary paths and unapproved applications are strictly rejected.
"""

import os
import shutil
import subprocess
from typing import Dict, List, Optional
from app.tools.base import BaseTool, ToolResult


class OpenApplicationTool(BaseTool):
    """Tool to safely launch approved desktop applications."""

    name: str = "open_application"
    description: str = "Opens an explicitly approved desktop application (Chrome, Notepad, Calculator)."

    def __init__(self) -> None:
        # Explicit allowlist mapping normalized names to executable candidates
        self._allowlist: Dict[str, List[str]] = {
            "chrome": [
                r"C:\Program Files\Google\Chrome\Application\chrome.exe",
                r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
                os.path.expandvars(r"%LOCALAPPDATA%\Google\Chrome\Application\chrome.exe"),
            ],
            "google chrome": [
                r"C:\Program Files\Google\Chrome\Application\chrome.exe",
                r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
                os.path.expandvars(r"%LOCALAPPDATA%\Google\Chrome\Application\chrome.exe"),
            ],
            "notepad": ["notepad.exe"],
            "calculator": ["calc.exe"],
            "calc": ["calc.exe"],
        }

    def _resolve_executable(self, key: str) -> Optional[str]:
        """Verify existence of the executable from the approved allowlist."""
        candidates = self._allowlist.get(key, [])
        for candidate in candidates:
            # If it's a full path, check if it exists on disk
            if os.path.isabs(candidate) and os.path.exists(candidate):
                return candidate
            # If it's a simple executable name (like notepad.exe), check PATH
            resolved = shutil.which(candidate)
            if resolved:
                return resolved
        return None

    def execute(self, entity: Optional[str]) -> ToolResult:
        """Validate the application name against the allowlist and launch it safely."""
        if not entity or not entity.strip():
            return ToolResult(
                success=False,
                message="Rejected: Application name must not be empty.",
            )

        clean_name = entity.strip().lower()

        # Security Check: Must be in explicit allowlist
        if clean_name not in self._allowlist:
            approved_apps = sorted(set(k.capitalize() for k in self._allowlist.keys()))
            return ToolResult(
                success=False,
                message=f"Rejected: Application '{entity}' is not in the approved allowlist. Allowed: {approved_apps}",
                data={"rejected_entity": entity, "allowed_applications": approved_apps},
            )

        # Resolve verified executable
        executable = self._resolve_executable(clean_name)
        if not executable:
            return ToolResult(
                success=False,
                message=f"Application '{entity}' is approved, but the executable could not be found on this system.",
                data={"entity": entity},
            )

        # Safe launch: Never pass shell=True, execute the exact binary directly
        try:
            subprocess.Popen([executable], shell=False)
            return ToolResult(
                success=True,
                message=f"Successfully launched application: {entity}",
                data={"executable": executable, "entity": entity},
            )
        except Exception as exc:
            return ToolResult(
                success=False,
                message=f"Failed to launch application '{entity}': {exc}",
                data={"error": str(exc)},
            )
