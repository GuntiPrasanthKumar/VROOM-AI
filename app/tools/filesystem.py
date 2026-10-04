"""Filesystem manipulation tools for VROOM AI.

Enforces strict sandboxing inside a dedicated VROOM workspace directory.
Arbitrary paths, absolute paths, path traversal, deletion, and overwriting are forbidden.
"""

from pathlib import Path
import re
from typing import Optional
from app.tools.base import BaseTool, ToolResult


class CreateFolderTool(BaseTool):
    """Tool to safely create a directory strictly inside the sandboxed VROOM workspace."""

    name: str = "create_folder"
    description: str = "Creates a new folder inside the sandboxed VROOM workspace directory."

    # Windows disallowed characters in file/directory names: <>:"/\|?*
    INVALID_CHARS_PATTERN = re.compile(r'[<>:"/\\|?*]')

    def __init__(self, workspace_root: Optional[Path] = None) -> None:
        """Initialize with a sandboxed workspace directory.

        Args:
            workspace_root: Optional custom Path. Defaults to './vroom_workspace'.
        """
        if workspace_root is None:
            # Anchor strictly inside the VROOM-AI project root
            self.workspace_root = Path(__file__).resolve().parent.parent.parent / "vroom_workspace"
        else:
            self.workspace_root = workspace_root.resolve()

        # Ensure the sandbox root exists safely
        self.workspace_root.mkdir(parents=True, exist_ok=True)

    def execute(self, entity: Optional[str]) -> ToolResult:
        """Validate the folder name and create it inside the sandbox."""
        if not entity or not entity.strip():
            return ToolResult(
                success=False,
                message="Rejected: Folder name must not be empty.",
            )

        folder_name = entity.strip()

        # Security Check 1: No path separators (prevents creating arbitrary sub-trees)
        if "/" in folder_name or "\\" in folder_name:
            return ToolResult(
                success=False,
                message=f"Rejected: Folder name '{folder_name}' contains path separators. Sub-paths and absolute paths are forbidden.",
                data={"rejected_entity": folder_name},
            )

        # Security Check 2: No Windows reserved characters
        if self.INVALID_CHARS_PATTERN.search(folder_name):
            return ToolResult(
                success=False,
                message=f"Rejected: Folder name '{folder_name}' contains illegal Windows characters (< > : \" / \\ | ? *).",
                data={"rejected_entity": folder_name},
            )

        # Security Check 3: No path traversal tricks (e.g. '..', '.')
        if folder_name in (".", "..") or folder_name.startswith(".."):
            return ToolResult(
                success=False,
                message=f"Rejected: Path traversal sequence detected in '{folder_name}'.",
                data={"rejected_entity": folder_name},
            )

        # Security Check 4: Sandbox Escape Verification
        target_path = (self.workspace_root / folder_name).resolve()
        try:
            # Check that target_path is relative to (inside) workspace_root
            target_path.relative_to(self.workspace_root)
        except ValueError:
            return ToolResult(
                success=False,
                message=f"Rejected: Sandbox violation. Target '{target_path}' is outside the VROOM workspace.",
                data={"rejected_path": str(target_path)},
            )

        # Security Check 5: No overwriting
        if target_path.exists():
            return ToolResult(
                success=False,
                message=f"Rejected: Folder '{folder_name}' already exists in workspace.",
                data={"path": str(target_path)},
            )

        # Safe execution
        try:
            target_path.mkdir(parents=False, exist_ok=False)
            return ToolResult(
                success=True,
                message=f"Successfully created folder: {folder_name} (in {self.workspace_root})",
                data={"folder_name": folder_name, "full_path": str(target_path)},
            )
        except Exception as exc:
            return ToolResult(
                success=False,
                message=f"Failed to create folder '{folder_name}': {exc}",
                data={"error": str(exc)},
            )
