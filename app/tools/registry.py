"""Tool registry for VROOM AI.

Maintains the authoritative registry of approved tools mapped by intent.
"""

from typing import Dict, List, Optional
from app.tools.base import BaseTool
from app.tools.desktop import OpenApplicationTool
from app.tools.browser import OpenWebsiteTool
from app.tools.filesystem import CreateFolderTool


class ToolRegistry:
    """Registry maintaining approved tools mapped by their intent identifier."""

    def __init__(self, register_defaults: bool = True) -> None:
        self._tools: Dict[str, BaseTool] = {}
        if register_defaults:
            self._register_default_tools()

    def _register_default_tools(self) -> None:
        """Register the baseline approved tools for VROOM."""
        self.register_tool(OpenApplicationTool())
        self.register_tool(OpenWebsiteTool())
        self.register_tool(CreateFolderTool())

    def register_tool(self, tool: BaseTool) -> None:
        """Register an approved tool under its canonical intent name.

        Args:
            tool: An instance implementing BaseTool.
        """
        self._tools[tool.name.lower()] = tool

    def get_tool(self, intent: str) -> Optional[BaseTool]:
        """Retrieve an approved tool by its intent name.

        Args:
            intent: The intent string (e.g. 'open_application').

        Returns:
            The matching BaseTool instance or None if unapproved/unregistered.
        """
        if not intent:
            return None
        return self._tools.get(intent.strip().lower())

    def list_approved_intents(self) -> List[str]:
        """Return a sorted list of all registered intent strings."""
        return sorted(self._tools.keys())
