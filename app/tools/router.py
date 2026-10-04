"""Tool Router for VROOM AI.

Acts as the secure, validating mediator between the AI Brain and the Operating System.
Ensures zero execution of arbitrary or unvetted intents.
"""

from typing import Any, Dict, Optional, Union
from app.brain.command_parser import ParsedCommand
from app.tools.base import ToolResult
from app.tools.registry import ToolRegistry


class ToolRouter:
    """Routes validated structured commands to approved tool handlers."""

    def __init__(self, registry: Optional[ToolRegistry] = None) -> None:
        """Initialize the router with a ToolRegistry.

        Args:
            registry: Optional custom ToolRegistry. Defaults to standard registry.
        """
        self.registry = registry if registry is not None else ToolRegistry()

    def route_and_execute(self, command: Union[Dict[str, Any], ParsedCommand]) -> ToolResult:
        """Receive a structured command, validate its intent, and dispatch it to an approved tool.

        Args:
            command: A dictionary with 'intent' and 'entity' keys, or a ParsedCommand instance.

        Returns:
            ToolResult indicating execution status or rejection reason.
        """
        # Step 1: Normalize and validate command structure
        if isinstance(command, ParsedCommand):
            intent = command.intent
            entity = command.entity
        elif isinstance(command, dict):
            intent = command.get("intent")
            entity = command.get("entity")
        else:
            return ToolResult(
                success=False,
                message=f"Rejected: Invalid command structure. Expected dict or ParsedCommand, got {type(command).__name__}.",
            )

        # Step 2: Validate intent presence
        if not intent or not str(intent).strip():
            return ToolResult(
                success=False,
                message="Rejected: Command contains no 'intent' field.",
            )

        clean_intent = str(intent).strip().lower()

        # Step 3: Check Tool Registry for approved tool
        tool = self.registry.get_tool(clean_intent)
        if tool is None:
            approved = self.registry.list_approved_intents()
            return ToolResult(
                success=False,
                message=f"Rejected: Intent '{intent}' does not map to any approved tool. Approved intents: {approved}",
                data={"rejected_intent": intent, "approved_intents": approved},
            )

        # Step 4: Dispatch to the vetted tool (which performs entity validation)
        return tool.execute(entity)
