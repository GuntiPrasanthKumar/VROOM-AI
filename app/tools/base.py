"""Base contracts and result data structures for VROOM AI tools.

Every tool in VROOM must adhere to this contract:
Input (Entity) -> Strict Validation -> Safe Execution -> Structured ToolResult
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class ToolResult:
    """Immutable result structure returned by every tool execution.

    Attributes:
        success: Boolean flag indicating if the tool executed successfully.
        message: Human-readable explanation of the outcome or rejection reason.
        data: Optional supplementary data or diagnostic details.
    """
    success: bool
    message: str
    data: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert the result to a standard JSON-serializable dictionary."""
        return {
            "success": self.success,
            "message": self.message,
            "data": self.data or {},
        }


class BaseTool(ABC):
    """Abstract base class that all VROOM tools must implement."""

    name: str
    description: str

    @abstractmethod
    def execute(self, entity: Optional[str]) -> ToolResult:
        """Execute the tool action against the provided entity.

        Args:
            entity: The target parameter extracted by the brain/parser (e.g. 'Chrome').

        Returns:
            ToolResult containing success status and explanation.
        """
        pass
