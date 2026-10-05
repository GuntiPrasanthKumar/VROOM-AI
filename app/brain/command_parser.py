"""Deterministic Natural Language Command Parser for VROOM AI.

Extracts structured intent and entities from raw text commands.
IMPORTANT: This module performs ONLY understanding. It NEVER executes actions.
"""

from dataclasses import dataclass, asdict
import re
from typing import Any, Dict, Optional, Set


@dataclass(frozen=True)
class ParsedCommand:
    """Represents a structured command extracted from natural language text.

    Attributes:
        intent: The identified goal or action (e.g., 'open_application', 'create_folder').
        entity: The target parameter or object of the action (e.g., 'Chrome', 'Projects').
        raw_text: The original user input string.
    """
    intent: str
    entity: Optional[str] = None
    raw_text: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert the parsed command to a standard Python dictionary."""
        return {
            "intent": self.intent,
            "entity": self.entity,
            "raw_text": self.raw_text,
        }


class CommandParser:
    """Parses text commands into structured Intent and Entity pairs.

    Uses deterministic pattern matching and rule-based normalization.
    Acts as the bridge between Speech-to-Text and future Tool Routing.
    """

    # Common popular websites to distinguish 'open_website' from 'open_application'
    KNOWN_WEBSITES: Set[str] = {
        "youtube",
        "google",
        "github",
        "reddit",
        "twitter",
        "wikipedia",
        "amazon",
        "netflix",
        "gmail",
        "facebook",
        "instagram",
        "linkedin",
    }

    # Exit keywords that indicate the user wants to close the assistant
    EXIT_KEYWORDS: Set[str] = {
        "exit",
        "quit",
        "stop",
        "bye",
        "goodbye",
        "close assistant",
    }

    def __init__(self) -> None:
        # Precompile regular expressions for efficiency and clarity
        # Matches: "create a folder called Projects", "make folder named Work", etc.
        self._folder_pattern = re.compile(
            r"^(?:create|make)\s+(?:a\s+)?folder(?:\s+(?:called|named))?\s+(.+)$",
            re.IGNORECASE,
        )

        # Matches: "open Chrome", "launch Notepad", "start Spotify", etc.
        self._open_pattern = re.compile(
            r"^(?:open|launch|start)\s+(.+)$",
            re.IGNORECASE,
        )

    def parse(self, text: str) -> ParsedCommand:
        """Parse a natural language text command into a ParsedCommand.

        Args:
            text: Raw natural language string (e.g. from speech recognition).

        Returns:
            ParsedCommand instance with intent, entity, and raw_text.
        """
        if not text or not text.strip():
            return ParsedCommand(intent="unknown", entity=None, raw_text=text or "")

        raw_text = text.strip()
        # Clean trailing punctuation commonly introduced by speech-to-text models
        normalized = raw_text.rstrip(".!?,;:").strip()
        lower_text = normalized.lower()

        # 1. Check for Exit Assistant commands
        if lower_text in self.EXIT_KEYWORDS or lower_text.startswith(("quit", "exit", "stop", "shut down", "power down", "bye")):
            return ParsedCommand(
                intent="exit_assistant",
                entity=None,
                raw_text=raw_text,
            )

        # 2. Check for Create Folder commands
        folder_match = self._folder_pattern.match(normalized)
        if folder_match:
            folder_name = folder_match.group(1).strip().strip("'\"")
            if folder_name:
                return ParsedCommand(
                    intent="create_folder",
                    entity=folder_name,
                    raw_text=raw_text,
                )

        # 3. Check for Open / Launch / Start commands
        open_match = self._open_pattern.match(normalized)
        if open_match:
            target = open_match.group(1).strip().strip("'\"")
            target_lower = target.lower()

            # Distinguish between websites and desktop applications
            is_website = (
                target_lower in self.KNOWN_WEBSITES
                or target_lower.startswith(("http://", "https://", "www."))
                or any(target_lower.endswith(tld) for tld in [".com", ".org", ".net", ".io", ".edu", ".dev"])
            )

            if is_website:
                return ParsedCommand(
                    intent="open_website",
                    entity=target,
                    raw_text=raw_text,
                )
            else:
                return ParsedCommand(
                    intent="open_application",
                    entity=target,
                    raw_text=raw_text,
                )

        # 4. Fallback for unrecognized or unsupported commands
        return ParsedCommand(
            intent="unknown",
            entity=None,
            raw_text=raw_text,
        )
