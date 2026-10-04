"""Browser automation tool for VROOM AI.

Safely opens approved websites in the user's default browser.
Arbitrary URLs and unvetted websites from the LLM are strictly rejected.
"""

from typing import Dict, Optional
import webbrowser
from app.tools.base import BaseTool, ToolResult


class OpenWebsiteTool(BaseTool):
    """Tool to safely open verified, allowlisted websites."""

    name: str = "open_website"
    description: str = "Opens an approved website (YouTube, Google, GitHub) in the default browser."

    def __init__(self) -> None:
        # Explicit allowlist of pre-vetted, hardcoded HTTPS URLs
        self._allowlist: Dict[str, str] = {
            "youtube": "https://www.youtube.com",
            "google": "https://www.google.com",
            "github": "https://www.github.com",
        }

    def execute(self, entity: Optional[str]) -> ToolResult:
        """Validate the website name and safely open the pre-approved URL."""
        if not entity or not entity.strip():
            return ToolResult(
                success=False,
                message="Rejected: Website name must not be empty.",
            )

        clean_name = entity.strip().lower()

        # Security Check: Reject arbitrary or unvetted domains/URLs
        if clean_name not in self._allowlist:
            approved_sites = sorted(k.capitalize() for k in self._allowlist.keys())
            return ToolResult(
                success=False,
                message=f"Rejected: Website '{entity}' is not in the approved allowlist. Allowed: {approved_sites}",
                data={"rejected_entity": entity, "allowed_websites": approved_sites},
            )

        target_url = self._allowlist[clean_name]

        try:
            # Opens in a new tab if possible (new=2)
            opened = webbrowser.open(target_url, new=2)
            if opened:
                return ToolResult(
                    success=True,
                    message=f"Successfully opened website: {entity} ({target_url})",
                    data={"url": target_url, "entity": entity},
                )
            else:
                return ToolResult(
                    success=False,
                    message=f"Browser controller reported failure when opening '{target_url}'.",
                    data={"url": target_url},
                )
        except Exception as exc:
            return ToolResult(
                success=False,
                message=f"Failed to open website '{entity}': {exc}",
                data={"error": str(exc)},
            )
