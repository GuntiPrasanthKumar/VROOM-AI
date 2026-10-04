"""Ollama Local LLM Client for VROOM AI.

Communicates with the local Ollama REST API (default: http://localhost:11434)
to provide local, private reasoning without external cloud APIs.
"""

import json
import urllib.request
import urllib.error
from typing import Any, Dict, List, Optional, Tuple


class OllamaClient:
    """Client for local LLM inference via Ollama HTTP API."""

    def __init__(
        self,
        model: str = "llama3.2:1b",
        base_url: str = "http://localhost:11434",
        timeout_seconds: float = 120.0,
    ) -> None:
        """Initialize the Ollama client.

        Args:
            model: Model tag in Ollama (e.g. 'llama3.2:1b', 'qwen2.5:1.5b').
            base_url: Base HTTP URL of the local Ollama daemon.
            timeout_seconds: Network request timeout in seconds.
        """
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout_seconds

    def check_connection(self) -> Tuple[bool, str, List[str]]:
        """Check if the Ollama daemon is running and retrieve installed models.

        Returns:
            Tuple of (is_running, status_message, list_of_model_names).
        """
        endpoint = f"{self.base_url}/api/tags"
        try:
            req = urllib.request.Request(endpoint, method="GET")
            with urllib.request.urlopen(req, timeout=3.0) as resp:
                if resp.status == 200:
                    data = json.loads(resp.read().decode("utf-8"))
                    models = [m.get("name") for m in data.get("models", [])]
                    return True, "Ollama service is online.", models
        except urllib.error.URLError as err:
            return False, f"Ollama service is unreachable at {self.base_url} ({err.reason})", []
        except Exception as exc:
            return False, f"Connection check failed: {exc}", []
        return False, "Ollama service returned unexpected status.", []

    def generate(self, prompt: str, system: Optional[str] = None) -> str:
        """Send a natural-language prompt to the local LLM and return the text response.

        Args:
            prompt: User message or prompt string.
            system: Optional system instruction prompt.

        Returns:
            The raw text response from the model.
        """
        endpoint = f"{self.base_url}/api/generate"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
        }
        if system:
            payload["system"] = system

        data_bytes = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            endpoint,
            data=data_bytes,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                result = json.loads(resp.read().decode("utf-8"))
                return result.get("response", "").strip()
        except urllib.error.URLError as err:
            raise ConnectionError(
                f"Could not connect to Ollama at {self.base_url}. Is Ollama running? ({err})"
            ) from err
        except Exception as exc:
            raise RuntimeError(f"Error during Ollama inference: {exc}") from exc

    def parse_command(self, user_command: str) -> Dict[str, Any]:
        """Ask the LLM to interpret a command and return validated structured JSON.

        Extracts 'intent' and 'entity'. Performs NO execution.

        Args:
            user_command: The natural language command text.

        Returns:
            Validated dictionary with 'intent', 'entity', and 'raw_text'.
        """
        system_prompt = (
            "You are a command parser for a desktop assistant. "
            "Your job is ONLY to extract the user's intent and target entity. "
            "DO NOT execute anything. "
            "You MUST respond ONLY with a valid JSON object with exactly two keys: 'intent' and 'entity'. "
            "Possible intents include: 'open_application', 'create_folder', 'open_website', 'exit_assistant', 'unknown'. "
            "If there is no entity, set entity to null. "
            "Examples:\n"
            "Input: 'Open Chrome' -> {\"intent\": \"open_application\", \"entity\": \"Chrome\"}\n"
            "Input: 'Create a folder called Projects' -> {\"intent\": \"create_folder\", \"entity\": \"Projects\"}\n"
            "Input: 'Open YouTube' -> {\"intent\": \"open_website\", \"entity\": \"YouTube\"}\n"
            "Input: 'Exit' -> {\"intent\": \"exit_assistant\", \"entity\": null}\n"
            "Input: 'Tell me a joke' -> {\"intent\": \"unknown\", \"entity\": null}"
        )

        endpoint = f"{self.base_url}/api/generate"
        payload = {
            "model": self.model,
            "prompt": f"Command: \"{user_command}\"",
            "system": system_prompt,
            "format": "json",  # Instructs Ollama to enforce valid JSON output
            "stream": False,
            "options": {
                "temperature": 0.0,  # Deterministic generation
            },
        }

        data_bytes = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            endpoint,
            data=data_bytes,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                res_body = json.loads(resp.read().decode("utf-8"))
                raw_response = res_body.get("response", "").strip()
        except urllib.error.URLError as err:
            raise ConnectionError(
                f"Could not connect to Ollama at {self.base_url}. Is Ollama running? ({err})"
            ) from err

        # 1. Validate that the response is valid JSON
        try:
            parsed = json.loads(raw_response)
        except json.JSONDecodeError as jde:
            return {
                "intent": "unknown",
                "entity": None,
                "raw_text": user_command,
                "error": f"Invalid JSON received from LLM: {jde}",
            }

        # 2. Validate expected fields
        if not isinstance(parsed, dict) or "intent" not in parsed:
            return {
                "intent": "unknown",
                "entity": None,
                "raw_text": user_command,
                "error": f"Missing 'intent' key in LLM response: {parsed}",
            }

        intent = str(parsed.get("intent", "unknown")).lower().strip()
        entity = parsed.get("entity")
        if entity is not None:
            entity = str(entity).strip()

        return {
            "intent": intent,
            "entity": entity,
            "raw_text": user_command,
        }
