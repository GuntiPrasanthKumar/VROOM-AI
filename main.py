"""VROOM AI - Intelligent Desktop Voice Assistant.

Main entry point for running the end-to-end voice assistant.
Coordinates Audio Capture -> Whisper STT -> Ollama Brain -> Tool Router -> Safe Tools -> Piper TTS.
"""

import sys
import platform

# Ensure UTF-8 output encoding on Windows so international path characters
# (such as Korean Hangul in OneDrive) and symbols do not crash the console stream.
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass

from app.orchestrator import AssistantOrchestrator


def main() -> int:
    """Main execution function for VROOM AI."""
    print("=" * 65)
    print("        VROOM AI — Intelligent Desktop Voice Assistant")
    print("=" * 65)
    print(f"Python Runtime : {platform.python_version()} on {platform.system()} {platform.release()}")
    print("Brain Engine   : Ollama (llama3.2:1b) [Local]")
    print("Speech-to-Text : Faster-Whisper (base, int8) [Local]")
    print("Text-to-Speech : Piper TTS (en_US-lessac-medium) [Local]")
    print("Security Mode  : Strict Tool Router & Sandboxed Workspace")
    print("=" * 65)

    try:
        orchestrator = AssistantOrchestrator(dev_mode=True)
        orchestrator.run_interactive(record_duration=4.0)
        return 0
    except Exception as exc:
        print(f"\n[FATAL ERROR] VROOM startup or runtime failed: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
