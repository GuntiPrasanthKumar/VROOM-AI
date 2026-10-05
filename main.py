"""VROOM AI - Intelligent Desktop Voice Assistant.

Main entry point for running the end-to-end voice assistant with Wake-Word Activation.
Lifecycle: WAITING -> ACTIVATED -> LISTENING -> PROCESSING -> EXECUTING -> RESPONDING -> WAITING.
"""

import sys
import platform

# Ensure UTF-8 output encoding on Windows so special characters and paths don't crash console
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass

from app.controller import ActivationController


def main() -> int:
    """Main execution function for VROOM AI."""
    print("=" * 65)
    print("        VROOM AI — Intelligent Desktop Voice Assistant")
    print("=" * 65)
    print(f"Python Runtime : {platform.python_version()} on {platform.system()} {platform.release()}")
    print("Activation 1   : 👏 Physical Acoustic Clap Detector [Local]")
    print("Activation 2   : OpenWakeWord ('Hey Jarvis') [Local]")
    print("Brain Engine   : Ollama (llama3.2:1b) [Local]")
    print("Speech-to-Text : Faster-Whisper (base, int8) [Local]")
    print("Text-to-Speech : Piper TTS (en_US-lessac-medium) [Local]")
    print("Security Mode  : Strict Tool Router & Sandboxed Workspace")
    print("Lifecycle Mode : Staged (CLAP -> WAKE -> LISTEN -> EXECUTE -> CLAP)")
    print("=" * 65)

    try:
        controller = ActivationController(dev_mode=True)
        controller.run(command_duration=4.0)
        return 0
    except Exception as exc:
        print(f"\n[FATAL ERROR] VROOM startup or runtime failed: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
