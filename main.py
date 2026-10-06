"""VROOM AI - Intelligent Desktop Voice Assistant.

Main entry point for running the end-to-end voice assistant with Wake-Word Activation.
Lifecycle: WAITING -> ACTIVATED -> LISTENING -> PROCESSING -> EXECUTING -> RESPONDING -> WAITING.
"""

import argparse
import platform
import sys

# Ensure UTF-8 output encoding on Windows so special characters and paths don't crash console
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass

from app.controller import ActivationController


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for VROOM AI runtime."""
    parser = argparse.ArgumentParser(
        description="VROOM AI — Intelligent Desktop Voice Assistant with Staged Activation."
    )
    parser.add_argument(
        "--wake-only",
        "-w",
        action="store_true",
        help="Activate directly via wake phrase ('Hey Jarvis') without requiring a physical clap.",
    )
    parser.add_argument(
        "--clap-timeout",
        type=float,
        default=None,
        help="Maximum time in seconds to wait for a clap before cycling (default: continuous).",
    )
    parser.add_argument(
        "--wake-timeout",
        type=float,
        default=6.0,
        help="Time in seconds to wait for the wake phrase after a clap (default: 6.0s).",
    )
    parser.add_argument(
        "--duration",
        "-d",
        type=float,
        default=4.0,
        help="Audio recording duration in seconds for user commands (default: 4.0s).",
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Suppress verbose diagnostic output.",
    )
    return parser.parse_args()


def main() -> int:
    """Main execution function for VROOM AI."""
    args = parse_args()
    mode = "wake_only" if args.wake_only else "staged"
    dev_mode = not args.quiet

    print("=" * 65)
    print("        VROOM AI - Intelligent Desktop Voice Assistant")
    print("=" * 65)
    print(f"Python Runtime : {platform.python_version()} on {platform.system()} {platform.release()}")
    if mode == "staged":
        print("Activation 1   : [CLAP] Physical Acoustic Clap Detector [Local, Calibrated]")
        print(f"Activation 2   : [VOICE] OpenWakeWord ('Hey Jarvis', timeout: {args.wake_timeout}s) [Local]")
        print("Lifecycle Mode : Staged (CLAP -> WAKE -> LISTEN -> EXECUTE -> CLAP)")
    else:
        print("Activation Gate: [VOICE] OpenWakeWord ('Hey Jarvis') [Direct Wake Mode]")
        print("Lifecycle Mode : Wake-Only (WAKE -> LISTEN -> EXECUTE -> WAKE)")
    print("Brain Engine   : Ollama (llama3.2:1b) [Local]")
    print("Speech-to-Text : Faster-Whisper (base, int8) [Local]")
    print("Text-to-Speech : Piper TTS (en_US-lessac-medium) [Local]")
    print("Security Mode  : Strict Tool Router & Sandboxed Workspace")
    print("=" * 65)

    try:
        controller = ActivationController(
            mode=mode,
            wake_timeout=args.wake_timeout,
            dev_mode=dev_mode,
        )
        controller.run(command_duration=args.duration)
        return 0
    except Exception as exc:
        print(f"\n[FATAL ERROR] VROOM startup or runtime failed: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
