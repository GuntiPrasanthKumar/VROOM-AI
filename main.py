"""VROOM AI - Intelligent Desktop Voice Assistant.

Entry point for the project. At this foundation stage, it verifies that
the Python runtime environment is properly configured.
"""

import sys
import platform

# Ensure UTF-8 output encoding on Windows so international path characters
# (such as Korean Hangul) and special symbols do not crash the console stream.
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass


def main() -> int:
    """Main execution function for VROOM AI."""
    print("=" * 60)
    print("       VROOM AI - Intelligent Desktop Voice Assistant")
    print("=" * 60)
    print(f"Python Version : {platform.python_version()}")
    print(f"Platform       : {platform.system()} {platform.release()} ({platform.machine()})")
    print(f"Executable     : {sys.executable}")
    print("Status         : Foundation stage initialized successfully.")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
