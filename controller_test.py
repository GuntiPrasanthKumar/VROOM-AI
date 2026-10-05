"""Automated test suite for the Activation Controller and State Machine in VROOM AI.

Tests the state machine across all required scenarios:
1. Wake word -> valid command ("Open Notepad")
2. Wake word -> unsupported command ("Delete all my files")
3. Wake word -> silence / timeout (no speech captured)
4. Wake word -> speech recognition noise / inaudible input
5. Wake word -> exit command ("Quit VROOM")
6. Repetition check: ensures the system repeatedly cycles back to WAITING without restarting.
"""

import sys
import time
from typing import List, Tuple
import numpy as np

# Ensure UTF-8 console output encoding on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass

from app.controller import ActivationController, AssistantState
from app.voice.text_to_speech import TextToSpeech


def resample_to_16k(audio_22k: np.ndarray, orig_sr: int = 22050) -> np.ndarray:
    """Resample 1D float32 audio array from 22.05 kHz to 16 kHz."""
    num_target = int(len(audio_22k) * 16000 / orig_sr)
    indices = np.linspace(0, len(audio_22k) - 1, num_target)
    return np.interp(indices, np.arange(len(audio_22k)), audio_22k).astype(np.float32)


def run_tests() -> bool:
    print("=" * 75)
    print("      VROOM AI — Activation Controller & State Machine Verification")
    print("=" * 75)

    recorded_transitions: List[Tuple[str, str]] = []

    def state_tracker(old_state: AssistantState, new_state: AssistantState):
        recorded_transitions.append((old_state.value, new_state.value))

    # Initialize single ActivationController instance
    print("[INIT] Initializing ActivationController...")
    controller = ActivationController(dev_mode=True, on_state_change=state_tracker)
    tts = TextToSpeech()
    print("[INIT] Controller and Subsystems loaded successfully.\n")

    # Synthesize wake-word audio ("Hey Jarvis") once
    synth_wake, sr_wake = tts.synthesize("Hey Jarvis")
    wake_16k = resample_to_16k(synth_wake, orig_sr=sr_wake)
    silence_pad = np.zeros(8000, dtype=np.float32)  # 0.5s padding
    padded_wake_16k = np.concatenate([silence_pad, wake_16k, silence_pad])

    test_cases = [
        {
            "id": "TEST 1: Wake Word -> Valid Command",
            "wake_audio": padded_wake_16k,
            "command_phrase": "Open Notepad",
            "expect_wake": True,
            "expect_tool_success": True,
            "expect_intent": "open_application",
            "expect_exit": False,
            "description": "Standard wake-word trigger followed by allowlisted application launch.",
        },
        {
            "id": "TEST 2: Wake Word -> Unsupported Command",
            "wake_audio": padded_wake_16k,
            "command_phrase": "Delete all my files",
            "expect_wake": True,
            "expect_tool_success": False,
            "expect_intent": "delete_files",
            "expect_exit": False,
            "description": "Wake-word trigger followed by unapproved command. Tool Router must reject.",
        },
        {
            "id": "TEST 3: Wake Word -> Silence / Timeout",
            "wake_audio": padded_wake_16k,
            "command_phrase": None,  # Pure silence
            "expect_wake": True,
            "expect_tool_success": None,
            "expect_intent": "timeout",
            "expect_exit": False,
            "description": "User activates assistant but remains silent. Must return to WAITING.",
        },
        {
            "id": "TEST 4: Wake Word -> Audio Noise / Unintelligible",
            "wake_audio": padded_wake_16k,
            "command_phrase": "NOISE",  # Low amplitude noise
            "expect_wake": True,
            "expect_tool_success": None,
            "expect_intent": "timeout",
            "expect_exit": False,
            "description": "Ambient noise without intelligible speech. Must return to WAITING.",
        },
        {
            "id": "TEST 5: Wake Word -> Graceful Exit Command",
            "wake_audio": padded_wake_16k,
            "command_phrase": "Exit VROOM",
            "expect_wake": True,
            "expect_tool_success": None,
            "expect_intent": "exit_assistant",
            "expect_exit": True,
            "description": "Wake-word trigger followed by exit request. Must set should_exit = True.",
        },
    ]

    all_passed = True
    performance_records = []

    for test in test_cases:
        t_id = test["id"]
        cmd_phrase = test["command_phrase"]
        print("\n" + "#" * 75)
        print(f"  RUNNING {t_id}")
        print(f"  Description : {test['description']}")
        print("#" * 75)

        # Prepare command audio
        if cmd_phrase is None:
            # 2.0s of pure silence
            cmd_audio_16k = np.zeros(32000, dtype=np.float32)
        elif cmd_phrase == "NOISE":
            # 2.0s of low amplitude random noise
            cmd_audio_16k = (np.random.randn(32000) * 0.005).astype(np.float32)
        else:
            synth_cmd, sr_cmd = tts.synthesize(cmd_phrase)
            cmd_audio_16k = resample_to_16k(synth_cmd, orig_sr=sr_cmd)

        # Record transition history before test
        history_len_before = len(recorded_transitions)

        # Execute cycle in controller
        outcome = controller.execute_with_audio(
            wake_audio=test["wake_audio"],
            command_audio=cmd_audio_16k,
        )

        # Verify state transitions occurred in this cycle
        transitions_this_test = recorded_transitions[history_len_before:]
        print(f"\n[TRANSITIONS OBSERVED]: {' -> '.join([t[0] for t in transitions_this_test] + [transitions_this_test[-1][1]]) if transitions_this_test else 'None'}")

        # Verification checks
        passed = True
        print("\n--- VERIFICATION ---")
        print(f"  Transcription     : {repr(outcome.get('transcription'))}")

        # 1. Wake word detection check
        if outcome.get("wake_detected") == test["expect_wake"]:
            print(f"  Wake Detection    : OK (detected={outcome.get('wake_detected')})")
        else:
            print(f"  Wake Detection    : [FAIL] Expected {test['expect_wake']}, got {outcome.get('wake_detected')}")
            passed = False

        # 2. Intent check
        observed_intent = outcome.get("intent")
        if test["expect_intent"] == "delete_files":
            # Can be delete_files or unknown
            if observed_intent in ("delete_files", "unknown"):
                print(f"  Intent Extracted  : OK ({observed_intent})")
            else:
                print(f"  Intent Extracted  : [FAIL] Expected delete_files/unknown, got {observed_intent}")
                passed = False
        elif test["expect_intent"] is not None:
            if observed_intent == test["expect_intent"]:
                print(f"  Intent Extracted  : OK ({observed_intent})")
            else:
                print(f"  Intent Extracted  : [FAIL] Expected {test['expect_intent']}, got {observed_intent}")
                passed = False

        # 3. Tool execution check
        tool_res = outcome.get("tool_result")
        if test["expect_tool_success"] is True:
            if tool_res is not None and tool_res.success:
                print(f"  Tool Execution    : OK (success={tool_res.success}, msg={tool_res.message})")
            else:
                print(f"  Tool Execution    : [FAIL] Expected tool success, got {tool_res}")
                passed = False
        elif test["expect_tool_success"] is False:
            if tool_res is not None and not tool_res.success:
                print(f"  Tool Security Check: OK (rejected as expected: {tool_res.message})")
            else:
                print(f"  Tool Security Check: [FAIL] Expected tool rejection, got {tool_res}")
                passed = False

        # 4. Return to WAITING check (or RESPONDING on exit)
        final_state = outcome.get("final_state")
        expected_final = AssistantState.RESPONDING.value if test["expect_exit"] else AssistantState.WAITING.value
        if final_state == expected_final:
            print(f"  Final State Check : OK (State: {final_state})")
        else:
            print(f"  Final State Check : [FAIL] Expected {expected_final}, got {final_state}")
            passed = False

        # 5. Exit flag check
        if outcome.get("should_exit") == test["expect_exit"]:
            print(f"  Exit Flag Check   : OK (should_exit={outcome.get('should_exit')})")
        else:
            print(f"  Exit Flag Check   : [FAIL] Expected should_exit={test['expect_exit']}, got {outcome.get('should_exit')}")
            passed = False

        status_str = "PASSED" if passed else "FAILED"
        print(f"  TEST STATUS       : [{status_str}]")

        if not passed:
            all_passed = False

        performance_records.append({
            "test": t_id.split(":")[0],
            "command": cmd_phrase or "(silence/timeout)",
            "transcription": outcome.get("transcription", ""),
            "metrics": outcome.get("metrics", {}),
            "response": outcome.get("response", ""),
        })

        time.sleep(0.5)

    # Print Latency and Telemetry Summary Table
    print("\n" + "=" * 75)
    print("              ACTIVATION CONTROLLER PERFORMANCE TELEMETRY")
    print("=" * 75)
    header = f"{'Test Case':<10} | {'Wake':<7} | {'STT':<7} | {'LLM':<7} | {'Tool':<7} | {'TTS':<7} | {'Total':<7}"
    print(header)
    print("-" * len(header))

    for rec in performance_records:
        m = rec["metrics"]
        wake_s = f"{m.get('wake_detection', 0.0):.2f}s"
        stt_s = f"{m.get('stt', 0.0):.2f}s"
        llm_s = f"{m.get('llm', 0.0):.2f}s"
        tool_s = f"{m.get('tool_execution', 0.0):.2f}s"
        tts_s = f"{m.get('tts', 0.0):.2f}s"
        tot_s = f"{m.get('total_interaction', 0.0):.2f}s"
        print(f"{rec['test']:<10} | {wake_s:<7} | {stt_s:<7} | {llm_s:<7} | {tool_s:<7} | {tts_s:<7} | {tot_s:<7}")

    print("=" * 75)
    print(f"Total Sequential Tests Run : {len(test_cases)}")
    print(f"Application Restarts Needed : 0 (Continuous single instance)")
    print(f"Overall Test Suite Result   : {'ALL 5 TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")
    print("=" * 75)

    return all_passed


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
