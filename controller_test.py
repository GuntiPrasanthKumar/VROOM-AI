"""Automated test suite for the Two-Stage Activation Controller (Clap + Wake Word).

Tests the 5 required scenarios:
1. Clap -> wake phrase -> valid command ("Open Notepad") -> returns to WAITING_FOR_CLAP.
2. Clap -> no wake phrase -> timeout -> returns to WAITING_FOR_CLAP.
3. No clap -> wake phrase spoken -> does NOT activate (Gate 1 holds).
4. Clap -> wake phrase -> unsupported command ("Delete all my files") -> rejected -> WAITING_FOR_CLAP.
5. Repeated activation cycles without restarting VROOM.
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


def generate_synthetic_clap(sample_rate: int = 16000) -> np.ndarray:
    """Generate a sharp impulsive transient sound wave (clap) with steep decay."""
    n_samples = int(0.04 * sample_rate)  # 40ms duration
    noise = np.random.uniform(-1.0, 1.0, n_samples)
    envelope = np.exp(-np.linspace(0, 15, n_samples))
    clap = (noise * envelope * 0.9).astype(np.float32)
    # Prepend silence and append silence
    silence = np.zeros(2048, dtype=np.float32)
    return np.concatenate([silence, clap, silence])


def run_tests() -> bool:
    print("=" * 75)
    print("  VROOM AI — Staged Activation (Clap + Wake Word) Verification Suite")
    print("=" * 75)

    recorded_transitions: List[Tuple[str, str]] = []

    def state_tracker(old_state: AssistantState, new_state: AssistantState):
        recorded_transitions.append((old_state.value, new_state.value))

    # Initialize single ActivationController instance
    print("[INIT] Initializing ActivationController with Clap and WakeWord detectors...")
    controller = ActivationController(dev_mode=True, on_state_change=state_tracker)
    tts = TextToSpeech()
    print("[INIT] Controller and Subsystems loaded successfully.\n")

    # Generate synthetic clap audio
    clap_audio = generate_synthetic_clap()

    # Synthesize wake-word audio ("Hey Jarvis")
    synth_wake, sr_wake = tts.synthesize("Hey Jarvis")
    wake_16k = resample_to_16k(synth_wake, orig_sr=sr_wake)
    silence_pad = np.zeros(8000, dtype=np.float32)
    padded_wake_16k = np.concatenate([silence_pad, wake_16k, silence_pad])

    # Synthesize commands
    synth_notepad, sr_np = tts.synthesize("Open Notepad")
    cmd_notepad_16k = resample_to_16k(synth_notepad, orig_sr=sr_np)

    synth_unsupported, sr_un = tts.synthesize("Delete all my files")
    cmd_unsupported_16k = resample_to_16k(synth_unsupported, orig_sr=sr_un)

    synth_chrome, sr_cr = tts.synthesize("Open Chrome")
    cmd_chrome_16k = resample_to_16k(synth_chrome, orig_sr=sr_cr)

    synth_exit, sr_ex = tts.synthesize("Exit VROOM")
    cmd_exit_16k = resample_to_16k(synth_exit, orig_sr=sr_ex)

    # Pure silence
    silence_audio = np.zeros(32000, dtype=np.float32)

    test_scenarios = [
        {
            "id": "TEST 1: Clap -> Wake Phrase -> Valid Command",
            "clap_audio": clap_audio,
            "wake_audio": padded_wake_16k,
            "command_audio": cmd_notepad_16k,
            "expect_clap": True,
            "expect_wake": True,
            "expect_tool_success": True,
            "expect_final_state": AssistantState.WAITING_FOR_CLAP.value,
            "expect_exit": False,
            "description": "Gate 1 (Clap) and Gate 2 (Wake) satisfied -> Launches Notepad -> returns to WAITING_FOR_CLAP.",
        },
        {
            "id": "TEST 2: Clap -> No Wake Phrase (Timeout)",
            "clap_audio": clap_audio,
            "wake_audio": silence_audio,
            "command_audio": None,
            "expect_clap": True,
            "expect_wake": False,
            "expect_tool_success": None,
            "expect_final_state": AssistantState.WAITING_FOR_CLAP.value,
            "expect_exit": False,
            "description": "Gate 1 (Clap) fires, but no wake phrase follows -> Times out -> returns to WAITING_FOR_CLAP.",
        },
        {
            "id": "TEST 3: No Clap -> Wake Phrase Spoken",
            "clap_audio": padded_wake_16k,  # Voice spoken while in WAITING_FOR_CLAP
            "wake_audio": padded_wake_16k,
            "command_audio": cmd_notepad_16k,
            "expect_clap": False,
            "expect_wake": False,
            "expect_tool_success": None,
            "expect_final_state": AssistantState.WAITING_FOR_CLAP.value,
            "expect_exit": False,
            "description": "User speaks wake phrase WITHOUT clapping -> VROOM must NOT activate (Gate 1 holds).",
        },
        {
            "id": "TEST 4: Clap -> Wake Phrase -> Unsupported Command",
            "clap_audio": clap_audio,
            "wake_audio": padded_wake_16k,
            "command_audio": cmd_unsupported_16k,
            "expect_clap": True,
            "expect_wake": True,
            "expect_tool_success": False,
            "expect_final_state": AssistantState.WAITING_FOR_CLAP.value,
            "expect_exit": False,
            "description": "Both gates satisfied, but command is dangerous -> Tool Router rejects -> returns to WAITING_FOR_CLAP.",
        },
        {
            "id": "TEST 5A: Repeated Cycle A (Chrome)",
            "clap_audio": clap_audio,
            "wake_audio": padded_wake_16k,
            "command_audio": cmd_chrome_16k,
            "expect_clap": True,
            "expect_wake": True,
            "expect_tool_success": True,
            "expect_final_state": AssistantState.WAITING_FOR_CLAP.value,
            "expect_exit": False,
            "description": "Subsequent consecutive cycle in same instance -> Launches Chrome -> WAITING_FOR_CLAP.",
        },
        {
            "id": "TEST 5B: Repeated Cycle B (Exit)",
            "clap_audio": clap_audio,
            "wake_audio": padded_wake_16k,
            "command_audio": cmd_exit_16k,
            "expect_clap": True,
            "expect_wake": True,
            "expect_tool_success": None,
            "expect_final_state": AssistantState.RESPONDING.value,
            "expect_exit": True,
            "description": "Consecutive exit cycle in same instance -> Sets should_exit=True -> Clean shutdown.",
        },
    ]

    all_passed = True
    performance_records = []

    for test in test_scenarios:
        t_id = test["id"]
        print("\n" + "#" * 75)
        print(f"  RUNNING {t_id}")
        print(f"  Description : {test['description']}")
        print("#" * 75)

        history_len_before = len(recorded_transitions)

        # Run cycle through controller
        outcome = controller.execute_with_audio(
            clap_audio=test["clap_audio"],
            wake_audio=test["wake_audio"],
            command_audio=test["command_audio"],
        )

        transitions_this_test = recorded_transitions[history_len_before:]
        print(f"\n[TRANSITIONS OBSERVED]: {' -> '.join([t[0] for t in transitions_this_test] + [transitions_this_test[-1][1]]) if transitions_this_test else 'None'}")

        # Verification checks
        passed = True
        print("\n--- VERIFICATION ---")

        # 1. Clap detection check
        if outcome.get("clap_detected") == test["expect_clap"]:
            print(f"  Clap Gate 1 Check : OK (clap_detected={outcome.get('clap_detected')})")
        else:
            print(f"  Clap Gate 1 Check : [FAIL] Expected {test['expect_clap']}, got {outcome.get('clap_detected')}")
            passed = False

        # 2. Wake detection check
        if outcome.get("wake_detected") == test["expect_wake"]:
            print(f"  Wake Gate 2 Check : OK (wake_detected={outcome.get('wake_detected')})")
        else:
            print(f"  Wake Gate 2 Check : [FAIL] Expected {test['expect_wake']}, got {outcome.get('wake_detected')}")
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
                print(f"  Tool Security     : OK (rejected as expected: {tool_res.message})")
            else:
                print(f"  Tool Security     : [FAIL] Expected tool rejection, got {tool_res}")
                passed = False

        # 4. Final state check
        final_state = outcome.get("final_state")
        if final_state == test["expect_final_state"]:
            print(f"  Final State Check : OK (State: {final_state})")
        else:
            print(f"  Final State Check : [FAIL] Expected {test['expect_final_state']}, got {final_state}")
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
            "metrics": outcome.get("metrics", {}),
            "final_state": final_state,
        })

        time.sleep(0.5)

    # Print Latency and Telemetry Summary Table
    print("\n" + "=" * 75)
    print("               STAGED ACTIVATION PERFORMANCE TELEMETRY")
    print("=" * 75)
    header = f"{'Test Case':<10} | {'Clap':<7} | {'Wake':<7} | {'STT':<7} | {'LLM':<7} | {'Tool':<7} | {'TTS':<7} | {'Total':<7}"
    print(header)
    print("-" * len(header))

    for rec in performance_records:
        m = rec["metrics"]
        clap_s = f"{m.get('clap_detection', 0.0):.2f}s"
        wake_s = f"{m.get('wake_detection', 0.0):.2f}s"
        stt_s = f"{m.get('stt', 0.0):.2f}s"
        llm_s = f"{m.get('llm', 0.0):.2f}s"
        tool_s = f"{m.get('tool_execution', 0.0):.2f}s"
        tts_s = f"{m.get('tts', 0.0):.2f}s"
        tot_s = f"{m.get('total_interaction', 0.0):.2f}s"
        print(f"{rec['test']:<10} | {clap_s:<7} | {wake_s:<7} | {stt_s:<7} | {llm_s:<7} | {tool_s:<7} | {tts_s:<7} | {tot_s:<7}")

    print("=" * 75)
    print(f"Total Scenarios Tested      : {len(test_scenarios)}")
    print(f"Application Restarts Needed : 0 (Continuous single instance)")
    print(f"Overall Test Suite Result   : {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")
    print("=" * 75)

    return all_passed


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
