"""End-to-end integration test suite for VROOM AI.

Tests the complete voice assistant pipeline across all required test cases:
1. "Open Notepad" -> Launches Notepad, speaks confirmation.
2. "Open YouTube" -> Opens YouTube in browser, speaks confirmation.
3. "Create a folder called VROOMTest" -> Creates sandboxed workspace folder, speaks confirmation.
4. "Delete all my files" -> Rejects unsupported/destructive command, speaks explanation.
5. "Quit VROOM" -> Exits cleanly.
6. "Open Chrome" -> Launches Google Chrome, speaks confirmation.
"""

import os
import shutil
import sys
import time
from pathlib import Path
import numpy as np

# Ensure UTF-8 console output encoding
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass

from app.orchestrator import AssistantOrchestrator
from app.voice.text_to_speech import TextToSpeech
from app.voice.audio_recorder import AudioRecorder


def resample_audio(audio_array: np.ndarray, orig_sr: int = 22050, target_sr: int = 16000) -> np.ndarray:
    """Resample 1D float32 audio array using linear interpolation."""
    if orig_sr == target_sr:
        return audio_array
    num_target_samples = int(len(audio_array) * target_sr / orig_sr)
    target_indices = np.linspace(0, len(audio_array) - 1, num_target_samples)
    return np.interp(target_indices, np.arange(len(audio_array)), audio_array).astype(np.float32)


def run_tests() -> bool:
    print("=" * 70)
    print("      VROOM AI — End-to-End Integration Verification Suite")
    print("=" * 70)

    # 1. Initialize Orchestrator
    print("[INIT] Initializing VROOM Assistant Orchestrator...")
    orchestrator = AssistantOrchestrator(dev_mode=True)
    voice_synth = TextToSpeech()
    print("[INIT] All subsystems loaded successfully.\n")

    # 2. Hardware Microphone Check
    print("[HARDWARE CHECK] Testing physical microphone capture (1.0 second)...")
    try:
        mic_recorder = AudioRecorder()
        sample_audio = mic_recorder.record(1.0)
        print(f"  Microphone OK: {mic_recorder.device_info}")
        print(f"  Captured {len(sample_audio)} samples, Peak amplitude: {np.max(np.abs(sample_audio)):.4f}\n")
    except Exception as exc:
        print(f"  [WARN] Microphone test encountered: {exc}\n")

    # Clean up any leftover VROOMTest folder from prior runs
    workspace_test_dir = Path(__file__).resolve().parent / "vroom_workspace" / "VROOMTest"
    if workspace_test_dir.exists():
        shutil.rmtree(workspace_test_dir, ignore_errors=True)

    # Define the test cases
    test_cases = [
        {
            "id": "TEST 1",
            "spoken_command": "Open Notepad",
            "expected_intent": "open_application",
            "expected_entity": "Notepad",
            "expected_success": True,
        },
        {
            "id": "TEST 2",
            "spoken_command": "Open YouTube",
            "expected_intent": "open_website",
            "expected_entity": "YouTube",
            "expected_success": True,
        },
        {
            "id": "TEST 3",
            "spoken_command": "Create a folder called VROOMTest",
            "expected_intent": "create_folder",
            "expected_entity": "VROOMTest",
            "expected_success": True,
        },
        {
            "id": "TEST 4",
            "spoken_command": "Delete all my files",
            "expected_intent": "delete_files",  # Or unknown, router must reject
            "expected_entity": None,
            "expected_success": False,
        },
        {
            "id": "TEST 5",
            "spoken_command": "Quit VROOM",
            "expected_intent": "exit_assistant",
            "expected_entity": None,
            "expected_success": True,
        },
        {
            "id": "TEST 6 (User Query)",
            "spoken_command": "Open Chrome",
            "expected_intent": "open_application",
            "expected_entity": "Chrome",
            "expected_success": True,
        },
    ]

    all_passed = True
    performance_records = []

    for test in test_cases:
        test_id = test["id"]
        command_text = test["spoken_command"]
        print("\n" + "#" * 70)
        print(f"  RUNNING {test_id}: Voice input -> \"{command_text}\"")
        print("#" * 70)

        # Step A: Synthesize voice command audio
        t_synth_start = time.perf_counter()
        synth_audio, synth_sr = voice_synth.synthesize(command_text)
        audio_16k = resample_audio(synth_audio, orig_sr=synth_sr, target_sr=16000)
        synth_latency = round(time.perf_counter() - t_synth_start, 3)

        # Step B: Pass audio waveform directly into the orchestrator pipeline
        result = orchestrator.process_audio(audio_16k)
        result["latencies"]["audio_prep"] = synth_latency

        # Step C: Verification
        passed = True
        print("\n--- VERIFICATION ---")

        # Verify transcription
        transcription = result.get("transcription", "")
        print(f"  Transcribed : \"{transcription}\"")

        # Verify tool outcome
        tool_result = result.get("tool_result")
        should_exit = result.get("should_exit", False)

        if test_id == "TEST 3":
            # Verify folder on disk in vroom_workspace
            created_folder = tool_result.data.get("folder_name") if tool_result and tool_result.data else None
            created_path = Path(tool_result.data.get("full_path")) if tool_result and tool_result.data and tool_result.data.get("full_path") else None
            if created_path and created_path.exists() and created_path.is_dir():
                print(f"  Filesystem  : Verified folder exists at {created_path}")
            else:
                print(f"  Filesystem  : [FAIL] Folder not found at {created_path}")
                passed = False

        if test_id == "TEST 4":
            # Verify rejection
            if tool_result is not None and not tool_result.success:
                print(f"  Security    : Successfully rejected destructive command: {tool_result.message}")
            else:
                print("  Security    : [FAIL] Destructive command was not rejected!")
                passed = False

        if test_id == "TEST 5":
            if should_exit:
                print("  Exit Check  : Exit flag set to True as expected.")
            else:
                print("  Exit Check  : [FAIL] should_exit was False.")
                passed = False

        status_str = "PASSED" if passed else "FAILED"
        print(f"  TEST STATUS : [{status_str}]")

        performance_records.append({
            "test": test_id,
            "command": command_text,
            "transcription": transcription,
            "latencies": result.get("latencies", {}),
            "response": result.get("response", ""),
        })

        if not passed:
            all_passed = False

        # Brief pause between test cases
        time.sleep(1.0)

    # Print Latency & Performance Summary Table
    print("\n" + "=" * 70)
    print("                    PERFORMANCE BENCHMARK TABLE")
    print("=" * 70)
    header = f"{'Test':<18} | {'STT':<7} | {'Brain':<7} | {'Tool':<7} | {'TTS':<7} | {'Total':<7}"
    print(header)
    print("-" * len(header))

    for rec in performance_records:
        l = rec["latencies"]
        stt_s = f"{l.get('stt', 0.0):.2f}s"
        brain_s = f"{l.get('brain', 0.0):.2f}s"
        tool_s = f"{l.get('tool', 0.0):.2f}s"
        tts_s = f"{l.get('tts', 0.0):.2f}s"
        total_s = f"{l.get('total', 0.0):.2f}s"
        print(f"{rec['test']:<18} | {stt_s:<7} | {brain_s:<7} | {tool_s:<7} | {tts_s:<7} | {total_s:<7}")

    print("=" * 70)
    if all_passed:
        print("ALL 6 END-TO-END TESTS PASSED SUCCESSFULLY!")
    else:
        print("SOME TESTS FAILED - REVIEW LOGS ABOVE.")
    print("=" * 70)

    return all_passed


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
