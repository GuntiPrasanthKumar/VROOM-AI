"""VROOM AI - Microphone Capture & Audio Dimension Verification.

Demonstrates the relationship between:
duration (seconds) x sampling rate (Hz) = total samples
and validates mono channel audio buffer dimensions.
"""

import sys
import time

# Ensure safe UTF-8 output on Windows console
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass

import sounddevice as sd


def main() -> int:
    print("=" * 60)
    print("       VROOM AI - Audio Fundamentals Experiment")
    print("=" * 60)

    # 1. Identify default audio input device
    default_input_idx = sd.default.device[0]
    default_dev = sd.query_devices(default_input_idx, kind="input")
    print(f"\n[DEVICE] Input Device: [{default_input_idx}] {default_dev['name']}")

    # 2. Audio Configuration Parameters
    sample_rate = 16000  # 16,000 samples per second (16 kHz)
    duration = 3.0       # 3 seconds
    channels = 1         # 1 channel (Mono)

    # 3. Calculate expected samples based on fundamental relationship
    # Formula: expected_samples = sample_rate x duration
    expected_samples = int(sample_rate * duration)

    print("\n--- Audio Parameters ---")
    print(f"  Sampling Rate (Hz)   : {sample_rate} Hz")
    print(f"  Channels             : {channels} (Mono)")
    print(f"  Duration (seconds)   : {duration} s")
    print(f"  Expected Samples     : {expected_samples} (calculated: {sample_rate} x {duration})")

    # 4. Record audio from microphone
    print(f"\n[RECORDING] Capturing {duration} seconds of audio... Please speak or make sound.")
    start_time = time.time()

    audio_data = sd.rec(
        frames=expected_samples,
        samplerate=sample_rate,
        channels=channels,
        dtype="float32",
        device=default_input_idx,
    )

    sd.wait()  # Block until the recording completes
    elapsed_time = time.time() - start_time
    print(f"[COMPLETED] Recording finished in {elapsed_time:.2f} seconds.")

    # 5. Extract dimensions and metrics from the captured audio array
    captured_samples = len(audio_data)
    array_shape = audio_data.shape
    data_type = audio_data.dtype

    print("\n--- Captured Audio Results ---")
    print(f"  Sample Rate          : {sample_rate} Hz")
    print(f"  Number of Channels   : {channels}")
    print(f"  Recording Duration   : {duration} s")
    print(f"  Captured Samples     : {captured_samples}")
    print(f"  Audio Array Shape    : {array_shape}")
    print(f"  Audio Data Type      : {data_type}")

    # 6. Compare expected vs captured samples
    print("\n--- Verification & Comparison ---")
    print(f"  Formula Check        : {sample_rate} Hz * {duration} s = {expected_samples} samples")
    print(f"  Captured vs Expected : {captured_samples} / {expected_samples}")

    if captured_samples == expected_samples and array_shape == (expected_samples, channels):
        print("  Match Status         : EXACT MATCH (100% of expected samples received)")
        print("\n" + "=" * 60)
        print("CONFIRMATION: Audio was successfully captured and verified!")
        print("=" * 60)
        return 0
    else:
        print("  Match Status         : MISMATCH")
        print("\n" + "=" * 60)
        print("WARNING: Captured sample count did not match expected sample count.")
        print("=" * 60)
        return 1


if __name__ == "__main__":
    sys.exit(main())
