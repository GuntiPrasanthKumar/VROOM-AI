"""Speech-to-Text verification test for VROOM AI.

Records 5 seconds of microphone audio and transcribes it locally using Faster-Whisper.
Displays detailed performance and timing metrics.
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

from app.voice.audio_recorder import AudioRecorder
from app.voice.speech_to_text import SpeechToText


def main() -> int:
    print("=" * 65)
    print("       VROOM AI - Speech-to-Text Verification (Faster-Whisper)")
    print("=" * 65)

    # 1. Initialize Audio Recorder
    try:
        recorder = AudioRecorder(sample_rate=16000, channels=1)
        print(f"\n[AUDIO] Input Device       : {recorder.device_info}")
        print(f"[AUDIO] Sample Rate        : {recorder.sample_rate} Hz (Mono)")
    except Exception as exc:
        print(f"\n[ERROR] Failed to initialize AudioRecorder: {exc}")
        return 1

    # 2. Initialize Faster-Whisper
    model_name = "base"
    print(f"\n[MODEL] Loading Faster-Whisper ('{model_name}' on CPU, int8)...")
    start_load = time.perf_counter()
    try:
        stt = SpeechToText(model_size=model_name, device="cpu", compute_type="int8")
        load_time = time.perf_counter() - start_load
        print(f"[MODEL] Loaded in          : {load_time:.2f} seconds")
    except Exception as exc:
        print(f"\n[ERROR] Failed to load Whisper model: {exc}")
        return 1

    # 3. Record Audio (5 seconds)
    duration_sec = 5.0
    print(f"\n[MIC] Recording {duration_sec} seconds...")
    print("      >>> Please speak now (e.g. 'Hello VROOM, this is a speech recognition test.') <<<")

    start_rec = time.perf_counter()
    try:
        audio_array = recorder.record(duration_seconds=duration_sec)
        rec_time = time.perf_counter() - start_rec
        print(f"[MIC] Captured             : {len(audio_array)} samples in {rec_time:.2f}s")
        print(f"[MIC] Amplitude range      : [{float(audio_array.min()):.4f}, {float(audio_array.max()):.4f}]")
    except Exception as exc:
        print(f"\n[ERROR] Audio capture failed: {exc}")
        return 1

    # 4. Transcribe Audio
    print("\n[STT] Transcribing audio with Faster-Whisper...")
    start_transcribe = time.perf_counter()
    try:
        text, meta = stt.transcribe(audio_array, language="en")
        transcribe_time = time.perf_counter() - start_transcribe
    except Exception as exc:
        print(f"\n[ERROR] Transcription failed: {exc}")
        return 1

    # 5. Report Results
    print("\n" + "=" * 65)
    print("                    TRANSCRIPTION RESULT")
    print("=" * 65)
    print(f"  Transcription      : \"{text}\"")
    print(f"  Detected Language  : {meta.get('language')} (confidence: {meta.get('language_probability')})")
    print(f"  Audio Duration     : {meta.get('duration')}s")
    print("\n--- Performance Metrics ---")
    print(f"  Model Size         : {model_name} (int8 quantized)")
    print(f"  Model Load Time    : {load_time:.2f}s")
    print(f"  Recording Duration : {duration_sec:.1f}s")
    print(f"  Transcription Time : {transcribe_time:.2f}s")
    print(f"  Real-Time Factor   : {transcribe_time / duration_sec:.2f}x (lower is faster)")
    print("=" * 65)

    if not text:
        print("[WARNING] Transcription is empty. If you did not speak, please retry while speaking clearly.")
    else:
        print("[SUCCESS] Speech-to-Text pipeline verified successfully!")

    return 0


if __name__ == "__main__":
    sys.exit(main())
