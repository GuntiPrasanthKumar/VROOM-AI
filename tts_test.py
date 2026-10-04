"""Verification test for VROOM AI Text-to-Speech (Piper TTS).

Tests local speech synthesis and audio playback:
1. Loads/downloads the lightweight Piper voice model (en_US-lessac-medium).
2. Converts text to speech waveform.
3. Plays the synthesized audio aloud through the default speaker.
4. Reports audio diagnostics and performance metrics.
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
from app.voice.text_to_speech import TextToSpeech


def main() -> int:
    print("=" * 70)
    print("       VROOM AI - Text-to-Speech Verification (Piper TTS)")
    print("=" * 70)

    # 1. Output Device Info
    try:
        default_out_idx = sd.default.device[1]
        out_dev_info = sd.query_devices(default_out_idx, kind="output")
        print(f"\n[AUDIO] Default Output Device: [{default_out_idx}] {out_dev_info['name']}")
    except Exception as exc:
        print(f"\n[WARNING] Could not query default output device: {exc}")

    # 2. Initialize Text-to-Speech Engine
    voice_model = "en_US-lessac-medium"
    print(f"\n[TTS] Initializing Piper TTS with voice: '{voice_model}'...")
    start_load = time.perf_counter()
    try:
        tts = TextToSpeech(voice_name=voice_model)
        load_time = time.perf_counter() - start_load
        print(f"[TTS] Engine loaded in        : {load_time:.2f} seconds")
    except Exception as exc:
        print(f"\n[ERROR] Failed to initialize TextToSpeech: {exc}")
        return 1

    # 3. Text to Synthesize
    test_sentence = "Hello. I am VROOM, your local desktop voice assistant."
    print(f"\n[INPUT] Text to Speak: \"{test_sentence}\"")

    # 4. Synthesize Waveform
    print("\n[SYNTHESIS] Generating audio waveform...")
    start_syn = time.perf_counter()
    try:
        audio_array, sample_rate = tts.synthesize(test_sentence)
        syn_time = time.perf_counter() - start_syn
        duration_sec = len(audio_array) / sample_rate if sample_rate > 0 else 0.0
    except Exception as exc:
        print(f"\n[ERROR] Synthesis failed: {exc}")
        return 1

    print(f"[SYNTHESIS] Status            : SUCCESS")
    print(f"[SYNTHESIS] Synthesis Time    : {syn_time:.3f} seconds")
    print(f"[SYNTHESIS] Output Sample Rate: {sample_rate} Hz (Mono)")
    print(f"[SYNTHESIS] Audio Samples     : {len(audio_array)}")
    print(f"[SYNTHESIS] Audio Duration    : {duration_sec:.2f} seconds")
    print(f"[SYNTHESIS] Waveform Shape    : {audio_array.shape} ({audio_array.dtype})")
    print(f"[SYNTHESIS] Real-Time Factor  : {syn_time / duration_sec:.2f}x (synthesis time / audio duration)")

    # 5. Audio Playback
    print("\n[PLAYBACK] Playing speech through default speaker...")
    try:
        sd.play(audio_array, samplerate=sample_rate)
        sd.wait()  # Block until audio finishes playing
        print("[PLAYBACK] Status             : SUCCESS (Audio played completely)")
    except Exception as exc:
        print(f"\n[ERROR] Audio playback failed: {exc}")
        return 1

    print("\n" + "=" * 70)
    print("STATUS: Text-to-Speech pipeline verified successfully!")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
