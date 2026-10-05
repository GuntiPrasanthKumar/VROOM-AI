"""Interactive microphone level monitor and clap tester for VROOM AI.
Records audio in real-time and prints live levels and clap detection metrics.
"""

import sys
import time
import numpy as np
import sounddevice as sd

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass

from app.voice.clap_detector import ClapDetector

print("=" * 65)
print("       VROOM AI — Real-Time Microphone & Clap Diagnostic")
print("=" * 65)
print("Testing your microphone for 12 seconds...")
print("Try:")
print("  1. CLAP your hands firmly")
print("  2. SPEAK normally ('Hey Jarvis')")
print("  3. Observe Peak amplitude, Crest factor, and Clap detection!")
print("=" * 65)

detector = ClapDetector(threshold=0.03, feedback=False)
ambient_noise = 0.001
start_time = time.time()
claps_detected = 0

with sd.InputStream(samplerate=16000, channels=1, dtype="float32", blocksize=512) as stream:
    while time.time() - start_time < 12.0:
        chunk, _ = stream.read(512)
        peak = float(np.max(np.abs(chunk)))
        rms = float(np.sqrt(np.mean(chunk ** 2)))
        crest = peak / (rms + 1e-6)
        
        is_clap, metrics = detector.process_frame(chunk)
        if is_clap:
            claps_detected += 1
            print(f"\n>>> 👏 CLAP DETECTED! (Peak: {metrics['peak']:.3f}, Crest: {metrics['crest_factor']:.1f}) <<<\n")
        elif peak > 0.015:
            bar = "#" * int(min(peak * 150, 40))
            sys.stdout.write(f"\rPeak: {peak:.4f} | RMS: {rms:.4f} | Crest: {crest:4.1f} | {bar:<40}")
            sys.stdout.flush()

print(f"\n\nDiagnostic complete! Total physical claps detected: {claps_detected}")
