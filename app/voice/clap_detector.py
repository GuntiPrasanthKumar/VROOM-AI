"""Acoustic Clap Detection component for VROOM AI.

Provides continuous, lightweight, 100% local audio monitoring for sharp
transient sound spikes (claps) to serve as the initial physical activation gate.
Uses a two-frame onset-and-decay analysis to cleanly distinguish claps from speech.
"""

import time
from typing import Callable, Dict, Optional, Tuple
import numpy as np
import sounddevice as sd


class ClapDetector:
    """Detects sharp acoustic transient energy spikes (claps) in real time."""

    SAMPLE_RATE: int = 16000
    FRAME_SIZE: int = 512  # 512 samples = 32ms at 16,000 Hz

    def __init__(
        self,
        threshold: float = 0.40,
        min_crest_factor: float = 3.8,
        min_onset_ratio: float = 10.0,
        max_decay_peak: float = 0.28,
        debounce_seconds: float = 0.6,
        device_index: Optional[int] = None,
    ) -> None:
        """Initialize the clap detector.

        Args:
            threshold: Minimum peak amplitude [0.0 to 1.0] for the clap candidate.
            min_crest_factor: Minimum peak-to-RMS ratio indicating impulsive transient.
            min_onset_ratio: Minimum ratio of candidate peak to preceding frame RMS.
            max_decay_peak: Maximum peak amplitude permitted in the subsequent frame
                            (claps decay within 30-50ms; speech vowels sustain energy).
            debounce_seconds: Minimum delay between consecutive claps to prevent double triggers.
            device_index: Optional sounddevice input device index. None uses default input.
        """
        self.threshold = threshold
        self.min_crest_factor = min_crest_factor
        self.min_onset_ratio = min_onset_ratio
        self.max_decay_peak = max_decay_peak
        self.debounce_seconds = debounce_seconds
        self.device_index = device_index

        self._last_clap_time: float = 0.0
        self._prev_rms: float = 0.005
        self._candidate_pending: bool = False
        self._candidate_metrics: Dict[str, float] = {}

    def reset(self) -> None:
        """Reset internal history and candidate state."""
        self._last_clap_time = 0.0
        self._prev_rms = 0.005
        self._candidate_pending = False
        self._candidate_metrics = {}

    def process_frame(self, frame: np.ndarray) -> Tuple[bool, Dict[str, float]]:
        """Evaluate a single 32ms audio frame using onset-and-decay verification.

        Args:
            frame: 1D or 2D NumPy array of audio samples (float32 [-1.0, 1.0]).

        Returns:
            Tuple of (is_clap_detected, metrics_dict).
        """
        if not isinstance(frame, np.ndarray):
            raise TypeError(f"Audio frame must be a numpy.ndarray, got {type(frame)}")

        if frame.ndim > 1:
            frame = frame.flatten()

        # Convert int16 to float32 normalized if needed
        if frame.dtype == np.int16:
            frame_float = frame.astype(np.float32) / 32768.0
        elif frame.dtype in (np.float32, np.float64):
            frame_float = frame.astype(np.float32)
        else:
            frame_float = frame.astype(np.float32)

        peak_amp = float(np.max(np.abs(frame_float))) if len(frame_float) > 0 else 0.0
        rms_energy = float(np.sqrt(np.mean(frame_float ** 2))) if len(frame_float) > 0 else 0.0
        crest_factor = peak_amp / (rms_energy + 1e-6)
        onset_ratio = peak_amp / (self._prev_rms + 1e-5)

        now = time.time()
        is_clap = False
        metrics = {
            "peak": round(peak_amp, 4),
            "rms": round(rms_energy, 4),
            "crest_factor": round(crest_factor, 2),
            "onset_ratio": round(onset_ratio, 2),
            "prev_rms": round(self._prev_rms, 4),
        }

        # Step 2: If a candidate was pending from previous frame, verify decay!
        if self._candidate_pending:
            self._candidate_pending = False
            # If the current frame decayed as expected for a transient clap
            if peak_amp <= self.max_decay_peak:
                if (now - self._last_clap_time) >= self.debounce_seconds:
                    is_clap = True
                    self._last_clap_time = now
                    metrics["confirmed"] = True

        # Step 1: Check if the current frame is an onset candidate
        if (
            peak_amp >= self.threshold
            and crest_factor >= self.min_crest_factor
            and onset_ratio >= self.min_onset_ratio
            and self._prev_rms < 0.06
        ):
            self._candidate_pending = True
            self._candidate_metrics = metrics

        self._prev_rms = rms_energy
        return is_clap, metrics

    def listen(
        self,
        timeout_seconds: Optional[float] = None,
        on_frame: Optional[Callable[[Dict[str, float]], None]] = None,
    ) -> bool:
        """Stream microphone audio and block until a clap is detected or timeout expires.

        Synchronously acquires and releases the microphone stream.

        Args:
            timeout_seconds: Maximum time in seconds to wait for a clap. None waits indefinitely.
            on_frame: Optional callback invoked on each 32ms frame with metrics.

        Returns:
            True if clap detected, False if timeout reached.
        """
        start_time = time.time()
        self.reset()

        try:
            with sd.InputStream(
                samplerate=self.SAMPLE_RATE,
                channels=1,
                dtype="float32",
                blocksize=self.FRAME_SIZE,
                device=self.device_index,
            ) as stream:
                while True:
                    frame, overflow = stream.read(self.FRAME_SIZE)
                    is_clap, metrics = self.process_frame(frame)

                    if on_frame is not None:
                        on_frame(metrics)

                    if is_clap:
                        return True

                    if timeout_seconds is not None:
                        if (time.time() - start_time) >= timeout_seconds:
                            return False

        except sd.PortAudioError as pa_err:
            raise RuntimeError(f"Audio input error during clap detection: {pa_err}") from pa_err
        except Exception as exc:
            raise RuntimeError(f"Unexpected error in clap detection loop: {exc}") from exc
