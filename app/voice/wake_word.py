"""Wake-word detection component for VROOM AI using OpenWakeWord.

Provides continuous, lightweight, 100% local audio monitoring for specific
activation phrases without sending audio to cloud servers or running full STT.
"""

import time
from typing import Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import sounddevice as sd

try:
    import openwakeword
    from openwakeword.model import Model
except ImportError:
    openwakeword = None
    Model = None


class WakeWordDetector:
    """Manages local wake-word detection using OpenWakeWord models."""

    # Standard audio specification required by OpenWakeWord models
    REQUIRED_SAMPLE_RATE: int = 16000
    REQUIRED_FRAME_SIZE: int = 1280  # 1280 samples = 80ms at 16,000 Hz

    def __init__(
        self,
        model_name: str = "hey_jarvis",
        threshold: float = 0.22,
        debounce_seconds: float = 1.5,
        device_index: Optional[int] = None,
    ) -> None:
        """Initialize the wake-word detector.

        Args:
            model_name: Pre-trained wake-word model identifier (e.g. 'hey_jarvis', 'alexa')
                        or path to a custom ONNX wake-word model.
            threshold: Confidence score cutoff between 0.0 and 1.0 to trigger activation.
            debounce_seconds: Minimum delay between consecutive activations to prevent
                              double-triggering from the same spoken utterance.
            device_index: Optional sounddevice input device index. None uses default input.
        """
        if Model is None:
            raise ImportError(
                "openwakeword is not installed. Please install it using: pip install openwakeword"
            )

        self.model_name = model_name
        self.threshold = threshold
        self.debounce_seconds = debounce_seconds
        self.device_index = device_index
        self._last_activation_time: float = 0.0

        # Load the wake-word model via ONNX runtime
        self._load_model()

    def _load_model(self) -> None:
        """Instantiate the OpenWakeWord model."""
        try:
            self.model = Model(
                wakeword_models=[self.model_name],
                inference_framework="onnx",
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load OpenWakeWord model '{self.model_name}': {exc}"
            ) from exc

    def reset(self) -> None:
        """Reset internal feature buffers to clear previous acoustic context."""
        if hasattr(self, "model") and self.model is not None:
            self.model.reset()

    def process_frame(self, frame: np.ndarray) -> Tuple[bool, float, Dict[str, float]]:
        """Process a single 80ms audio frame and evaluate wake-word activation.

        Args:
            frame: NumPy array of audio samples (expected 1280 samples at 16 kHz).
                   Can be 1D or 2D (samples, 1), and float32 [-1, 1] or int16 [-32768, 32767].

        Returns:
            Tuple of:
              - is_detected (bool): True if confidence exceeds threshold and debounce has elapsed.
              - score (float): Confidence score for the active wake-word model.
              - all_scores (dict): Dictionary mapping model names to raw confidence scores.
        """
        if not isinstance(frame, np.ndarray):
            raise TypeError(f"Audio frame must be a numpy.ndarray, got {type(frame)}")

        # Flatten if 2D (e.g. shape (1280, 1))
        if frame.ndim > 1:
            frame = frame.flatten()

        # Check sample count
        if len(frame) != self.REQUIRED_FRAME_SIZE:
            # If slightly mismatched, slice or pad to expected frame size
            if len(frame) > self.REQUIRED_FRAME_SIZE:
                frame = frame[: self.REQUIRED_FRAME_SIZE]
            else:
                frame = np.pad(frame, (0, self.REQUIRED_FRAME_SIZE - len(frame)))

        # Convert float32 [-1.0, 1.0] to int16 PCM [-32768, 32767]
        if frame.dtype == np.float32 or frame.dtype == np.float64:
            frame_int16 = (np.clip(frame, -1.0, 1.0) * 32767.0).astype(np.int16)
        else:
            frame_int16 = frame.astype(np.int16)

        # Run inference
        all_scores: Dict[str, float] = self.model.predict(frame_int16)

        # Retrieve confidence score for our selected model
        score: float = all_scores.get(self.model_name, 0.0)

        # Evaluate detection and debounce cooldown
        now = time.time()
        is_detected = False
        if score >= self.threshold:
            if (now - self._last_activation_time) >= self.debounce_seconds:
                is_detected = True
                self._last_activation_time = now

        return is_detected, score, all_scores

    def listen(
        self,
        timeout_seconds: Optional[float] = None,
        on_frame: Optional[Callable[[float], None]] = None,
    ) -> bool:
        """Stream audio from the microphone and block until the wake word is detected or timeout.

        Args:
            timeout_seconds: Optional max seconds to listen before giving up. None listens indefinitely.
            on_frame: Optional callback receiving current confidence score every 80ms.

        Returns:
            True if wake word was detected, False if timeout was reached or stopped.
        """
        start_time = time.time()
        self.reset()
        print(f"[WAKE] Listening... (model: '{self.model_name}', threshold: {self.threshold:.2f})")

        try:
            with sd.InputStream(
                samplerate=self.REQUIRED_SAMPLE_RATE,
                channels=1,
                dtype="int16",
                blocksize=self.REQUIRED_FRAME_SIZE,
                device=self.device_index,
            ) as stream:
                while True:
                    # Read exactly 1280 samples (80 ms) synchronously from microphone
                    chunk, overflow = stream.read(self.REQUIRED_FRAME_SIZE)

                    detected, score, _ = self.process_frame(chunk)

                    if on_frame is not None:
                        on_frame(score)

                    if detected:
                        print(f"\n[WAKE] Detected (confidence: {score:.2f})")
                        return True

                    if timeout_seconds is not None:
                        if (time.time() - start_time) >= timeout_seconds:
                            return False

        except sd.PortAudioError as pa_err:
            raise RuntimeError(f"Audio input error during wake-word listening: {pa_err}") from pa_err
        except Exception as exc:
            raise RuntimeError(f"Unexpected error in wake-word detection loop: {exc}") from exc
