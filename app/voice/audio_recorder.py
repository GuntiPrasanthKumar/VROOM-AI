"""Audio recorder component for VROOM AI.

Captures mono audio from the system's default input device at 16,000 Hz,
suitable for speech recognition models.
"""

from typing import Optional, Tuple
import numpy as np
import sounddevice as sd


class AudioRecorder:
    """Manages microphone input capture using sounddevice."""

    def __init__(self, sample_rate: int = 16000, channels: int = 1):
        self.sample_rate = sample_rate
        self.channels = channels
        self._device_index, self._device_name = self._detect_default_device()

    def _detect_default_device(self) -> Tuple[int, str]:
        """Detect and return the default input device index and name."""
        try:
            default_idx = sd.default.device[0]
            if default_idx == -1:
                # Fallback: find first device with input channels
                devices = sd.query_devices()
                for idx, dev in enumerate(devices):
                    if dev["max_input_channels"] > 0:
                        return idx, dev["name"]
                raise RuntimeError("No audio input devices found on this system.")

            dev_info = sd.query_devices(default_idx, kind="input")
            return default_idx, dev_info["name"]
        except Exception as exc:
            raise RuntimeError(f"Failed to detect audio input device: {exc}") from exc

    @property
    def device_info(self) -> str:
        """Return formatted device identification string."""
        return f"[{self._device_index}] {self._device_name}"

    def record(self, duration_seconds: float) -> np.ndarray:
        """Record audio for a specified duration and return a 1D float32 numpy array.

        Args:
            duration_seconds: Recording duration in seconds.

        Returns:
            1D numpy array of shape (num_samples,) containing audio normalized in [-1.0, 1.0].
        """
        if duration_seconds <= 0:
            raise ValueError(f"Duration must be positive, got {duration_seconds}")

        total_frames = int(self.sample_rate * duration_seconds)

        try:
            # sd.rec records asynchronously into a numpy buffer
            raw_audio = sd.rec(
                frames=total_frames,
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype="float32",
                device=self._device_index,
            )
            # Block until recording finishes
            sd.wait()
        except sd.PortAudioError as pa_err:
            raise RuntimeError(
                f"PortAudio error during recording on device {self.device_info}: {pa_err}"
            ) from pa_err
        except Exception as exc:
            raise RuntimeError(f"Unexpected error during audio capture: {exc}") from exc

        # Flatten shape (frames, 1) -> (frames,) as expected by Whisper
        audio_1d = raw_audio.flatten().astype(np.float32)
        return audio_1d
