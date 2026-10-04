"""Speech-to-Text component for VROOM AI using Faster-Whisper.

Performs fast, local transcription of 16 kHz audio without cloud API calls.
"""

from typing import Dict, Optional, Tuple
import numpy as np


class SpeechToText:
    """Wrapper around Faster-Whisper for local audio transcription."""

    def __init__(
        self,
        model_size: str = "base",
        device: str = "cpu",
        compute_type: str = "int8",
    ):
        """Initialize the Faster-Whisper model.

        Args:
            model_size: Model size ('tiny', 'base', 'small', 'medium', 'large-v3').
                        'base' (~74M params) offers a superior balance of English
                        command accuracy and fast local CPU inference (~250MB RAM).
            device: 'cpu' or 'cuda'. Defaults to 'cpu'.
            compute_type: Quantization mode. 'int8' is optimal for CPU inference speed.
        """
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self._model = None
        self._load_model()

    def _load_model(self) -> None:
        """Load the Whisper model using Faster-Whisper."""
        try:
            from faster_whisper import WhisperModel
        except ImportError as err:
            raise ImportError(
                "faster-whisper is not installed. Install it via: pip install faster-whisper"
            ) from err

        try:
            # CTranslate2 handles model download and local caching automatically
            self._model = WhisperModel(
                self.model_size,
                device=self.device,
                compute_type=self.compute_type,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load Faster-Whisper model '{self.model_size}' "
                f"on device '{self.device}' with compute_type '{self.compute_type}': {exc}"
            ) from exc

    def transcribe(
        self,
        audio_array: np.ndarray,
        language: Optional[str] = "en",
    ) -> Tuple[str, Dict[str, any]]:
        """Transcribe a 1D float32 NumPy array sampled at 16,000 Hz.

        Args:
            audio_array: 1-dimensional float32 NumPy array of audio samples.
            language: Target language code ('en' for English).

        Returns:
            Tuple of (transcribed_text, metadata_dict).
        """
        if self._model is None:
            raise RuntimeError("Whisper model is not initialized.")

        if not isinstance(audio_array, np.ndarray):
            raise TypeError(f"audio_array must be a numpy.ndarray, got {type(audio_array)}")

        if audio_array.ndim != 1:
            raise ValueError(f"Expected 1D audio array, got shape {audio_array.shape}")

        if audio_array.dtype != np.float32:
            audio_array = audio_array.astype(np.float32)

        try:
            # transcribe returns a generator of segments and an Info object
            segments, info = self._model.transcribe(
                audio_array,
                beam_size=5,
                language=language,
                vad_filter=True,  # Voice Activity Detection to skip silence
            )

            text_chunks = [segment.text.strip() for segment in segments]
            full_text = " ".join(text_chunks).strip()

            metadata = {
                "language": info.language,
                "language_probability": round(info.language_probability, 4),
                "duration": round(info.duration, 2),
            }
            return full_text, metadata
        except Exception as exc:
            raise RuntimeError(f"Error during transcription inference: {exc}") from exc
