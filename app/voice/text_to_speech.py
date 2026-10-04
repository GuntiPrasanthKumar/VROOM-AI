"""Text-to-Speech component for VROOM AI using Piper TTS.

Converts text responses into spoken audio using lightweight, local neural TTS models.
100% private, offline, and free without cloud dependencies.
"""

import os
from pathlib import Path
import shutil
import sys
import urllib.request
from typing import Optional, Tuple
import numpy as np
import sounddevice as sd

try:
    from piper import PiperVoice
    import piper.voice
except ImportError:
    PiperVoice = None


class TextToSpeech:
    """Manages local neural speech synthesis and playback using Piper TTS."""

    BASE_DOWNLOAD_URL = "https://huggingface.co/rhasspy/piper-voices/resolve/main"

    # Supported default voices and their relative HF repository paths
    VOICE_REGISTRY = {
        "en_US-lessac-medium": "en/en_US/lessac/medium",
        "en_US-lessac-low": "en/en_US/lessac/low",
    }

    def __init__(
        self,
        voice_name: str = "en_US-lessac-medium",
        models_dir: Optional[Path] = None,
        use_cuda: bool = False,
    ) -> None:
        """Initialize Text-to-Speech engine.

        Args:
            voice_name: Voice model identifier (default: 'en_US-lessac-medium').
            models_dir: Directory where model files are stored. Defaults to '<project>/models/piper'.
            use_cuda: Whether to use CUDA acceleration in ONNX Runtime. Defaults to False (CPU).
        """
        if PiperVoice is None:
            raise ImportError(
                "piper-tts is not installed. Please install it using: pip install piper-tts"
            )

        self.voice_name = voice_name
        self.use_cuda = use_cuda

        if models_dir is None:
            self.models_dir = Path(__file__).resolve().parent.parent.parent / "models" / "piper"
        else:
            self.models_dir = Path(models_dir).resolve()

        self.model_path = self.models_dir / f"{self.voice_name}.onnx"
        self.config_path = self.models_dir / f"{self.voice_name}.onnx.json"
        self._voice: Optional[PiperVoice] = None

        self._ensure_model_available()
        self._load_voice()

    def _get_safe_espeak_dir(self) -> Path:
        """Ensure espeak-ng-data is located in an ASCII-safe path on Windows.

        Native C/C++ libraries (espeakbridge) on Windows fail when paths contain
        non-ASCII characters (such as Korean Hangul '문서' in OneDrive paths).
        This helper mirrors espeak-ng-data to %LOCALAPPDATA%/piper/espeak-ng-data.
        """
        local_app_data = Path(os.environ.get("LOCALAPPDATA", os.path.expanduser("~"))) / "piper"
        safe_espeak_dir = local_app_data / "espeak-ng-data"

        default_espeak = Path(piper.voice.ESPEAK_DATA_DIR)

        # If already cached in safe directory and valid, return it
        if safe_espeak_dir.exists() and (safe_espeak_dir / "phontab").exists():
            return safe_espeak_dir

        # Otherwise copy from venv package directory to safe local app data
        if default_espeak.exists():
            local_app_data.mkdir(parents=True, exist_ok=True)
            if safe_espeak_dir.exists():
                shutil.rmtree(safe_espeak_dir, ignore_errors=True)
            shutil.copytree(default_espeak, safe_espeak_dir)
            return safe_espeak_dir

        return default_espeak

    def _ensure_model_available(self) -> None:
        """Ensure both the ONNX model and JSON config exist locally, downloading if necessary."""
        self.models_dir.mkdir(parents=True, exist_ok=True)

        if self.model_path.exists() and self.config_path.exists():
            return

        rel_path = self.VOICE_REGISTRY.get(self.voice_name)
        if not rel_path:
            raise ValueError(
                f"Unknown voice '{self.voice_name}'. Available default voices: {list(self.VOICE_REGISTRY.keys())}"
            )

        onnx_url = f"{self.BASE_DOWNLOAD_URL}/{rel_path}/{self.voice_name}.onnx"
        json_url = f"{self.BASE_DOWNLOAD_URL}/{rel_path}/{self.voice_name}.onnx.json"

        print(f"[TTS] Voice model files missing locally in {self.models_dir}")
        print(f"[TTS] Downloading '{self.voice_name}' from Hugging Face...")

        self._download_file(json_url, self.config_path, "config")
        self._download_file(onnx_url, self.model_path, "model weights (~60 MB)")
        print(f"[TTS] Successfully downloaded voice model '{self.voice_name}'.")

    def _download_file(self, url: str, destination: Path, label: str) -> None:
        """Download a file with user-agent header and atomic replacement."""
        temp_dest = destination.with_suffix(".tmp")
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "VROOM-AI/1.0"})
            with urllib.request.urlopen(req, timeout=120) as resp, open(temp_dest, "wb") as out_file:
                while chunk := resp.read(1024 * 1024):
                    out_file.write(chunk)
            temp_dest.replace(destination)
        except Exception as exc:
            if temp_dest.exists():
                temp_dest.unlink()
            raise RuntimeError(
                f"Failed to download Piper voice {label} from {url}: {exc}"
            ) from exc

    def _load_voice(self) -> None:
        """Load the ONNX voice model into memory."""
        safe_espeak = self._get_safe_espeak_dir()
        try:
            self._voice = PiperVoice.load(
                model_path=str(self.model_path),
                config_path=str(self.config_path),
                use_cuda=self.use_cuda,
                espeak_data_dir=safe_espeak,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load Piper voice model from '{self.model_path}': {exc}"
            ) from exc

    def synthesize(self, text: str) -> Tuple[np.ndarray, int]:
        """Synthesize text into a normalized float32 audio waveform numpy array.

        Args:
            text: Text string to speak.

        Returns:
            Tuple of (audio_waveform_array, sample_rate).
        """
        if not text or not text.strip():
            return np.zeros(0, dtype=np.float32), 22050

        if self._voice is None:
            raise RuntimeError("Piper voice model is not initialized.")

        chunks = []
        sample_rate = 22050  # Default fallback
        for chunk in self._voice.synthesize(text.strip()):
            sample_rate = chunk.sample_rate
            if chunk.audio_float_array is not None and len(chunk.audio_float_array) > 0:
                chunks.append(chunk.audio_float_array)

        if not chunks:
            return np.zeros(0, dtype=np.float32), sample_rate

        full_audio = np.concatenate(chunks).astype(np.float32)
        return full_audio, sample_rate

    def speak(self, text: str, blocking: bool = True) -> bool:
        """Synthesize text and play it through the default audio output device.

        Args:
            text: Text string to speak aloud.
            blocking: If True, waits until playback completes before returning.

        Returns:
            True if playback succeeded, False otherwise.
        """
        if not text or not text.strip():
            return False

        try:
            audio_array, sample_rate = self.synthesize(text)
            if len(audio_array) == 0:
                return False

            sd.play(audio_array, samplerate=sample_rate)
            if blocking:
                sd.wait()
            return True
        except sd.PortAudioError as pa_err:
            raise RuntimeError(f"Audio playback error: {pa_err}") from pa_err
        except Exception as exc:
            raise RuntimeError(f"Error during TTS playback: {exc}") from exc
