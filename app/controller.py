"""Activation Controller and State Machine for VROOM AI.

Coordinates the transition between low-power Wake-Word monitoring and
full conversational execution:
WAITING -> ACTIVATED -> LISTENING -> PROCESSING -> EXECUTING -> RESPONDING -> WAITING.

Enforces strict microphone ownership, prevents device contention,
and maintains the security boundaries of the Tool Router.
"""

from enum import Enum
import time
from typing import Any, Callable, Dict, Optional, Tuple
import numpy as np

from app.orchestrator import AssistantOrchestrator
from app.tools.base import ToolResult
from app.voice.wake_word import WakeWordDetector


class AssistantState(str, Enum):
    """Explicit lifecycle states for VROOM AI."""

    WAITING = "WAITING"          # Low-power monitoring of audio for wake phrase
    ACTIVATED = "ACTIVATED"      # Wake word confirmed; activation cue presented
    LISTENING = "LISTENING"      # Microphone actively capturing single user command
    PROCESSING = "PROCESSING"    # STT transcription and LLM semantic understanding
    EXECUTING = "EXECUTING"      # Tool Router vetting and safe tool invocation
    RESPONDING = "RESPONDING"    # Piper TTS speaking natural language feedback


class ActivationController:
    """Manages the state machine and coordinates Wake-Word with the Assistant pipeline."""

    def __init__(
        self,
        orchestrator: Optional[AssistantOrchestrator] = None,
        wake_detector: Optional[WakeWordDetector] = None,
        wake_model_name: str = "hey_jarvis",
        wake_threshold: float = 0.5,
        dev_mode: bool = True,
        on_state_change: Optional[Callable[[AssistantState, AssistantState], None]] = None,
    ) -> None:
        """Initialize the activation controller.

        Args:
            orchestrator: AssistantOrchestrator instance (manages STT, LLM, Router, TTS).
            wake_detector: WakeWordDetector instance (manages OpenWakeWord).
            wake_model_name: Default wake-word model name ('hey_jarvis').
            wake_threshold: Confidence threshold for wake-word activation (0.0 to 1.0).
            dev_mode: Whether to print diagnostic development logs.
            on_state_change: Optional callback invoked on state transitions.
        """
        self.dev_mode = dev_mode
        self.on_state_change = on_state_change

        # Initialize underlying subsystems
        self.orchestrator = orchestrator or AssistantOrchestrator(dev_mode=dev_mode)
        self.wake_detector = wake_detector or WakeWordDetector(
            model_name=wake_model_name,
            threshold=wake_threshold,
        )

        # Initial state
        self._state: AssistantState = AssistantState.WAITING

    @property
    def state(self) -> AssistantState:
        """Return the current lifecycle state."""
        return self._state

    def _set_state(self, new_state: AssistantState) -> None:
        """Transition to a new state and notify listeners."""
        old_state = self._state
        self._state = new_state
        if self.dev_mode:
            print(f"\n[STATE TRANSITION] {old_state.value} ──> {new_state.value}")
        if self.on_state_change is not None:
            self.on_state_change(old_state, new_state)

    def _play_activation_cue(self) -> None:
        """Provide a brief, non-intrusive activation cue to indicate readiness."""
        print("[ACTIVATED] Wake word detected! Listening for command...")
        # Brief friendly spoken cue
        try:
            self.orchestrator.tts.speak("Listening...", blocking=True)
        except Exception:
            # Fall back to silent/visual cue if audio device is unavailable
            pass

    def run_cycle(
        self,
        wake_timeout: Optional[float] = None,
        command_duration: float = 4.0,
    ) -> Dict[str, Any]:
        """Execute one complete lifecycle cycle:
        WAITING -> ACTIVATED -> LISTENING -> PROCESSING -> EXECUTING -> RESPONDING -> WAITING.

        Returns:
            Dictionary containing timing metrics, transcription, result, and exit status.
        """
        metrics: Dict[str, float] = {}
        t_cycle_start = time.perf_counter()

        # =====================================================================
        # STATE 1: WAITING (Microphone owned by WakeWordDetector)
        # =====================================================================
        self._set_state(AssistantState.WAITING)
        if self.dev_mode:
            print(f"[WAITING] Monitoring audio for wake word '{self.wake_detector.model_name}'...")

        t_wake_start = time.perf_counter()
        wake_detected = self.wake_detector.listen(timeout_seconds=wake_timeout)
        metrics["wake_detection"] = round(time.perf_counter() - t_wake_start, 3)

        if not wake_detected:
            if self.dev_mode:
                print(f"[WAITING] Wake-word listening timed out after {wake_timeout}s.")
            return {
                "wake_detected": False,
                "should_exit": False,
                "metrics": metrics,
                "final_state": self._state.value,
            }

        # =====================================================================
        # STATE 2: ACTIVATED (Microphone released by WakeWordDetector)
        # =====================================================================
        self._set_state(AssistantState.ACTIVATED)
        t_act_start = time.perf_counter()
        self._play_activation_cue()
        metrics["activation_cue"] = round(time.perf_counter() - t_act_start, 3)

        # =====================================================================
        # STATE 3: LISTENING (Microphone owned by AudioRecorder)
        # =====================================================================
        self._set_state(AssistantState.LISTENING)
        print(f"[LISTENING] Speak command now ({command_duration:.1f}s)...")

        t_rec_start = time.perf_counter()
        command_audio = self.orchestrator.recorder.record(duration_seconds=command_duration)
        metrics["command_recording"] = round(time.perf_counter() - t_rec_start, 3)

        # =====================================================================
        # STATE 4: PROCESSING (STT Transcription & LLM Understanding)
        # =====================================================================
        self._set_state(AssistantState.PROCESSING)

        # Step 4a: Speech-to-Text
        t_stt_start = time.perf_counter()
        try:
            transcription, stt_meta = self.orchestrator.stt.transcribe(command_audio)
        except Exception as exc:
            transcription = ""
            stt_meta = {"error": str(exc)}
            if self.dev_mode:
                print(f"[STT ERROR] Failed to transcribe: {exc}")

        metrics["stt"] = round(time.perf_counter() - t_stt_start, 3)
        clean_text = transcription.strip() if transcription else ""

        if self.dev_mode:
            print(f"[PROCESSING] Transcription: \"{clean_text}\" (took {metrics['stt']}s)")

        # Handle Timeout / Silence: No speech detected in recording
        if not clean_text:
            if self.dev_mode:
                print("[PROCESSING] No speech detected in command recording (silence/timeout).")

            self._set_state(AssistantState.RESPONDING)
            silence_response = "I didn't hear anything. Returning to standby."
            t_tts_start = time.perf_counter()
            self.orchestrator._speak(silence_response)
            metrics["tts"] = round(time.perf_counter() - t_tts_start, 3)
            metrics["total_interaction"] = round(time.perf_counter() - t_cycle_start, 3)

            # Return to WAITING
            self._set_state(AssistantState.WAITING)
            return {
                "wake_detected": True,
                "transcription": "",
                "intent": "timeout",
                "entity": None,
                "tool_result": None,
                "response": silence_response,
                "should_exit": False,
                "metrics": metrics,
                "final_state": self._state.value,
            }

        # Step 4b: LLM Command Understanding
        t_brain_start = time.perf_counter()
        intent = "unknown"
        entity = None
        try:
            llm_res = self.orchestrator.llm.parse_command(clean_text)
            intent = llm_res.get("intent", "unknown")
            entity = llm_res.get("entity")
        except Exception as exc:
            if self.dev_mode:
                print(f"[LLM WARN] LLM failed ({exc}). Using deterministic fallback.")
            parsed = self.orchestrator.parser.parse(clean_text)
            intent = parsed.intent
            entity = parsed.entity

        metrics["llm"] = round(time.perf_counter() - t_brain_start, 3)

        if self.dev_mode:
            print(f"[PROCESSING] Intent: '{intent}', Entity: '{entity}' (took {metrics['llm']}s)")

        # Handle Exit Assistant command
        clean_lower = clean_text.lower().rstrip(".!?,;:").strip()
        should_exit = (
            intent == "exit_assistant"
            or clean_lower in ("exit", "quit", "stop", "bye", "goodbye")
            or clean_lower.startswith(("quit", "exit", "stop", "shut down", "power down", "bye"))
        )
        if should_exit:
            intent = "exit_assistant"
            self._set_state(AssistantState.RESPONDING)
            exit_response = "Goodbye! Powering down."
            t_tts_start = time.perf_counter()
            self.orchestrator._speak(exit_response)
            metrics["tts"] = round(time.perf_counter() - t_tts_start, 3)
            metrics["total_interaction"] = round(time.perf_counter() - t_cycle_start, 3)

            return {
                "wake_detected": True,
                "transcription": clean_text,
                "intent": "exit_assistant",
                "entity": None,
                "tool_result": None,
                "response": exit_response,
                "should_exit": True,
                "metrics": metrics,
                "final_state": self._state.value,
            }

        # =====================================================================
        # STATE 5: EXECUTING (Tool Router Vetting & Tool Invocation)
        # =====================================================================
        self._set_state(AssistantState.EXECUTING)
        t_tool_start = time.perf_counter()
        tool_result = self.orchestrator.router.route_and_execute({"intent": intent, "entity": entity})
        metrics["tool_execution"] = round(time.perf_counter() - t_tool_start, 3)

        if self.dev_mode:
            status_lbl = "SUCCESS" if tool_result.success else "REJECTED/FAILED"
            print(f"[EXECUTING] Tool: {intent} -> {status_lbl} ({tool_result.message})")

        # =====================================================================
        # STATE 6: RESPONDING (Natural Language Formatting & Piper TTS)
        # =====================================================================
        self._set_state(AssistantState.RESPONDING)
        response_text = self.orchestrator._generate_natural_response(intent, entity, tool_result)

        t_tts_start = time.perf_counter()
        if self.dev_mode:
            print(f"[RESPONDING] Speaking: \"{response_text}\"")

        self.orchestrator._speak(response_text)
        metrics["tts"] = round(time.perf_counter() - t_tts_start, 3)
        metrics["total_interaction"] = round(time.perf_counter() - t_cycle_start, 3)

        # =====================================================================
        # CYCLE COMPLETE: Return to WAITING
        # =====================================================================
        self._set_state(AssistantState.WAITING)

        return {
            "wake_detected": True,
            "transcription": clean_text,
            "intent": intent,
            "entity": entity,
            "tool_result": tool_result,
            "response": response_text,
            "should_exit": False,
            "metrics": metrics,
            "final_state": self._state.value,
        }

    def execute_with_audio(
        self,
        wake_audio: Optional[np.ndarray],
        command_audio: Optional[np.ndarray],
    ) -> Dict[str, Any]:
        """Execute a simulated lifecycle cycle with injected audio arrays for testing.

        Args:
            wake_audio: 16 kHz audio array to feed into WakeWordDetector.
            command_audio: 16 kHz audio array to feed into SpeechToText.

        Returns:
            Dictionary containing execution metadata, state transitions, and metrics.
        """
        metrics: Dict[str, float] = {}
        t_cycle_start = time.perf_counter()

        # Step 1: WAITING
        self._set_state(AssistantState.WAITING)
        t_wake_start = time.perf_counter()
        wake_detected = False

        if wake_audio is not None and len(wake_audio) > 0:
            chunk_size = self.wake_detector.REQUIRED_FRAME_SIZE
            self.wake_detector.reset()
            for i in range(0, len(wake_audio) - chunk_size, chunk_size):
                chunk = wake_audio[i : i + chunk_size]
                is_det, _, _ = self.wake_detector.process_frame(chunk)
                if is_det:
                    wake_detected = True
                    break

        metrics["wake_detection"] = round(time.perf_counter() - t_wake_start, 3)

        if not wake_detected:
            return {
                "wake_detected": False,
                "should_exit": False,
                "metrics": metrics,
                "final_state": self._state.value,
            }

        # Step 2: ACTIVATED
        self._set_state(AssistantState.ACTIVATED)
        t_act_start = time.perf_counter()
        metrics["activation_cue"] = round(time.perf_counter() - t_act_start, 3)

        # Step 3: LISTENING
        self._set_state(AssistantState.LISTENING)
        metrics["command_recording"] = 0.0  # Injected audio

        # Step 4: PROCESSING
        self._set_state(AssistantState.PROCESSING)
        t_stt_start = time.perf_counter()

        if command_audio is not None and len(command_audio) > 0:
            try:
                transcription, _ = self.orchestrator.stt.transcribe(command_audio)
            except Exception as exc:
                transcription = ""
        else:
            transcription = ""

        metrics["stt"] = round(time.perf_counter() - t_stt_start, 3)
        clean_text = transcription.strip()

        # Handle Timeout / Silence
        if not clean_text:
            self._set_state(AssistantState.RESPONDING)
            silence_resp = "I didn't hear anything. Returning to standby."
            t_tts = time.perf_counter()
            self.orchestrator._speak(silence_resp)
            metrics["tts"] = round(time.perf_counter() - t_tts, 3)
            metrics["total_interaction"] = round(time.perf_counter() - t_cycle_start, 3)

            self._set_state(AssistantState.WAITING)
            return {
                "wake_detected": True,
                "transcription": "",
                "intent": "timeout",
                "entity": None,
                "tool_result": None,
                "response": silence_resp,
                "should_exit": False,
                "metrics": metrics,
                "final_state": self._state.value,
            }

        # LLM Understanding
        t_brain = time.perf_counter()
        try:
            llm_res = self.orchestrator.llm.parse_command(clean_text)
            intent = llm_res.get("intent", "unknown")
            entity = llm_res.get("entity")
        except Exception:
            parsed = self.orchestrator.parser.parse(clean_text)
            intent = parsed.intent
            entity = parsed.entity

        metrics["llm"] = round(time.perf_counter() - t_brain, 3)

        # Handle Exit
        clean_lower = clean_text.lower().rstrip(".!?,;:").strip()
        should_exit = (
            intent == "exit_assistant"
            or clean_lower in ("exit", "quit", "stop", "bye", "goodbye")
            or clean_lower.startswith(("quit", "exit", "stop", "shut down", "power down", "bye"))
        )
        if should_exit:
            intent = "exit_assistant"
            self._set_state(AssistantState.RESPONDING)
            exit_resp = "Goodbye! Powering down."
            t_tts = time.perf_counter()
            self.orchestrator._speak(exit_resp)
            metrics["tts"] = round(time.perf_counter() - t_tts, 3)
            metrics["total_interaction"] = round(time.perf_counter() - t_cycle_start, 3)
            return {
                "wake_detected": True,
                "transcription": clean_text,
                "intent": "exit_assistant",
                "entity": None,
                "tool_result": None,
                "response": exit_resp,
                "should_exit": True,
                "metrics": metrics,
                "final_state": self._state.value,
            }

        # Step 5: EXECUTING
        self._set_state(AssistantState.EXECUTING)
        t_tool = time.perf_counter()
        tool_result = self.orchestrator.router.route_and_execute({"intent": intent, "entity": entity})
        metrics["tool_execution"] = round(time.perf_counter() - t_tool, 3)

        # Step 6: RESPONDING
        self._set_state(AssistantState.RESPONDING)
        response_text = self.orchestrator._generate_natural_response(intent, entity, tool_result)
        t_tts = time.perf_counter()
        self.orchestrator._speak(response_text)
        metrics["tts"] = round(time.perf_counter() - t_tts, 3)
        metrics["total_interaction"] = round(time.perf_counter() - t_cycle_start, 3)

        # Return to WAITING
        self._set_state(AssistantState.WAITING)

        return {
            "wake_detected": True,
            "transcription": clean_text,
            "intent": intent,
            "entity": entity,
            "tool_result": tool_result,
            "response": response_text,
            "should_exit": False,
            "metrics": metrics,
            "final_state": self._state.value,
        }

    def run(self, command_duration: float = 4.0) -> None:
        """Run the continuous wake-word activation loop until interrupted or exit."""
        print("=" * 65)
        print("      VROOM AI — Continuous Wake-Word Activation Mode")
        print("=" * 65)
        print(f"Wake Phrase : 'Hey Jarvis' (Model: {self.wake_detector.model_name})")
        print("Status      : Waiting for wake word... Speak to activate.")
        print("=" * 65)

        try:
            while True:
                cycle_outcome = self.run_cycle(command_duration=command_duration)
                if cycle_outcome.get("should_exit"):
                    print("\n[CONTROLLER] Exit command processed. Shutting down gracefully.")
                    break
        except KeyboardInterrupt:
            print("\n[CONTROLLER] Interrupted by user. Exiting cleanly.")
            self._set_state(AssistantState.WAITING)
            self.orchestrator._speak("Goodbye.")
