"""Activation Controller and State Machine for VROOM AI.

Coordinates the staged activation sequence:
WAITING_FOR_CLAP -> WAITING_FOR_WAKE_WORD -> ACTIVATED -> LISTENING -> PROCESSING -> EXECUTING -> RESPONDING -> WAITING_FOR_CLAP.

Enforces strict sequential microphone ownership to avoid audio device conflicts,
handles activation timeouts, and preserves the Tool Router security boundary.
"""

from enum import Enum
import time
from typing import Any, Callable, Dict, Optional, Tuple
import numpy as np

from app.orchestrator import AssistantOrchestrator
from app.tools.base import ToolResult
from app.voice.clap_detector import ClapDetector
from app.voice.wake_word import WakeWordDetector


class AssistantState(str, Enum):
    """Explicit lifecycle states for VROOM AI."""

    WAITING_FOR_CLAP = "WAITING_FOR_CLAP"              # Standby: Monitoring for physical acoustic clap
    WAITING_FOR_WAKE_WORD = "WAITING_FOR_WAKE_WORD"    # Gated: Monitoring for spoken wake phrase with timeout
    ACTIVATED = "ACTIVATED"                            # Double-gate confirmed: Providing activation cue
    LISTENING = "LISTENING"                            # Microphone recording user command
    PROCESSING = "PROCESSING"                          # Whisper STT & Ollama LLM understanding
    EXECUTING = "EXECUTING"                            # Tool Router validation and execution
    RESPONDING = "RESPONDING"                          # Piper TTS speaking output


class ActivationController:
    """Manages the two-stage (Clap + Wake Word) activation state machine."""

    def __init__(
        self,
        orchestrator: Optional[AssistantOrchestrator] = None,
        clap_detector: Optional[ClapDetector] = None,
        wake_detector: Optional[WakeWordDetector] = None,
        wake_model_name: str = "hey_jarvis",
        wake_threshold: float = 0.42,
        wake_timeout: float = 6.0,
        mode: str = "staged",
        dev_mode: bool = True,
        on_state_change: Optional[Callable[[AssistantState, AssistantState], None]] = None,
    ) -> None:
        """Initialize the activation controller.

        Args:
            orchestrator: AssistantOrchestrator instance (manages STT, LLM, Router, TTS).
            clap_detector: ClapDetector instance (evaluates transient acoustic spikes).
            wake_detector: WakeWordDetector instance (manages OpenWakeWord).
            wake_model_name: Default wake-word model name ('hey_jarvis').
            wake_threshold: Confidence threshold for wake-word activation (0.0 to 1.0).
            wake_timeout: Time in seconds to wait for the wake phrase after a clap before timing out.
            mode: Activation mode - 'staged' (Clap + Wake phrase) or 'wake_only' (Direct wake phrase).
            dev_mode: Whether to print diagnostic development logs.
            on_state_change: Optional callback invoked on state transitions.
        """
        self.dev_mode = dev_mode
        self.mode = mode.lower().strip()
        self.wake_timeout = wake_timeout
        self.on_state_change = on_state_change

        # Initialize underlying subsystems
        self.orchestrator = orchestrator or AssistantOrchestrator(dev_mode=dev_mode)
        self.clap_detector = clap_detector or ClapDetector(feedback=dev_mode)
        self.wake_detector = wake_detector or WakeWordDetector(
            model_name=wake_model_name,
            threshold=wake_threshold,
        )

        # Initial state
        self._state: AssistantState = (
            AssistantState.WAITING_FOR_WAKE_WORD if self.mode == "wake_only" else AssistantState.WAITING_FOR_CLAP
        )

    @property
    def state(self) -> AssistantState:
        """Return the current lifecycle state."""
        return self._state

    def _set_state(self, new_state: AssistantState) -> None:
        """Transition to a new state and notify listeners."""
        old_state = self._state
        self._state = new_state
        if self.dev_mode:
            print(f"\n[STATE] {old_state.value} --> {new_state.value}")
        if self.on_state_change is not None:
            self.on_state_change(old_state, new_state)

    def _play_activation_cue(self) -> None:
        """Provide a brief activation cue indicating VROOM is listening for a command."""
        print("[STATE] VROOM activated. Listening for command...")
        try:
            self.orchestrator.tts.speak("Listening...", blocking=True)
        except Exception:
            pass

    def run_cycle(
        self,
        clap_timeout: Optional[float] = None,
        wake_timeout: Optional[float] = None,
        command_duration: float = 4.0,
    ) -> Dict[str, Any]:
        """Execute one complete lifecycle cycle according to the configured mode.

        In 'staged' mode:
        WAITING_FOR_CLAP -> WAITING_FOR_WAKE_WORD -> ACTIVATED -> LISTENING -> PROCESSING -> EXECUTING -> RESPONDING -> WAITING_FOR_CLAP.

        In 'wake_only' mode:
        WAITING_FOR_WAKE_WORD -> ACTIVATED -> LISTENING -> PROCESSING -> EXECUTING -> RESPONDING -> WAITING_FOR_WAKE_WORD.

        Returns:
            Dictionary containing timing metrics, transcription, result, and exit status.
        """
        metrics: Dict[str, float] = {}
        t_cycle_start = time.perf_counter()
        effective_wake_timeout = wake_timeout if wake_timeout is not None else (
            self.wake_timeout if self.mode == "staged" else None
        )

        # =====================================================================
        # STAGE 1: WAITING_FOR_CLAP (Only in staged mode)
        # =====================================================================
        if self.mode == "staged":
            self._set_state(AssistantState.WAITING_FOR_CLAP)
            if self.dev_mode:
                print("[STATE] Waiting for clap...")

            t_clap_start = time.perf_counter()
            clap_detected = self.clap_detector.listen(timeout_seconds=clap_timeout)
            metrics["clap_detection"] = round(time.perf_counter() - t_clap_start, 3)

            if not clap_detected:
                if self.dev_mode:
                    print("[STATE] Clap listening timed out.")
                return {
                    "clap_detected": False,
                    "wake_detected": False,
                    "should_exit": False,
                    "metrics": metrics,
                    "final_state": self._state.value,
                }

            # Brief pause to let acoustic reverberation clear and release mic stream cleanly
            time.sleep(0.1)
        else:
            clap_detected = True
            metrics["clap_detection"] = 0.0

        # =====================================================================
        # STAGE 2: WAITING_FOR_WAKE_WORD (Microphone owned by WakeWordDetector)
        # =====================================================================
        self._set_state(AssistantState.WAITING_FOR_WAKE_WORD)
        if self.dev_mode:
            tout_str = f"{effective_wake_timeout}s" if effective_wake_timeout is not None else "continuous"
            print(f"[STATE] Listening for wake phrase ('{self.wake_detector.model_name}', timeout: {tout_str})...")

        def on_wake_frame(score: float) -> None:
            if score >= 0.15 and self.dev_mode:
                print(f"\r[WAKE] Hearing wake phrase candidate... (Confidence: {score:.2f}){' '*15}", end="", flush=True)

        t_wake_start = time.perf_counter()
        wake_detected = self.wake_detector.listen(
            timeout_seconds=effective_wake_timeout,
            on_frame=on_wake_frame,
        )
        metrics["wake_detection"] = round(time.perf_counter() - t_wake_start, 3)

        # Timeout: Wake phrase was NOT spoken
        if not wake_detected:
            if self.dev_mode:
                print(f"\r[TIMEOUT] No wake phrase detected within {effective_wake_timeout}s. Returning to standby.{' '*20}\n", flush=True)
            fallback_state = AssistantState.WAITING_FOR_WAKE_WORD if self.mode == "wake_only" else AssistantState.WAITING_FOR_CLAP
            self._set_state(fallback_state)
            return {
                "clap_detected": clap_detected,
                "wake_detected": False,
                "should_exit": False,
                "metrics": metrics,
                "final_state": self._state.value,
            }

        print(f"\r[WAKE] Wake phrase detected! [CONFIRMED]{' '*35}\n", flush=True)

        # =====================================================================
        # STAGE 3: ACTIVATED (Double gate satisfied, prepare command capture)
        # =====================================================================
        self._set_state(AssistantState.ACTIVATED)
        t_act_start = time.perf_counter()
        self._play_activation_cue()
        metrics["activation_cue"] = round(time.perf_counter() - t_act_start, 3)

        # =====================================================================
        # STAGE 4: LISTENING (Microphone owned by AudioRecorder)
        # =====================================================================
        self._set_state(AssistantState.LISTENING)
        print(f"[STATE] Listening for command ({command_duration:.1f}s)... Speak now!")

        t_rec_start = time.perf_counter()
        command_audio = self.orchestrator.recorder.record(duration_seconds=command_duration)
        metrics["command_recording"] = round(time.perf_counter() - t_rec_start, 3)

        # =====================================================================
        # STAGE 5: PROCESSING (STT Transcription & LLM Understanding)
        # =====================================================================
        self._set_state(AssistantState.PROCESSING)

        # 5a: Speech-to-Text
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
            print(f"[STT] Transcription: \"{clean_text}\" (took {metrics['stt']}s)")

        # Handle Timeout / Silence: No speech detected
        if not clean_text:
            if self.dev_mode:
                print("[PROCESSING] No speech detected in command recording (silence/timeout).")

            self._set_state(AssistantState.RESPONDING)
            silence_response = "I didn't hear anything. Returning to standby."
            t_tts_start = time.perf_counter()
            self.orchestrator._speak(silence_response)
            metrics["tts"] = round(time.perf_counter() - t_tts_start, 3)
            metrics["total_interaction"] = round(time.perf_counter() - t_cycle_start, 3)

            # Return to standby
            fallback_state = AssistantState.WAITING_FOR_WAKE_WORD if self.mode == "wake_only" else AssistantState.WAITING_FOR_CLAP
            self._set_state(fallback_state)
            return {
                "clap_detected": clap_detected,
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

        # 5b: LLM Command Understanding
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
            print(f"[LLM] Intent: '{intent}', Entity: '{entity}' (took {metrics['llm']}s)")

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
                "clap_detected": True,
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
        # STAGE 6: EXECUTING (Tool Router Vetting & Tool Invocation)
        # =====================================================================
        self._set_state(AssistantState.EXECUTING)
        t_tool_start = time.perf_counter()
        tool_result = self.orchestrator.router.route_and_execute({"intent": intent, "entity": entity})
        metrics["tool_execution"] = round(time.perf_counter() - t_tool_start, 3)

        if self.dev_mode:
            status_lbl = "SUCCESS" if tool_result.success else "REJECTED/FAILED"
            print(f"[TOOL] Result: {status_lbl} ({tool_result.message})")

        # =====================================================================
        # STAGE 7: RESPONDING (Natural Language Formatting & Piper TTS)
        # =====================================================================
        self._set_state(AssistantState.RESPONDING)
        response_text = self.orchestrator._generate_natural_response(intent, entity, tool_result)

        t_tts_start = time.perf_counter()
        if self.dev_mode:
            print(f"[TTS] Response: \"{response_text}\"")

        self.orchestrator._speak(response_text)
        metrics["tts"] = round(time.perf_counter() - t_tts_start, 3)
        metrics["total_interaction"] = round(time.perf_counter() - t_cycle_start, 3)

        # =====================================================================
        # CYCLE COMPLETE: Return to standby state
        # =====================================================================
        fallback_state = AssistantState.WAITING_FOR_WAKE_WORD if self.mode == "wake_only" else AssistantState.WAITING_FOR_CLAP
        self._set_state(fallback_state)
        if self.dev_mode:
            standby_msg = "[STATE] Waiting for 'Hey Jarvis'..." if self.mode == "wake_only" else "[STATE] Waiting for clap..."
            print(standby_msg)

        return {
            "clap_detected": clap_detected,
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
        clap_audio: Optional[np.ndarray],
        wake_audio: Optional[np.ndarray],
        command_audio: Optional[np.ndarray],
    ) -> Dict[str, Any]:
        """Execute a simulated lifecycle cycle with injected audio arrays for testing.

        Args:
            clap_audio: 16 kHz audio array to feed into ClapDetector.
            wake_audio: 16 kHz audio array to feed into WakeWordDetector.
            command_audio: 16 kHz audio array to feed into SpeechToText.

        Returns:
            Dictionary containing execution metadata, state transitions, and metrics.
        """
        metrics: Dict[str, float] = {}
        t_cycle_start = time.perf_counter()

        # Step 1: WAITING_FOR_CLAP
        self._set_state(AssistantState.WAITING_FOR_CLAP)
        t_clap_start = time.perf_counter()
        clap_detected = False

        if clap_audio is not None and len(clap_audio) > 0:
            frame_sz = self.clap_detector.FRAME_SIZE
            self.clap_detector.reset()
            for i in range(0, len(clap_audio) - frame_sz, frame_sz):
                chunk = clap_audio[i : i + frame_sz]
                is_det, _ = self.clap_detector.process_frame(chunk)
                if is_det:
                    clap_detected = True
                    break

        metrics["clap_detection"] = round(time.perf_counter() - t_clap_start, 3)

        if not clap_detected:
            return {
                "clap_detected": False,
                "wake_detected": False,
                "should_exit": False,
                "metrics": metrics,
                "final_state": self._state.value,
            }

        # Step 2: WAITING_FOR_WAKE_WORD
        self._set_state(AssistantState.WAITING_FOR_WAKE_WORD)
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
            # Timeout -> Return to WAITING_FOR_CLAP
            self._set_state(AssistantState.WAITING_FOR_CLAP)
            return {
                "clap_detected": True,
                "wake_detected": False,
                "should_exit": False,
                "metrics": metrics,
                "final_state": self._state.value,
            }

        # Step 3: ACTIVATED
        self._set_state(AssistantState.ACTIVATED)
        t_act_start = time.perf_counter()
        metrics["activation_cue"] = round(time.perf_counter() - t_act_start, 3)

        # Step 4: LISTENING
        self._set_state(AssistantState.LISTENING)
        metrics["command_recording"] = 0.0

        # Step 5: PROCESSING
        self._set_state(AssistantState.PROCESSING)
        t_stt_start = time.perf_counter()

        if command_audio is not None and len(command_audio) > 0:
            try:
                transcription, _ = self.orchestrator.stt.transcribe(command_audio)
            except Exception:
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

            self._set_state(AssistantState.WAITING_FOR_CLAP)
            return {
                "clap_detected": True,
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
                "clap_detected": True,
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

        # Step 6: EXECUTING
        self._set_state(AssistantState.EXECUTING)
        t_tool = time.perf_counter()
        tool_result = self.orchestrator.router.route_and_execute({"intent": intent, "entity": entity})
        metrics["tool_execution"] = round(time.perf_counter() - t_tool, 3)

        # Step 7: RESPONDING
        self._set_state(AssistantState.RESPONDING)
        response_text = self.orchestrator._generate_natural_response(intent, entity, tool_result)
        t_tts = time.perf_counter()
        self.orchestrator._speak(response_text)
        metrics["tts"] = round(time.perf_counter() - t_tts, 3)
        metrics["total_interaction"] = round(time.perf_counter() - t_cycle_start, 3)

        # Return to WAITING_FOR_CLAP
        self._set_state(AssistantState.WAITING_FOR_CLAP)

        return {
            "clap_detected": True,
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
        """Run the continuous activation loop until interrupted or exit."""
        print("=" * 65)
        if self.mode == "wake_only":
            print("     VROOM AI - Direct Wake-Word Mode ('Hey Jarvis')")
            print("=" * 65)
            print("Activation Gate  : [VOICE] Spoken Wake Phrase ('Hey Jarvis')")
            print("Standby Status   : [STATE] Waiting for 'Hey Jarvis'...")
        else:
            print("     VROOM AI - Staged Activation Mode (Clap + Wake Word)")
            print("=" * 65)
            print("Activation Gate 1: [CLAP] Physical Acoustic Clap")
            print(f"Activation Gate 2: [VOICE] Spoken Wake Phrase ('Hey Jarvis', timeout: {self.wake_timeout}s)")
            print("Standby Status   : [STATE] Waiting for clap...")
        print("=" * 65)

        try:
            while True:
                cycle_outcome = self.run_cycle(command_duration=command_duration)
                if cycle_outcome.get("should_exit"):
                    print("\n[CONTROLLER] Exit command processed. Shutting down gracefully.")
                    break
        except KeyboardInterrupt:
            print("\n[CONTROLLER] Interrupted by user. Exiting cleanly.")
            standby = AssistantState.WAITING_FOR_WAKE_WORD if self.mode == "wake_only" else AssistantState.WAITING_FOR_CLAP
            self._set_state(standby)
            self.orchestrator._speak("Goodbye.")
