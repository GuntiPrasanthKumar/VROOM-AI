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
    CONVERSATION_SESSION = "CONVERSATION_SESSION"      # Multi-turn continuous conversation session
    LISTENING = "LISTENING"                            # Microphone recording user command
    PROCESSING = "PROCESSING"                          # Whisper STT & Ollama LLM understanding
    EXECUTING = "EXECUTING"                            # Tool Router validation and execution
    RESPONDING = "RESPONDING"                          # Piper TTS speaking output
    SESSION_END = "SESSION_END"                        # Session concluded, transitioning back to standby


SESSION_TERMINATION_KEYWORDS = (
    "goodbye",
    "sleep",
    "go to sleep",
    "deactivate",
    "standby",
    "stop listening",
)

EXIT_ASSISTANT_KEYWORDS = (
    "exit assistant",
    "quit",
    "exit",
    "shut down",
    "power down",
)


def is_session_termination(text: str) -> bool:
    """Return True if text commands VROOM to conclude the conversation session."""
    clean = text.lower().rstrip(".!?,;:").strip()
    if clean in SESSION_TERMINATION_KEYWORDS:
        return True
    for kw in ("sleep", "go to sleep", "deactivate", "standby", "stop listening"):
        if clean.startswith(kw):
            return True
    return False


def is_exit_assistant(text: str, intent: str = "unknown") -> bool:
    """Return True if text commands VROOM to completely shut down the application."""
    clean = text.lower().rstrip(".!?,;:").strip()
    if is_session_termination(clean):
        return False
    if clean in EXIT_ASSISTANT_KEYWORDS:
        return True
    for kw in ("quit", "exit", "shut down", "power down"):
        if clean.startswith(kw):
            return True
    if intent == "exit_assistant":
        return True
    return False


class ActivationController:
    """Manages the two-stage (Clap + Wake Word) activation state machine."""

    def __init__(
        self,
        orchestrator: Optional[AssistantOrchestrator] = None,
        clap_detector: Optional[ClapDetector] = None,
        wake_detector: Optional[WakeWordDetector] = None,
        wake_model_name: str = "hey_jarvis",
        wake_threshold: float = 0.22,
        wake_timeout: float = 8.0,
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
            print(f"\n[STATE] {new_state.value}")
        if self.on_state_change is not None:
            self.on_state_change(old_state, new_state)

    def _play_activation_cue(self) -> None:
        """Provide a brief audio chime cue indicating VROOM is listening for a command."""
        print("[STATE] VROOM activated. Ready for command.")
        try:
            # Play a crisp 120ms dual-tone chime (880 Hz -> 1320 Hz) so microphone capture starts almost immediately
            sr = 16000
            t1 = np.linspace(0, 0.05, int(sr * 0.05), False)
            t2 = np.linspace(0, 0.07, int(sr * 0.07), False)
            tone1 = 0.10 * np.sin(2 * np.pi * 880 * t1)
            tone2 = 0.12 * np.sin(2 * np.pi * 1320 * t2)
            silence = np.zeros(int(sr * 0.02))
            chime = np.concatenate([tone1, silence, tone2]).astype(np.float32)
            import sounddevice as sd
            sd.play(chime, samplerate=sr)
            sd.wait()
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

            t_clap_start = time.perf_counter()
            clap_detected = self.clap_detector.listen(timeout_seconds=clap_timeout)
            metrics["clap_detection"] = round(time.perf_counter() - t_clap_start, 3)

            if not clap_detected:
                if self.dev_mode:
                    print("[CLAP] Listening timed out.")
                return {
                    "clap_detected": False,
                    "wake_detected": False,
                    "should_exit": False,
                    "metrics": metrics,
                    "final_state": self._state.value,
                }

            # Brief pause to let acoustic reverberation clear and release PortAudio endpoint cleanly
            time.sleep(0.05)
        else:
            clap_detected = True
            metrics["clap_detection"] = 0.0

        # =====================================================================
        # STAGE 2: WAITING_FOR_WAKE_WORD (Microphone owned by WakeWordDetector)
        # =====================================================================
        self._set_state(AssistantState.WAITING_FOR_WAKE_WORD)

        def on_wake_frame(score: float) -> None:
            if score >= 0.12 and self.dev_mode:
                print(f"\r[WAKE] Hearing candidate... (Score: {score:.2f}){' '*15}", end="", flush=True)

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

        # Brief pause to cleanly release stream before playing chime and recording
        time.sleep(0.05)

        # =====================================================================
        # STAGE 3: ACTIVATED (Double gate satisfied, prepare command capture)
        # =====================================================================
        self._set_state(AssistantState.ACTIVATED)
        t_act_start = time.perf_counter()
        self._play_activation_cue()
        metrics["activation_cue"] = round(time.perf_counter() - t_act_start, 3)

        # =====================================================================
        # STAGE 4: CONVERSATION_SESSION (Continuous Multi-Command Session)
        # =====================================================================
        self._set_state(AssistantState.CONVERSATION_SESSION)
        if self.dev_mode:
            print("[SESSION] Conversation session active. Speak multiple commands freely.")
            print("[SESSION] Say 'sleep', 'goodbye', or 'deactivate' to end session.")

        session_turns: list = []
        turn_count = 0
        should_exit = False
        session_active = True

        while session_active:
            turn_count += 1
            if turn_count > 1:
                self._set_state(AssistantState.CONVERSATION_SESSION)

            # 4a: LISTENING (Microphone owned by AudioRecorder)
            self._set_state(AssistantState.LISTENING)
            t_rec_start = time.perf_counter()
            command_audio = self.orchestrator.recorder.record(duration_seconds=command_duration)
            t_rec_dur = round(time.perf_counter() - t_rec_start, 3)

            # 4b: PROCESSING (STT & LLM)
            self._set_state(AssistantState.PROCESSING)
            t_stt_start = time.perf_counter()
            print("[STT] Transcribing...")
            try:
                transcription, stt_meta = self.orchestrator.stt.transcribe(command_audio)
            except Exception as exc:
                transcription = ""
                stt_meta = {"error": str(exc)}
                if self.dev_mode:
                    print(f"[STT ERROR] Failed to transcribe: {exc}")

            t_stt_dur = round(time.perf_counter() - t_stt_start, 3)
            clean_text = transcription.strip() if transcription else ""
            print(f"[STT] Result: \"{clean_text}\" (took {t_stt_dur}s)")

            # Handle Silence / No Speech -> Session Concludes
            if not clean_text:
                if self.dev_mode:
                    print("[PROCESSING] No speech detected in command recording (silence/timeout).")
                self._set_state(AssistantState.SESSION_END)
                silence_resp = "Entering standby."
                self._set_state(AssistantState.RESPONDING)
                print(f"[TTS] Speaking response: \"{silence_resp}\"")
                t_tts_start = time.perf_counter()
                self.orchestrator._speak(silence_resp)
                print("[TTS] Playback complete")
                session_turns.append({
                    "turn": turn_count,
                    "transcription": "",
                    "intent": "timeout",
                    "entity": None,
                    "tool_result": None,
                    "response": silence_resp,
                })
                session_active = False
                break

            # 4c: LLM Command Understanding
            t_brain_start = time.perf_counter()
            intent = "unknown"
            entity = None
            print(f"[LLM] Parsing command: \"{clean_text}\"")
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

            t_brain_dur = round(time.perf_counter() - t_brain_start, 3)
            print(f"[LLM] Intent: '{intent}', Entity: '{entity}' (took {t_brain_dur}s)")

            # 4d: Check for Session Termination ("goodbye", "sleep", "go to sleep", "deactivate", "standby")
            if is_session_termination(clean_text):
                self._set_state(AssistantState.SESSION_END)
                sleep_response = "Going to sleep."
                self._set_state(AssistantState.RESPONDING)
                print(f"[TTS] Speaking response: \"{sleep_response}\"")
                t_tts_start = time.perf_counter()
                self.orchestrator._speak(sleep_response)
                print("[TTS] Playback complete")
                session_turns.append({
                    "turn": turn_count,
                    "transcription": clean_text,
                    "intent": "session_termination",
                    "entity": None,
                    "tool_result": None,
                    "response": sleep_response,
                })
                session_active = False
                break

            # 4e: Check for Full Application Shutdown ("quit", "exit", "shut down", "power down")
            if is_exit_assistant(clean_text, intent):
                intent = "exit_assistant"
                self._set_state(AssistantState.RESPONDING)
                exit_response = "Goodbye! Powering down."
                print(f"[TTS] Speaking response: \"{exit_response}\"")
                t_tts_start = time.perf_counter()
                self.orchestrator._speak(exit_response)
                print("[TTS] Playback complete")
                should_exit = True
                session_turns.append({
                    "turn": turn_count,
                    "transcription": clean_text,
                    "intent": "exit_assistant",
                    "entity": None,
                    "tool_result": None,
                    "response": exit_response,
                })
                session_active = False
                break

            # 4f: EXECUTING (Regular tool execution)
            self._set_state(AssistantState.EXECUTING)
            print(f"[ROUTER] Routing intent '{intent}'...")
            t_tool_start = time.perf_counter()
            tool_result = self.orchestrator.router.route_and_execute({"intent": intent, "entity": entity})
            t_tool_dur = round(time.perf_counter() - t_tool_start, 3)

            status_lbl = "SUCCESS" if tool_result.success else "REJECTED/FAILED"
            print(f"[ROUTER] Result: {status_lbl} ({tool_result.message})")

            # 4g: RESPONDING (Natural Language Formatting & Piper TTS)
            self._set_state(AssistantState.RESPONDING)
            response_text = self.orchestrator._generate_natural_response(intent, entity, tool_result)

            t_tts_start = time.perf_counter()
            print(f"[TTS] Speaking response: \"{response_text}\"")
            self.orchestrator._speak(response_text)
            print("[TTS] Playback complete")
            t_tts_dur = round(time.perf_counter() - t_tts_start, 3)

            session_turns.append({
                "turn": turn_count,
                "transcription": clean_text,
                "intent": intent,
                "entity": entity,
                "tool_result": tool_result,
                "response": response_text,
                "latencies": {
                    "recording": t_rec_dur,
                    "stt": t_stt_dur,
                    "brain": t_brain_dur,
                    "tool": t_tool_dur,
                    "tts": t_tts_dur,
                },
            })

            # Small settling pause between turns to allow audio device cleanup
            time.sleep(0.08)

        # =====================================================================
        # SESSION COMPLETE: Return to standby state
        # =====================================================================
        if not should_exit:
            fallback_state = AssistantState.WAITING_FOR_WAKE_WORD if self.mode == "wake_only" else AssistantState.WAITING_FOR_CLAP
            self._set_state(fallback_state)
            if self.dev_mode:
                standby_msg = "[STATE] Waiting for 'Hey Jarvis'..." if self.mode == "wake_only" else "[STATE] Waiting for clap..."
                print(standby_msg)

        last_turn = session_turns[-1] if session_turns else {}
        metrics["total_interaction"] = round(time.perf_counter() - t_cycle_start, 3)
        return {
            "clap_detected": clap_detected,
            "wake_detected": True,
            "transcription": last_turn.get("transcription", ""),
            "intent": last_turn.get("intent", ""),
            "entity": last_turn.get("entity"),
            "tool_result": last_turn.get("tool_result"),
            "response": last_turn.get("response", ""),
            "should_exit": should_exit,
            "turns": session_turns,
            "turn_count": len(session_turns),
            "metrics": metrics,
            "final_state": self._state.value,
        }

    def execute_with_audio(
        self,
        clap_audio: Optional[np.ndarray],
        wake_audio: Optional[np.ndarray],
        command_audio: Optional[Any],
    ) -> Dict[str, Any]:
        """Execute a simulated lifecycle cycle with injected audio arrays for testing.

        Supports single command audio arrays or a list of arrays for multi-turn sessions.

        Args:
            clap_audio: 16 kHz audio array to feed into ClapDetector.
            wake_audio: 16 kHz audio array to feed into WakeWordDetector.
            command_audio: Single 16 kHz audio array or list of arrays to feed turn-by-turn.

        Returns:
            Dictionary containing execution metadata, state transitions, turns, and metrics.
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

        # Step 4: CONVERSATION_SESSION
        self._set_state(AssistantState.CONVERSATION_SESSION)

        # Normalize command_audio to list of arrays
        if command_audio is None:
            audio_turns = [None]
        elif isinstance(command_audio, list):
            audio_turns = command_audio
        else:
            audio_turns = [command_audio]

        session_turns: list = []
        should_exit = False

        for turn_idx, cmd_aud in enumerate(audio_turns, 1):
            if turn_idx > 1:
                self._set_state(AssistantState.CONVERSATION_SESSION)

            # Listen
            self._set_state(AssistantState.LISTENING)

            # Processing (STT)
            self._set_state(AssistantState.PROCESSING)
            t_stt_start = time.perf_counter()
            if cmd_aud is not None and len(cmd_aud) > 0:
                try:
                    transcription, _ = self.orchestrator.stt.transcribe(cmd_aud)
                except Exception:
                    transcription = ""
            else:
                transcription = ""

            clean_text = transcription.strip()
            metrics[f"stt_turn_{turn_idx}"] = round(time.perf_counter() - t_stt_start, 3)

            # Handle Timeout / Silence
            if not clean_text:
                self._set_state(AssistantState.SESSION_END)
                silence_resp = "Entering standby."
                self._set_state(AssistantState.RESPONDING)
                self.orchestrator._speak(silence_resp)
                session_turns.append({
                    "turn": turn_idx,
                    "transcription": "",
                    "intent": "timeout",
                    "entity": None,
                    "tool_result": None,
                    "response": silence_resp,
                })
                break

            # LLM
            t_brain = time.perf_counter()
            try:
                llm_res = self.orchestrator.llm.parse_command(clean_text)
                intent = llm_res.get("intent", "unknown")
                entity = llm_res.get("entity")
            except Exception:
                parsed = self.orchestrator.parser.parse(clean_text)
                intent = parsed.intent
                entity = parsed.entity

            metrics[f"llm_turn_{turn_idx}"] = round(time.perf_counter() - t_brain, 3)

            # Handle Session Termination
            if is_session_termination(clean_text):
                self._set_state(AssistantState.SESSION_END)
                sleep_resp = "Going to sleep."
                self._set_state(AssistantState.RESPONDING)
                self.orchestrator._speak(sleep_resp)
                session_turns.append({
                    "turn": turn_idx,
                    "transcription": clean_text,
                    "intent": "session_termination",
                    "entity": None,
                    "tool_result": None,
                    "response": sleep_resp,
                })
                break

            # Handle Exit Assistant
            if is_exit_assistant(clean_text, intent):
                intent = "exit_assistant"
                self._set_state(AssistantState.RESPONDING)
                exit_resp = "Goodbye! Powering down."
                self.orchestrator._speak(exit_resp)
                should_exit = True
                session_turns.append({
                    "turn": turn_idx,
                    "transcription": clean_text,
                    "intent": "exit_assistant",
                    "entity": None,
                    "tool_result": None,
                    "response": exit_resp,
                })
                break

            # Execute tool
            self._set_state(AssistantState.EXECUTING)
            tool_result = self.orchestrator.router.route_and_execute({"intent": intent, "entity": entity})

            # Respond
            self._set_state(AssistantState.RESPONDING)
            response_text = self.orchestrator._generate_natural_response(intent, entity, tool_result)
            self.orchestrator._speak(response_text)

            session_turns.append({
                "turn": turn_idx,
                "transcription": clean_text,
                "intent": intent,
                "entity": entity,
                "tool_result": tool_result,
                "response": response_text,
            })

        # Session Concluded -> Return to standby if not exited
        if not should_exit:
            self._set_state(AssistantState.WAITING_FOR_CLAP)

        last_turn = session_turns[-1] if session_turns else {}
        metrics["total_interaction"] = round(time.perf_counter() - t_cycle_start, 3)
        return {
            "clap_detected": True,
            "wake_detected": True,
            "transcription": last_turn.get("transcription", ""),
            "intent": last_turn.get("intent", ""),
            "entity": last_turn.get("entity"),
            "tool_result": last_turn.get("tool_result"),
            "response": last_turn.get("response", ""),
            "should_exit": should_exit,
            "turns": session_turns,
            "turn_count": len(session_turns),
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
