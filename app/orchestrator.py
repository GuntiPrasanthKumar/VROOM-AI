"""Orchestration layer for VROOM AI.

Coordinates the end-to-end voice assistant pipeline:
Microphone Capture -> Speech-to-Text -> Brain/LLM Understanding ->
Tool Router -> Safe Tool Execution -> Natural Response Generation -> Text-to-Speech.
"""

import time
from typing import Any, Dict, Optional, Tuple
import numpy as np

from app.voice.audio_recorder import AudioRecorder
from app.voice.speech_to_text import SpeechToText
from app.voice.text_to_speech import TextToSpeech
from app.brain.command_parser import CommandParser, ParsedCommand
from app.brain.llm_client import OllamaClient
from app.tools.base import ToolResult
from app.tools.registry import ToolRegistry
from app.tools.router import ToolRouter


class AssistantOrchestrator:
    """Central orchestrator that coordinates all VROOM AI subsystems."""

    def __init__(
        self,
        recorder: Optional[AudioRecorder] = None,
        stt: Optional[SpeechToText] = None,
        parser: Optional[CommandParser] = None,
        llm: Optional[OllamaClient] = None,
        router: Optional[ToolRouter] = None,
        tts: Optional[TextToSpeech] = None,
        dev_mode: bool = True,
    ) -> None:
        """Initialize the assistant orchestrator with all necessary subsystems.

        Subsystems can be passed in or instantiated with default configurations.
        """
        self.dev_mode = dev_mode

        # Initialize voice capture subsystem
        self.recorder = recorder or AudioRecorder()

        # Initialize speech-to-text subsystem (Faster-Whisper)
        self.stt = stt or SpeechToText()

        # Initialize brain / understanding subsystems
        self.parser = parser or CommandParser()
        self.llm = llm or OllamaClient()

        # Initialize tool execution subsystem
        self.router = router or ToolRouter(ToolRegistry())

        # Initialize voice output subsystem (Piper TTS)
        self.tts = tts or TextToSpeech()

    def _generate_natural_response(
        self,
        intent: str,
        entity: Optional[str],
        tool_result: Optional[ToolResult],
    ) -> str:
        """Convert a structured tool outcome into a natural, conversational response."""
        entity_name = entity or "the requested item"

        # Handle exit intent
        if intent == "exit_assistant":
            return "Goodbye! Have a great day."

        # Handle tool execution results
        if tool_result is not None:
            if tool_result.success:
                if intent == "open_application":
                    return f"{entity_name} is now open."
                elif intent == "open_website":
                    return f"Opening {entity_name} now."
                elif intent == "create_folder":
                    return f"Folder {entity_name} has been created in your workspace."
                return tool_result.message

            # Unsuccessful tool execution
            if intent == "open_application":
                if "not in the approved allowlist" in tool_result.message:
                    return f"I cannot open {entity_name}. That application is not in the approved allowlist."
                elif "could not be found" in tool_result.message:
                    return f"I could not find {entity_name} on your system."
                return f"I was unable to open {entity_name}."

            if intent == "open_website":
                if "not in the approved allowlist" in tool_result.message:
                    return f"I cannot open {entity_name}. That website is not in the approved allowlist."
                return f"I was unable to open {entity_name}."

            if intent == "create_folder":
                if "already exists" in tool_result.message:
                    return f"Folder {entity_name} already exists in your workspace."
                elif "Sandbox violation" in tool_result.message:
                    return f"Cannot create folder {entity_name}. It is outside the workspace sandbox."
                return f"Could not create folder {entity_name}."

            # Unapproved intent rejected by router
            if "does not map to any approved tool" in tool_result.message:
                return "I can't perform that action. That capability is not available or not permitted."

            return "I was unable to complete that action."

        # Fallback for unrecognized intent without tool execution
        return "I don't know how to do that yet."

    def process_command_text(self, text: str) -> Dict[str, Any]:
        """Process a text command through Brain, Router, Tool, and TTS.

        Args:
            text: Transcribed or typed command string.

        Returns:
            Dictionary containing execution metadata, result, response text, and latencies.
        """
        latencies: Dict[str, float] = {}
        t_total_start = time.perf_counter()

        clean_text = text.strip() if text else ""
        if not clean_text:
            response = "I didn't hear anything. Please try again."
            t_tts_start = time.perf_counter()
            self._speak(response)
            latencies["tts"] = round(time.perf_counter() - t_tts_start, 3)
            latencies["total"] = round(time.perf_counter() - t_total_start, 3)
            return {
                "transcription": "",
                "intent": "none",
                "entity": None,
                "tool_result": None,
                "response": response,
                "should_exit": False,
                "latencies": latencies,
            }

        # Step 1: Command Understanding (Ollama LLM with deterministic fallback)
        t_brain_start = time.perf_counter()
        intent = "unknown"
        entity = None
        brain_source = "llm"

        try:
            llm_result = self.llm.parse_command(clean_text)
            intent = llm_result.get("intent", "unknown")
            entity = llm_result.get("entity")
        except Exception as exc:
            # Fallback to deterministic regex parser if Ollama is unreachable
            brain_source = "deterministic_fallback"
            if self.dev_mode:
                print(f"[WARN] Ollama inference failed ({exc}). Using deterministic parser fallback.")
            parsed = self.parser.parse(clean_text)
            intent = parsed.intent
            entity = parsed.entity

        latencies["brain"] = round(time.perf_counter() - t_brain_start, 3)

        if self.dev_mode:
            print(f"\n[LLM / BRAIN] (via {brain_source})")
            print(f"  Intent: {intent}")
            print(f"  Entity: {entity}")
            print(f"  Latency: {latencies['brain']}s")

        # Step 2: Handle Exit Assistant
        should_exit = (
            intent == "exit_assistant"
            or clean_text.lower() in ("exit", "quit", "stop", "bye", "goodbye")
        )
        if should_exit:
            response = "Goodbye! Have a great day."
            if self.dev_mode:
                print("\n[ROUTER]")
                print("  Action: Exit Assistant requested. Stopping loop.")

            t_tts_start = time.perf_counter()
            self._speak(response)
            latencies["tts"] = round(time.perf_counter() - t_tts_start, 3)
            latencies["total"] = round(time.perf_counter() - t_total_start, 3)

            return {
                "transcription": clean_text,
                "intent": "exit_assistant",
                "entity": None,
                "tool_result": None,
                "response": response,
                "should_exit": True,
                "latencies": latencies,
            }

        # Step 3: Route & Execute Tool
        t_tool_start = time.perf_counter()
        tool_result: ToolResult

        if self.dev_mode:
            print("\n[ROUTER]")
            print(f"  Routing Intent: {intent}")

        # Dispatch command dict to router
        tool_result = self.router.route_and_execute({"intent": intent, "entity": entity})
        latencies["tool"] = round(time.perf_counter() - t_tool_start, 3)

        if self.dev_mode:
            print("\n[TOOL]")
            status_label = "SUCCESS" if tool_result.success else "REJECTED/FAILED"
            print(f"  Result: {status_label}")
            print(f"  Message: {tool_result.message}")
            print(f"  Latency: {latencies['tool']}s")

        # Step 4: Natural Response Generation
        response = self._generate_natural_response(intent, entity, tool_result)

        # Step 5: Text-to-Speech Output
        t_tts_start = time.perf_counter()
        if self.dev_mode:
            print("\n[TTS]")
            print(f"  Response: {response}")

        self._speak(response)
        latencies["tts"] = round(time.perf_counter() - t_tts_start, 3)
        latencies["total"] = round(time.perf_counter() - t_total_start, 3)

        if self.dev_mode:
            print(f"  Playback Latency: {latencies['tts']}s")
            print("\n[LATENCY SUMMARY]")
            print(
                f"  Brain: {latencies['brain']}s | "
                f"Tool: {latencies['tool']}s | "
                f"TTS: {latencies['tts']}s | "
                f"Total: {latencies['total']}s"
            )

        return {
            "transcription": clean_text,
            "intent": intent,
            "entity": entity,
            "tool_result": tool_result,
            "response": response,
            "should_exit": False,
            "latencies": latencies,
        }

    def process_audio(self, audio_array: np.ndarray) -> Dict[str, Any]:
        """Process an audio waveform array through STT and the complete pipeline.

        Args:
            audio_array: 1D float32 NumPy array sampled at 16,000 Hz.

        Returns:
            Dictionary containing execution metadata, transcription, and latencies.
        """
        t_stt_start = time.perf_counter()
        try:
            transcription, meta = self.stt.transcribe(audio_array)
        except Exception as exc:
            transcription = ""
            meta = {"error": str(exc)}
            if self.dev_mode:
                print(f"[STT ERROR] Failed to transcribe audio: {exc}")

        stt_latency = round(time.perf_counter() - t_stt_start, 3)

        if self.dev_mode:
            print("\n" + "=" * 60)
            print("[STT]")
            print(f"  Transcription: \"{transcription}\"")
            print(f"  Language: {meta.get('language')} (prob: {meta.get('language_probability')})")
            print(f"  Latency: {stt_latency}s")

        result = self.process_command_text(transcription)
        result["latencies"]["stt"] = stt_latency
        result["latencies"]["total"] = round(result["latencies"].get("total", 0.0) + stt_latency, 3)

        return result

    def _speak(self, text: str) -> None:
        """Safely speak text using Piper TTS with terminal fallback."""
        try:
            self.tts.speak(text, blocking=True)
        except Exception as exc:
            if self.dev_mode:
                print(f"[TTS WARN] Audio playback failed ({exc}). Message displayed on screen.")

    def run_interactive(self, record_duration: float = 4.0) -> None:
        """Run the continuous user-interactive command loop."""
        ready_message = "VROOM is ready. Press Enter when you want to speak."
        print("\n" + "=" * 60)
        print(f"  {ready_message}")
        print("  (Tip: You can also type text directly and press Enter)")
        print("=" * 60)

        # Initial spoken announcement
        self._speak("VROOM is ready. Press Enter when you want to speak.")

        while True:
            try:
                user_input = input("\n[VROOM] Press Enter to speak (or type a command / 'quit'): ")
                stripped = user_input.strip()

                # If user typed a manual text command
                if stripped:
                    outcome = self.process_command_text(stripped)
                    if outcome.get("should_exit"):
                        break
                    continue

                # User pressed Enter without typing -> Record microphone audio
                print(f"\n[MIC] Recording for {record_duration:.1f} seconds... Speak now!")
                t_rec_start = time.perf_counter()
                audio_array = self.recorder.record(duration_seconds=record_duration)
                rec_latency = round(time.perf_counter() - t_rec_start, 3)

                if self.dev_mode:
                    print(f"[MIC] Captured {len(audio_array)} samples in {rec_latency}s.")

                outcome = self.process_audio(audio_array)
                outcome["latencies"]["recording"] = rec_latency

                if outcome.get("should_exit"):
                    break

            except KeyboardInterrupt:
                print("\n[VROOM] Interrupted by user. Exiting cleanly.")
                self._speak("Goodbye.")
                break
            except Exception as exc:
                print(f"\n[ERROR] An unexpected error occurred in loop: {exc}")
                self._speak("I encountered an unexpected error.")
