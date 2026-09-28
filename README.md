# VROOM AI — Intelligent Desktop Voice Assistant

VROOM AI is a modular, privacy-focused, and intelligent voice-controlled desktop assistant for Windows.

---

## 🎯 Learning & Engineering Philosophy

This project is built from the ground up with a strict engineering principle: **understand every component before abstracting it**.

- **No Black Boxes:** Every module, library, and algorithm is introduced incrementally.
- **Deep Explanations:** Focus on *what* it is, *why* it is chosen, *how* it works under the hood, and its alternatives.
- **Zero Premature Dependencies:** Dependencies are added strictly on-demand as each module is built.
- **Security by Default:** Zero unrestricted destructive actions. Sensitive commands will always require confirmation.

---

## 🏛️ Planned Modular Architecture

As the project matures, VROOM AI will follow a clean, decoupled structure:

```text
vroom-ai/
│
├── app/
│   ├── core/         # Configuration, event bus, lifecycle, logging
│   ├── voice/        # Audio capture, wake-word detection, speech-to-text (STT), text-to-speech (TTS)
│   ├── brain/        # Local LLM reasoning, prompt engine, intent classification
│   ├── tools/        # Tool registry, function execution definitions
│   ├── automation/   # Desktop control, window management, keyboard/mouse actions
│   ├── vision/       # Screen capture, OCR, visual understanding
│   ├── memory/       # Conversation history, local vector database / state persistence
│   └── ui/           # Status overlay, visual feedback
│
├── tests/            # Unit and integration test suites
├── config/           # YAML/JSON settings and safety policies
├── scripts/          # Developer tooling and setup utilities
├── docs/             # Technical deep dives and concept notes
├── .gitignore
├── requirements.txt
└── main.py           # Application entry point
```

---

## 🚀 Getting Started (Foundation Stage)

### Prerequisites
- Python 3.10+
- Git

### Setup
1. Clone or open the repository:
   ```bash
   cd VROOM-AI
   ```

2. Activate the virtual environment (Windows PowerShell):
   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```

3. Run the initial entry point:
   ```bash
   python main.py
   ```
