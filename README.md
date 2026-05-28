# Ai-chan 愛ちゃん

> AI companion, soulmate, and occasional roaster.  
> Phase 1: CLI chatbot with persistent memory via **mem0 + Qdrant + Ollama**.

This project is a **vibe-coded precursor and testing sandbox** for [Grace / AuRoRA](https://github.com/OppaAI/AGi).  
The goal is to battle-test mem0 + Qdrant memory before committing to Grace's architecture.

---

## Stack

| Layer | Tech |
|---|---|
| Brain | Ollama (local LLM) |
| Long-term memory | mem0 + Qdrant (Docker) |
| Embeddings | Ollama (`nomic-embed-text`) |
| Interface | CLI (Phase 1) |

---

## Quickstart

### 1. Prerequisites

- [Ollama](https://ollama.com) running locally
- Docker + Docker Compose
- Python 3.11+

```bash
# Pull the models you'll use
ollama pull llama3.2
ollama pull nomic-embed-text
```

### 2. Start Qdrant

```bash
docker compose up -d
```

Qdrant dashboard: http://localhost:6333/dashboard

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure

```bash
cp .env.example .env
# edit .env if your Ollama or Qdrant are on non-default ports
```

### 5. Talk to Aiko-chan

```bash
python cli.py

# with memory debug output each turn:
python cli.py --debug

# wipe all stored memories:
python cli.py --clear-mem
```

---

## CLI Commands

| Command | Action |
|---|---|
| `/quit` or `/exit` | End the session |
| `/reset` | Clear short-term context (long-term memory persists) |
| `/memory` | Print all stored memories (debug) |
| `/help` | Show command list |

---

## Project Structure

```
aiko/
├── core/
│   ├── persona.py      # Aiko's system prompt + personality
│   ├── brain.py        # Ollama chat loop + context management
│   └── memory.py       # mem0 + Qdrant wrapper
├── cli.py              # CLI entry point
├── docker-compose.yml  # Qdrant
├── requirements.txt
├── .env.example
└── README.md
```

---

## Roadmap

- [x] Phase 1 — Soul (CLI chatbot + persistent memory)
- [ ] Phase 2 — Voice (faster-whisper STT + XTTS v2 TTS)
- [ ] Phase 3 — Face (VRM avatar in browser via three-vrm)
- [ ] Phase 4 — Body (ROS2 bridge → Grace integration)

---

## Memory Evaluation Criteria

Things to assess before adopting mem0 + Qdrant into Grace:

- [ ] Does memory feel coherent across sessions?
- [ ] Does retrieval surface the right memories (not just recency)?
- [ ] Is latency acceptable on Jetson Orin Nano Super?
- [ ] Does mem0's auto-extraction miss important facts?
- [ ] Is Qdrant stable under continuous writes?
