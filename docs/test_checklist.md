# Aiko-chan — Phase Testing Checklist

> Track test status per phase. Mark `[x]` when confirmed working on target hardware.
> Hardware targets: dev machine (x86 GPU) and Jetson Orin Nano Super (8 GB).

---

## Phase 1 — Soul

Core pipeline: Ollama inference + mem0/Qdrant memory + SearXNG search.

### Startup & environment

- [ ] `.env` loads correctly; missing keys fail with a clear error message
- [ ] Qdrant starts and is reachable at configured host/port
- [ ] SearXNG starts and returns results for a test query
- [ ] Ollama is reachable and the configured model responds
- [ ] `uv sync` installs cleanly with no dependency conflicts

### Memory — write path

- [ ] Conversation turn queues a memory write without blocking the chat loop
- [ ] Memory write completes and is visible in the Qdrant dashboard
- [ ] Async write worker does not crash on Ollama extraction failure
- [ ] Write latency is acceptable under continuous conversation (no queue buildup)

### Memory — read path

- [ ] Retrieved memories are relevant to the query (not just most recent)
- [ ] `access_count` and `last_accessed_at` metadata update on each retrieval hit
- [ ] Memory context is correctly injected into the next prompt
- [ ] `/memory` command lists all stored memories accurately

### Memory — lifecycle

- [ ] `/clear` wipes all memories from Qdrant
- [ ] `/remember` pins the previous exchange (pin survives cleanup pass)
- [ ] Startup decay cleanup removes memories below threshold after the grace period
- [ ] Pinned memories are NOT removed by cleanup or dream pruning

### Memory quality

- [ ] Memory feels coherent across cold-start sessions
- [ ] Extraction quality is stable across at least two different Ollama models
- [ ] Model does not confabulate memories from empty or sparse context
- [ ] Ebbinghaus decay scoring produces sensible eviction ordering

### Dream / consolidation

- [ ] `dream()` scheduler fires at local midnight
- [ ] Near-duplicate memories are merged by vector similarity
- [ ] Salient memories receive a boost; decayed memories are pruned
- [ ] Dream pass completes without crashing on an empty memory store

### Web search

- [ ] `/web <query>` returns results and Aiko answers from them
- [ ] Search trigger fires automatically during conversation when appropriate
- [ ] SearXNG secret auth passes correctly

### Session commands

- [ ] `/reset` clears short-term context; long-term memory persists
- [ ] `/quit` and `/exit` exit cleanly

### Jetson-specific

- [ ] Qdrant stable under continuous writes during a 30-minute session
- [ ] No OOM errors with Ollama model loaded alongside Qdrant
- [ ] Write queue does not grow unbounded over a long session

---

## Phase 1.5 — Stream

Streaming inference, curses TUI, decoupled TTS pipeline.

### TUI

- [ ] Full-screen curses TUI renders correctly at standard terminal sizes
- [ ] Cyberpunk ASCII identity banner displays on startup
- [ ] Streaming tokens appear in real time in the response pane
- [ ] No rendering artifacts or flicker during streaming
- [ ] TUI recovers cleanly from a terminal resize

### Streaming inference

- [ ] LLM response streams token-by-token without buffering the full response
- [ ] Background LLM warmup eliminates cold-start latency on first message
- [ ] `/think <question>` runs a reasoning turn and suppresses raw `<think>` scratchpad output

### TTS pipeline

- [ ] MioTTS `/health` endpoint is reachable at configured URL
- [ ] TTS begins playing before the LLM response finishes streaming
- [ ] Background TTS warmup eliminates first-utterance latency
- [ ] `/voice` toggle turns TTS on and off at runtime without crash
- [ ] Audio plays correctly on both dev machine and Jetson

### Persona

- [ ] `persona/soul.md` loads and is injected into the system prompt
- [ ] `persona/identity.md` banner and color map render correctly in TUI
- [ ] Aiko's personality and voice are consistent across a multi-turn session

### Non-blocking memory

- [ ] Memory queue worker is non-blocking; chat loop does not stall on write
- [ ] No synchronous memory write bottlenecks visible in response latency

### Reflection (optional)

- [ ] If `GITHUB_TOKEN` / `GITHUB_REPO` are missing, reflection publishing fails safely and logs the reason
- [ ] If configured, Hugo markdown post is generated and uploaded correctly after dream pass

---

## Phase 2 — Voice

Microphone input, VAD, faster-whisper ASR, Interactive Talk mode.

### ASR — faster-whisper

- [ ] faster-whisper loads the configured model on startup
- [ ] Transcription is accurate for typical conversational speech
- [ ] Transcription latency is acceptable on Jetson (target: < 2 s for short utterances)
- [ ] ASR handles silence / background noise without spurious transcriptions
- [ ] `/listen` toggle enables and disables ASR at runtime

### Silero VAD

- [ ] VAD gates the microphone correctly; only speech segments are passed to ASR
- [ ] VAD does not cut off the end of utterances (tuned `min_silence_duration`)
- [ ] VAD does not false-trigger on keyboard noise or ambient sound
- [ ] VAD + ASR pipeline runs in real time without blocking the chat loop

### Interactive Talk mode

- [ ] Full hands-free conversation loop works: speak → transcribe → LLM → TTS
- [ ] Turn-taking is natural; Aiko does not interrupt mid-utterance
- [ ] System handles overlapping TTS playback + new VAD activity gracefully
- [ ] No audio feedback loop between speaker output and microphone input

### Voice + TTS integration

- [ ] TTS voice output is not captured by the microphone and re-transcribed
- [ ] `/voice` and `/listen` toggles work independently and in combination
- [ ] Voice mode works on Jetson with all other services (Ollama, Qdrant, SearXNG) running concurrently

### Jetson-specific

- [ ] faster-whisper runs on Jetson with Jetson AI Lab wheels
- [ ] Total RAM + VRAM headroom is sufficient with Ollama + Qdrant + ASR loaded simultaneously
- [ ] No thermal throttling causes ASR latency spikes during sustained use

---

*See the main [README](README.md) for architecture and setup.*
