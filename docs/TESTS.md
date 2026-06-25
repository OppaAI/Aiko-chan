[← Back to README](../README.md)
# Aiko-chan 愛子ちゃん — Industrial Test Plan

This document is the master validation plan for Aiko-chan. It is intentionally broader than a smoke-test checklist: use it for feature acceptance, regression gates, release hardening, stress/load testing, troubleshooting, and field-readiness checks on constrained hardware such as the Jetson Orin Nano.

Supporting suite folders live under [`tests/`](../tests/README.md):

| Suite | Scope |
|---|---|
| [`tests/unit/`](../tests/unit/README.md) | Pure function/module validation that should run without external services. |
| [`tests/integration/`](../tests/integration/README.md) | Boundaries between local services: LLM, SearXNG, MioTTS, ASR, sqlite-vec, WebUI. |
| [`tests/system/`](../tests/system/README.md) | End-to-end user journeys through TUI/WebUI/voice/agent workflows. |
| [`tests/load/`](../tests/load/README.md) | Throughput, concurrent requests, queue pressure, and resource saturation. |
| [`tests/error/`](../tests/error/README.md) | Fault injection, negative tests, and graceful degradation. |
| [`tests/performance/`](../tests/performance/README.md) | Latency, token rate, ASR/TTS timing, recall timing, startup timing. |
| [`tests/stability/`](../tests/stability/README.md) | Long-running soak tests, memory leak checks, restart/reconnect recovery. |
| [`tests/functionality/`](../tests/functionality/README.md) | User-facing acceptance criteria by phase and feature. |

---

## 0. Test Discipline

### 0.1 Release Gate Levels

| Gate | Purpose | Required before |
|---|---|---|
| G0 — Static sanity | Docs/code import, config lint, no obvious breakage | Every commit |
| G1 — Unit/module | Pure Python behavior and safe path validation | Merging code changes |
| G2 — Integration | External/local service boundaries | Changing dependencies or service adapters |
| G3 — System smoke | TUI/WebUI/Text end-to-end paths | Any release candidate |
| G4 — Stress/performance | Latency, load, resource pressure | Hardware field use or demo builds |
| G5 — Soak/stability | Multi-hour runtime and recovery | Long unattended deployments |

### 0.2 Pass/Fail Criteria

A test passes only when all of these are true:

- command exits with status `0`, unless the test explicitly expects a controlled failure;
- no uncaught traceback appears in terminal or log output;
- no service silently hangs beyond the test timeout;
- user-facing errors are actionable and do not leak raw stack traces;
- persistent files are created only in approved paths (`workspace/`, configured sqlite DB path, logs/cache paths);
- temporary test artifacts are either cleaned up or clearly listed for manual cleanup.

### 0.3 Evidence to Capture

For every G3+ run, save:

- command line and git commit SHA;
- `.env` values with secrets redacted;
- hardware target (`uname -a`, CPU/GPU/JetPack if relevant);
- service versions (`uv --version`, `docker compose ps`, LLM server/version if available);
- logs from Aiko, SearXNG, LLM server, MioTTS, and system audio;
- latency/resource notes for startup, first token, ASR, TTS, and memory recall.

---

## 1. Pre-flight — Stack Health

Run before any phase, integration, performance, or stability test.

### 1.1 Static Repository Checks

- [ ] `git status --short` is clean except intentional test artifacts.
- [ ] `python --version` reports Python 3.12.x.
- [ ] `uv --version` succeeds.
- [ ] `uv sync` completes without dependency conflicts.
- [ ] `python -m py_compile main.py core/think.py core/wakeup.py core/agentic.py core/memorize.py core/listen.py core/speak.py core/schedule.py` completes.
- [ ] `python - <<'PY'
from pathlib import Path
for p in ['README.md','docs/INSTALL.md','docs/ARCHITECTURE.md','docs/HISTORY.md','docs/TESTS.md']:
    assert Path(p).exists(), p
print('docs present')
PY` completes.

### 1.2 Service Health

- [ ] `docker compose up -d` starts the SearXNG stack.
- [ ] `docker compose ps` shows `aiko_searxng` / `searxng` as running.
- [ ] `curl "http://localhost:8081/search?q=test&format=json"` returns JSON and not HTML/error text.
- [ ] `curl http://localhost:8080/v1/models` returns JSON containing the configured `LLM_MODEL` alias.
- [ ] `curl http://localhost:8001/health` returns a healthy response when voice mode is being tested.
- [ ] `uv run python -c "import sqlite_vec, fastembed, sherpa_onnx, silero_vad, sounddevice, websockets; print('OK')"` prints `OK`.
- [ ] `uv run python core/speak.py --devices` lists expected audio output devices when TTS is tested.

### 1.3 Persistent Storage and Secrets

- [ ] `SQLITE_MEMORY_PATH` exists or its parent directory is writable and persistent.
- [ ] `FASTEMBED_CACHE_PATH` exists or its parent directory is writable and persistent.
- [ ] `WORKSPACE_ROOT` exists or can be created.
- [ ] `SCHEDULE_PATH` parent directory exists or can be created.
- [ ] `.env` does not contain placeholder values for required runtime tests (`SEARXNG_SECRET`, `LLM_MODEL`, `LLM_BASE_URL`).
- [ ] Secrets are redacted from logs/screenshots before sharing test artifacts.

---

## 2. Unit Test Checklist

These tests should avoid live network/model dependencies unless explicitly stated.

### 2.1 CLI and Command Parsing

- [ ] `uv run python - <<'PY'
from main import parse_args
print('parse_args import OK')
PY` imports without launching the app.
- [ ] `_match_voice_command('remember this')` returns `/remember`.
- [ ] `_match_voice_command('uh hey aiko forget that')` returns `/reset` or a documented no-match if fuzzy confidence is too low.
- [ ] `_match_voice_command('tell me a story')` returns `None`.
- [ ] conflicting flags `--tui --webui` exit with a clear message.
- [ ] `--clear-mem` path instantiates memory and exits without starting TUI/WebUI.

### 2.2 Text Sanitization and Streaming Helpers

- [ ] `core.speak.sanitize_for_tts()` strips markdown symbols, emoji noise, and unsafe punctuation while preserving readable text.
- [ ] `core.speak._split_oversized_text()` splits long text without dropping the tail.
- [ ] `core.think.split_stream_sentences()` returns complete sentence chunks and preserves incomplete fragments.
- [ ] streaming helper tests include ASCII, Japanese punctuation, emoji, markdown, empty strings, and very long tokens.

### 2.3 Memory Decay and Storage Helpers

- [ ] `core.forget.compute_weighted_score()` decreases as `last_accessed` ages.
- [ ] pinned memories are never selected for cleanup.
- [ ] grace-period memories are protected.
- [ ] `_sanitize_fts_query()` handles quotes, punctuation, emoji, empty strings, and reserved FTS characters.
- [ ] sqlite payload read/write helpers preserve IDs, roles, text, timestamps, access counts, and pinned flags.
- [ ] duplicate merge thresholds are covered for below-threshold, exact-threshold, and above-threshold cases.

### 2.4 Scheduling

- [ ] `_parse_time_of_day()` accepts valid `HH:MM` and rejects invalid hour/minute values.
- [ ] `_normalize_weekdays()` accepts strings/lists and rejects unknown weekday names.
- [ ] `_normalize_relative_days()` handles `today`, `tomorrow`, `day after tomorrow`, numeric strings, and invalid values.
- [ ] `calculate_next_due()` covers `once`, `hourly`, `daily`, `weekdays`, `weekly`, `biweekly`, `monthly`, and `custom_weekdays`.
- [ ] schedule record migration preserves legacy reminder data.
- [ ] canceling a schedule is idempotent and does not corrupt the JSON file.

### 2.5 Toolkit Safety

- [ ] `core.toolkit.common.safe_path('../secret')` rejects path traversal.
- [ ] `read_workspace_file()` rejects absolute paths and traversal outside `WORKSPACE_ROOT`.
- [ ] `repo_read_file('../.env')` rejects traversal outside the repository.
- [ ] `repo_file_tree()` respects skip directories and result limits.
- [ ] `repo_search_text()` handles no-match, many-match, binary files, and large files gracefully.
- [ ] `save_note()` enforces `MAX_WRITE_CHARS` and writes only under `WORKSPACE_ROOT`.
- [ ] photo ingestion tools produce dry-run plans without moving or modifying original image files.

### 2.6 Agentic Tool Validation

- [ ] `tool_schemas()` returns unique tool names.
- [ ] every registered tool schema has a dispatch handler.
- [ ] required arguments are enforced before handler execution.
- [ ] unknown tools produce structured failures instead of exceptions.
- [ ] retryable vs non-retryable tool failures are classified consistently.
- [ ] `final_answer` verification rejects claims that contradict tool observations.

---

## 3. Integration Test Checklist

### 3.1 LLM Endpoint Integration

- [ ] `curl http://localhost:8080/v1/models` lists `LLM_MODEL`.
- [ ] a non-streaming chat completion returns valid JSON with content.
- [ ] a streaming chat completion emits incremental chunks and closes cleanly.
- [ ] invalid model name returns a controlled error without crashing Aiko.
- [ ] timeout behavior honors `LLM_TIMEOUT` and logs a useful message.
- [ ] `/think <question>` uses the higher reasoning token budget and suppresses raw `<think>` blocks in user-facing output.

### 3.2 Memory Integration

- [ ] first launch creates/open the sqlite memory DB at `SQLITE_MEMORY_PATH`.
- [ ] adding memory writes rows and vector data.
- [ ] recall returns relevant memories for semantically similar queries.
- [ ] FTS and vector recall are combined without duplicate rows in the final result.
- [ ] `/remember` pins the last turn and pinned memories survive cleanup.
- [ ] `/clear` wipes persistent memories and does not delete unrelated files.
- [ ] `dream(dry_run=True)` completes without modifying the DB.
- [ ] full `dream()` consolidation updates/merges memories and reports counts.

### 3.3 SearXNG and Web Tools

- [ ] `web_search()` returns summarized results for a normal query.
- [ ] `web_search()` handles zero results with a friendly message.
- [ ] `fetch_and_extract()` rejects localhost/private IP targets unless explicitly allowed by code policy.
- [ ] `fetch_and_extract()` truncates huge pages to configured max chars.
- [ ] `/web <query>` displays a searching state and then a grounded answer.
- [ ] automatic search routing triggers for current/news/weather/current-version prompts and avoids search for normal personal chat.

### 3.4 MioTTS Integration

- [ ] `curl http://localhost:8001/health` succeeds.
- [ ] `uv run python core/speak.py --wait "Hello"` plays audio and exits.
- [ ] `uv run python core/speak.py --synced --wait "Hello"` emits synced callbacks without deadlock.
- [ ] long assistant replies are chunked and all chunks play in order.
- [ ] unavailable MioTTS server fails gracefully when Aiko is in `--text` mode.
- [ ] unavailable MioTTS server in voice mode produces a user-facing warning and keeps text chat usable.

### 3.5 ASR, VAD, Speaker Verification, and Barge-in

- [ ] `AikoListen.load_asr()` downloads/loads SenseVoice files on first boot.
- [ ] `AikoListen.load_vad()` initializes Silero VAD and warmup completes.
- [ ] a clean spoken phrase transcribes correctly in the expected language.
- [ ] silence does not trigger a fake utterance.
- [ ] background music/noise does not repeatedly trigger VAD.
- [ ] `LISTEN_MAX_SECONDS` caps a long recording.
- [ ] optional speaker verification returns `True`, `False`, or `None` without blocking transcription.
- [ ] barge-in stops or interrupts TTS without deadlocking the TTS or listen threads.

### 3.6 WebUI Integration

- [ ] `uv run python main.py --webui --text` starts HTTP and WebSocket servers.
- [ ] `curl http://localhost:8787/` returns the WebUI HTML.
- [ ] browser connects to the WebSocket on port `8765`.
- [ ] typed browser input reaches `AikoWeb.get_input()` and is answered.
- [ ] server broadcasts chat, token, commit, phase, vitals, voice, expression, and viseme messages with valid JSON.
- [ ] browser mic frames are accepted without crashing the backend.
- [ ] refreshing the browser reconnects without losing the Python process.
- [ ] multiple browser clients receive broadcasts consistently or are explicitly unsupported with documented behavior.

### 3.7 Schedule Runner Integration

- [ ] scheduling a one-shot reminder writes a valid job to `workspace/schedule.json`.
- [ ] due jobs are discovered by `ScheduleRunner` within `SCHEDULE_POLL_SECONDS` plus a small tolerance.
- [ ] `announce` jobs inject a user-visible reminder turn.
- [ ] `agentic` jobs call the agentic path with the scheduled task text.
- [ ] missed jobs after restart are handled according to documented policy.
- [ ] canceled jobs remain disabled or removed consistently.

---

## 4. System / End-to-End User Journeys

### 4.1 Text-Only Baseline

- [ ] Start: `uv run python main.py --text`.
- [ ] Send greeting; Aiko responds in persona.
- [ ] Send factual current query; search is either triggered automatically or `/web` succeeds.
- [ ] Send `/reset`; previous short-term context is not used.
- [ ] Send `/remember` after a turn; memory is pinned.
- [ ] Restart with `--text`; pinned memory is visible/recalled.
- [ ] Exit with `/quit`; process exits cleanly.

### 4.2 Curses TUI Full Voice

- [ ] Start: `uv run python main.py`.
- [ ] Boot UI displays all subsystem loading/done/skip states.
- [ ] Speak a short question; ASR transcript appears.
- [ ] LLM response streams incrementally.
- [ ] TTS speaks the response fully.
- [ ] Say a supported voice command such as “remember this”; it maps to the correct slash command.
- [ ] Toggle `/listen` and verify mic input is ignored, then restored.
- [ ] Toggle `/voice` and verify TTS is disabled, then restored.
- [ ] Interrupt TTS with speech; no deadlock occurs.
- [ ] Exit cleanly and audio devices are released.

### 4.3 Browser WebUI Text Journey

- [ ] Start: `uv run python main.py --webui --text`.
- [ ] Open `http://<host>:8787/`.
- [ ] VRM avatar asset loads.
- [ ] Send a text message from browser.
- [ ] Chat token streaming appears in browser.
- [ ] Backend terminal remains responsive.
- [ ] Refresh browser; connection recovers.
- [ ] Send `/help`; command list renders in the browser.
- [ ] Exit via `/quit` from browser.

### 4.4 Browser WebUI Voice Journey

- [ ] Start: `uv run python main.py --webui`.
- [ ] Browser prompts for microphone permission.
- [ ] Mic frames reach backend and trigger voice-state updates.
- [ ] Speech is transcribed and appears in chat.
- [ ] Response streams and TTS plays on the configured output path.
- [ ] Browser voice-state indicator returns to idle after the turn.
- [ ] Refresh/reconnect while idle and during a response; backend remains alive.

### 4.5 Agentic Task Journey

- [ ] Ask Aiko to create a plan; tool loop uses planning tools and returns a checklist.
- [ ] Ask Aiko to save a note; file appears under `workspace/notes/`.
- [ ] Ask Aiko to inspect its architecture; repo tools read/search relevant files.
- [ ] Ask Aiko to scan photo workspace; photo tools report counts and write only reports.
- [ ] Ask Aiko to schedule a reminder; schedule file updates and the reminder fires.
- [ ] Force a tool error (bad path, bad URL, missing arg); final answer discloses the failure honestly.

---

## 5. Phase Acceptance Matrix

### Phase 1 — Soul

*Local LLM, persona, memory, web search.*

- [ ] Local OpenAI-compatible LLM responds with streamed tokens.
- [ ] Persona from `persona/soul.md` and context files loads on startup.
- [ ] memory write queue does not block response streaming.
- [ ] memory recall improves follow-up answers.
- [ ] `/memory`, `/clear`, `/remember`, and `--clear-mem` work.
- [ ] web search returns grounded answers and does not dump raw JSON.
- [ ] failure of search service does not crash chat.

### Phase 1.5 — Stream

*Curses UI, streaming, voice-output groundwork.*

- [ ] TUI layout renders at minimum supported terminal size.
- [ ] TUI resizes without exceptions or garbled persistent state.
- [ ] streamed tokens are visible progressively.
- [ ] `/help` and all built-in commands render correctly.
- [ ] TTS toggle changes state and does not affect text output.
- [ ] startup warmup avoids first-turn cold stalls where services are already warm.

### Phase 2 — Voice

*ASR, VAD, MioTTS, barge-in.*

- [ ] SenseVoice ASR handles English and at least one configured secondary language sample.
- [ ] Silero VAD correctly starts/stops on speech/silence.
- [ ] audio-device selection is documented and reproducible.
- [ ] end-to-end latency for a short warm turn is recorded.
- [ ] barge-in is tested during short and long TTS playback.
- [ ] ASR/TTS failures degrade to typed text instead of killing the process.

### Phase 2.5 — Agent

*Tools, skills, scheduling, verification.*

- [ ] skill registry lists all local skillsets.
- [ ] tool schemas include web, fetch, planning, workspace, scheduling, reminders, skills, photos, repo tools, and final answer.
- [ ] every tool has happy-path and failure-path tests.
- [ ] schedule/reminder jobs persist and fire.
- [ ] final-answer verifier catches unsupported claims after tool failures.
- [ ] agent iteration cap prevents infinite loops.
- [ ] agent memory recall limit is respected.

### Phase 3 — Face

*VRM avatar, browser rendering, expression, lip-sync.*

- [ ] VRM asset loads from `webui/static/assets/Aiko.vrm`.
- [ ] idle animation runs for 10 minutes without visible freeze.
- [ ] expression events blend and return to idle.
- [ ] viseme events animate mouth movement and stop after audio ends.
- [ ] renderer handles browser resize and device pixel ratio changes.
- [ ] frontend handles missing VRM asset with a visible error instead of a blank page.

### Phase 4 — Presence

*Emotional/relationship state; currently roadmap-facing until implemented.*

- [ ] mood state initializes, mutates, persists, and reloads.
- [ ] positive/negative/neutral interactions produce bounded state changes.
- [ ] relationship score cannot overflow or jump unexpectedly.
- [ ] proactive messages obey inactivity and user-active suppression rules.
- [ ] privacy-sensitive state is stored locally only.

### Phase 5 — Mobile

*Mobile/WAN/push; roadmap-facing until implemented.*

- [ ] remote auth exists before WAN exposure.
- [ ] mobile text chat survives network changes.
- [ ] mobile voice input and playback work on device.
- [ ] push notifications are rate-limited and actionable.
- [ ] mobile avatar layout handles small and large screens.

### Phase 6 — Multimodal

*Image/camera; roadmap-facing until implemented.*

- [ ] image uploads are size/type validated.
- [ ] unsupported files fail safely.
- [ ] image understanding references concrete image details.
- [ ] webcam processing can be toggled and releases the device.
- [ ] camera frames are not persisted unless explicitly requested.

### Phase 7 — Autonomy

*Scheduled operation, self-directed learning, dream/reflect.*

- [ ] scheduler starts and stops cleanly.
- [ ] idle learner respects inactivity threshold.
- [ ] autonomous research stores useful memories without spamming duplicates.
- [ ] dream consolidation runs without data loss.
- [ ] reflection publishing fails safely when GitHub env vars are absent.
- [ ] unattended operation recovers from service restarts.

---

## 6. Load, Stress, and Resource Tests

### 6.1 LLM and Streaming Load

- [ ] Run 25 sequential text turns; no context corruption, exceptions, or increasing latency trend beyond documented tolerance.
- [ ] Run 5 concurrent WebUI clients sending text turns; behavior is either correct or documented as single-user only with graceful rejection.
- [ ] Send a very long user prompt near context limits; response remains bounded by token settings.
- [ ] Send rapid `/reset`, `/memory`, `/web`, `/voice`, and `/listen` commands during/after responses; no race-condition crash.
- [ ] Kill/restart the LLM server while Aiko is idle and during generation; Aiko reports failure and can recover on next turn after service returns.

### 6.2 Memory Load

- [ ] Insert 100, 1,000, and 10,000 synthetic memories in a test DB; measure recall latency and DB size.
- [ ] recall latency target is recorded for Jetson and desktop hardware.
- [ ] cleanup time is recorded at each DB size.
- [ ] dream consolidation time is recorded at each DB size.
- [ ] concurrent background writes do not corrupt sqlite DB.
- [ ] disk-full simulation produces a controlled error and preserves existing DB.

### 6.3 Voice Load

- [ ] 30 consecutive voice turns complete without ASR/TTS deadlock.
- [ ] 10 long TTS replies play fully without memory growth beyond accepted threshold.
- [ ] repeated barge-in attempts during TTS do not leave audio threads stuck.
- [ ] noisy-room sample does not cause runaway transcriptions.
- [ ] microphone unplug/replug is handled or produces a clear recoverable error.

### 6.4 WebUI Load

- [ ] browser remains responsive during a 2,000-token response.
- [ ] WebSocket reconnect loop does not leak backend threads.
- [ ] static asset server handles repeated refreshes.
- [ ] invalid WebSocket JSON/binary frames are ignored or rejected without crashing.
- [ ] browser tab left open overnight still receives vitals or reconnects cleanly.

### 6.5 Agentic Load

- [ ] max-iteration tasks stop at `MAX_AGENT_ITER` with an honest partial/failure response.
- [ ] repeated tool failures do not trigger infinite retry loops.
- [ ] large fetched pages are truncated before entering prompt context.
- [ ] workspace note writes respect size caps.
- [ ] schedule file remains valid after rapid create/cancel cycles.

---

## 7. Error Injection and Troubleshooting Matrix

| Fault | How to inject | Expected behavior | Troubleshooting notes |
|---|---|---|---|
| LLM server down | stop `llama-server` | user-facing model error; process remains alive | check `LLM_BASE_URL`, port 8080, model alias |
| wrong `LLM_MODEL` | set invalid alias | clear model-not-found error | verify `/v1/models` output |
| SearXNG down | `docker compose stop searxng` | `/web` reports search failure, chat continues | restart `docker compose up -d` |
| MioTTS down | stop TTS server | text response still appears; TTS warning logged | run `curl :8001/health` and `core/speak.py --devices` |
| ASR model missing | clear model cache/offline | startup reports ASR load failure | verify Hugging Face cache/offline mode |
| mic unavailable | unset/wrong `LISTEN_DEVICE` | voice input error, typed mode still usable | list devices with Python/sounddevice tools |
| speaker unavailable | wrong `MIOTTS_DEVICE` | TTS playback error, no process crash | run `core/speak.py --devices` |
| sqlite path unwritable | set path under read-only dir | memory error surfaced; chat should continue if possible | fix `SQLITE_MEMORY_PATH` permissions |
| workspace path traversal | request `../.env` | tool rejects path | verify safe-path tests |
| corrupt schedule JSON | write invalid JSON | scheduler handles/migrates or reports safely | backup and recreate `workspace/schedule.json` |
| WebSocket invalid payload | send malformed JSON | backend ignores/rejects and logs warning | browser devtools + backend logs |
| browser missing VRM | rename asset temporarily | visible frontend error, no backend crash | restore `webui/static/assets/Aiko.vrm` |
| GitHub token absent | unset token | reflection publish skipped/fails safely | set `GITHUB_TOKEN` only for publish tests |

---

## 8. Performance Targets and Measurements

Record actual values per hardware target. These targets are starting points, not hard promises for all models.

| Metric | Warm target | Warning threshold | Notes |
|---|---:|---:|---|
| app boot to chat-ready (`--text`) | < 20 s | > 45 s | excludes model cold download |
| app boot to voice-ready | < 60 s | > 120 s | ASR/TTS/model cache dependent |
| first token after submit | < 2 s | > 5 s | warm local LLM |
| short text turn complete | < 8 s | > 20 s | depends on model/token budget |
| memory recall | < 500 ms | > 2 s | measure at 1k/10k memories |
| `/web` search result available | < 5 s | > 15 s | depends on engines/network |
| ASR final transcript after silence | < 2 s | > 5 s | SenseVoice CPU on Jetson may vary |
| TTS first audio | < 3 s | > 8 s | MioTTS model/server dependent |
| barge-in response | < 500 ms | > 1.5 s | after threshold confirmed |
| WebUI token display lag | < 250 ms | > 1 s | browser/backend same LAN |
| 8-hour idle RSS growth | < 10% | > 25% | stability/soak target |

Suggested measurement commands:

```bash
/usr/bin/time -v uv run python main.py --text
watch -n 1 'ps -o pid,rss,pcpu,pmem,cmd -C python | head -20'
nvidia-smi dmon   # desktop NVIDIA only
jtop              # Jetson
```

---

## 9. Stability and Soak Tests

### 9.1 Idle Soak

- [ ] Start `uv run python main.py --text` and leave idle for 8 hours.
- [ ] RSS memory growth stays under target.
- [ ] scheduler thread remains alive.
- [ ] no repeated error logs appear.
- [ ] clean `/quit` after soak exits without hanging.

### 9.2 Conversation Soak

- [ ] Run at least 100 mixed turns: chat, memory, web, agent tasks, scheduling, and commands.
- [ ] no unbounded token repetition occurs.
- [ ] background memory queue drains after turns.
- [ ] memory DB remains valid after process exit/restart.
- [ ] average latency and 95th percentile latency are recorded.

### 9.3 Voice Soak

- [ ] Run 2 hours of intermittent voice use.
- [ ] no audio device lock persists after toggling `/voice` and `/listen`.
- [ ] VAD does not drift into always-listening or never-listening state.
- [ ] barge-in still works after long uptime.
- [ ] process exits cleanly.

### 9.4 WebUI Soak

- [ ] Keep a browser session connected for 8 hours.
- [ ] refresh every 30 minutes; reconnect succeeds.
- [ ] browser devtools show no runaway console errors.
- [ ] backend thread count does not continuously increase.
- [ ] VRM render remains stable.

---

## 10. Security, Privacy, and Safety Tests

- [ ] `.env`, memory DB, logs, and workspace files are not exposed by WebUI static serving.
- [ ] repo/file tools reject absolute paths, `..`, symlink escapes, and binary dumps.
- [ ] web fetch blocks private/localhost addresses unless intentionally allowed.
- [ ] agent final answer discloses failed external actions and does not claim actions were completed when tools failed.
- [ ] no tool can send email, buy/book/order, or post externally unless an explicit future tool implements and tests that behavior.
- [ ] user private facts are stored only locally unless reflection publishing is explicitly configured.
- [ ] GitHub reflection publishing redacts secrets and handles API failures safely.
- [ ] WebUI WAN exposure is not used without authentication/reverse-proxy controls.

---

## 11. Regression Checklist

Run after any significant change.

- [ ] Phase 1 memory round trip still works: store → restart → recall.
- [ ] TUI launches and streams without errors.
- [ ] WebUI launches and accepts at least one browser text turn.
- [ ] `/reset`, `/memory`, `/clear`, `/remember`, `/think`, `/web`, `/voice`, `/listen`, `/help` behave correctly.
- [ ] `--text`, `--no-asr`, `--debug`, `--webui`, and `--clear-mem` flags behave correctly.
- [ ] SearXNG recovers after `docker compose down && docker compose up -d`.
- [ ] LLM server restart does not require deleting memory or workspace files.
- [ ] `uv sync` still succeeds after dependency changes.
- [ ] docs and test-suite READMEs mention any newly added subsystem.

---

## 12. Troubleshooting Quick Reference

### Aiko hangs during boot

1. Check whether `LLM_BASE_URL` responds: `curl http://localhost:8080/v1/models`.
2. Start with `--text` to skip TTS/ASR.
3. Confirm sqlite path permissions.
4. Check model downloads/cache paths and disk space.

### TTS does not play

1. `curl http://localhost:8001/health`.
2. `uv run python core/speak.py --devices`.
3. Set `MIOTTS_DEVICE` or default PulseAudio/PipeWire sink.
4. Test `uv run python core/speak.py --wait "test"`.

### ASR does not transcribe

1. Confirm microphone device and permissions.
2. Verify `ASR_MODEL` can be downloaded or exists in cache.
3. Lower/raise `LISTEN_VAD_THRESHOLD` based on false negatives/positives.
4. Test in a quiet room before testing noisy environments.

### WebUI does not load

1. Check terminal URL and port (`AIKO_HTTP_PORT`, default 8787).
2. Check WebSocket port (`AIKO_WS_PORT`, default 8765).
3. Open browser devtools for static asset or WebSocket errors.
4. Confirm `webui/static/assets/Aiko.vrm` exists.

### Agent tools behave incorrectly

1. Re-run tool schema listing.
2. Test the underlying `core.toolkit.*` function directly.
3. Check workspace path caps and safe-path failures.
4. Inspect final-answer verification logs for rejected unsupported claims.
