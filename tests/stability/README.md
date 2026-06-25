# Stability Test Suite

Long-running and recovery-focused tests.

## Soak Runs

- 8-hour text idle;
- 100-turn mixed conversation;
- 2-hour intermittent voice session;
- 8-hour WebUI connected browser session;
- scheduler due-job overnight run;
- service restart recovery for LLM, SearXNG, MioTTS, and browser reconnects.

## Failure Conditions

Unbounded memory growth, stuck audio devices, unrecoverable WebSocket reconnect loops, corrupted memory/schedule files, or hung shutdowns fail the suite.
