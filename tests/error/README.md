# Error Injection Test Suite

Verify graceful degradation and clear troubleshooting output.

## Faults to Inject

- LLM server down/wrong model/timeout;
- SearXNG down or returning no results;
- MioTTS down or audio device invalid;
- ASR model missing, microphone unavailable, noisy input;
- sqlite path unwritable, corrupt DB copy, disk full;
- corrupt `workspace/schedule.json`;
- WebSocket invalid JSON/binary frames;
- path traversal attempts in workspace and repo tools;
- absent GitHub reflection credentials.

## Expected Behavior

Aiko should not crash, should not leak secrets, should explain partial failure honestly, and should remain usable in reduced-capability mode where possible.
