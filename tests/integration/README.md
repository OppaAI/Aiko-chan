# Integration Test Suite

Validate contracts between Aiko and local services.

## Boundaries

- OpenAI-compatible LLM at `LLM_BASE_URL`.
- SearXNG search at `SEARXNG_URL`.
- MioTTS synthesis at `MIOTTS_API_URL`.
- SenseVoice/Silero ASR/VAD initialization.
- sqlite-vec + fastembed persistent memory.
- WebUI HTTP/WebSocket bridge.
- schedule runner and `workspace/schedule.json`.

## Required Evidence

- service health output;
- request/response shape;
- timeout behavior;
- controlled failure behavior when the service is stopped or misconfigured.
