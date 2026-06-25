# Aiko-chan Test Suites

This directory organizes the industrial test plan described in [`docs/TESTS.md`](../docs/TESTS.md). The current files are suite specifications/checklists; executable tests should be added under the matching suite as the codebase becomes more automated.

| Directory | Purpose | Typical trigger |
|---|---|---|
| `unit/` | Pure Python functions and safety helpers. | Every code change. |
| `integration/` | Local service boundaries: LLM, SearXNG, MioTTS, ASR, sqlite-vec, WebUI. | Adapter/config/dependency changes. |
| `system/` | Full user journeys through TUI, WebUI, voice, and agentic mode. | Release candidate. |
| `load/` | Concurrent users, queue pressure, large memory DBs, long prompts. | Demo/field hardening. |
| `error/` | Fault injection and graceful degradation. | Release candidate and incident fixes. |
| `performance/` | Latency/resource measurements. | Model, hardware, or dependency changes. |
| `stability/` | Soak, restart, reconnect, leak checks. | Long-running deployments. |
| `functionality/` | Phase/feature acceptance. | Feature completion. |

## Naming Convention for Future Executable Tests

- Python unit tests: `tests/unit/test_<module>_<behavior>.py`
- Integration tests: `tests/integration/test_<service>_<contract>.py`
- Scenario scripts: `tests/system/<scenario>.md` or `tests/system/<scenario>.py`
- Artifacts/logs: keep outside git, e.g. `workspace/test-runs/<date>/`

## Minimum Metadata for Manual Test Runs

Capture the following with every G3+ run:

- git commit SHA;
- hardware target and OS;
- model names/quantization;
- relevant `.env` values with secrets redacted;
- commands executed;
- pass/fail result;
- logs, screenshots, or measurements.
