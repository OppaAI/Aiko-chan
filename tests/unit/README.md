# Unit Test Suite

Focus on deterministic functions that do not require live external services.

## Priority Areas

- CLI parsing and voice-command fuzzy matching in `main.py`.
- TTS text sanitization and chunk splitting in `core/speak.py`.
- stream sentence splitting in `core/think.py`.
- memory decay scoring and FTS query sanitization in `core/forget.py` and `core/memorize.py`.
- schedule calculation and legacy migration in `core/schedule.py`.
- safe path handling in `core/toolkit/common.py`, `planning.py`, and `architecture.py`.
- agent tool schema validation and failure classification in `core/agentic.py`.

## Suggested First Automation Targets

1. `test_voice_command_matching.py`
2. `test_sanitize_for_tts.py`
3. `test_schedule_calculate_next_due.py`
4. `test_toolkit_safe_paths.py`
5. `test_agentic_schema_dispatch.py`
