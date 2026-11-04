# Playground self-coding with an 8K context

Use a separate Aiko-Playground checkout (with `GOALS.md`, `loop/WORKER.md`,
and `sandbox/run.py`). Set `AIKO_PLAYGROUND_DIR` to its absolute path when
it is not beside Aiko-chan or at `~/Aiko-Playground`. The file tools use
that same setting; a separate `AIKO_WORKER_ROOTS` setting is unnecessary.

The ordinary **Aiko-Playground idle build** schedule record must have
`worker_mode: true` and `lean_context: true`. Startup re-syncs these owned
fields on existing records, preserving enabled/paused state and cadence.
It runs only after the configured idle threshold. `loop/disabled` stops it.
Restart Aiko after updating the code so the scheduler and tools reload.

Keep `LLM_CTX_SIZE=8192` in sync with the inference server's actual context.
Worker mode sends a short coding system prompt, seven tool schemas, and the
scheduled task. It does **not** inject SOUL/persona, personal memories,
knowledge retrieval, similar experiences, or previous chat. It also skips
capability embedding at the AikoThink entry point. Normal chat retains its
usual context and behavior.

For each goal, read only the relevant instructions/goal/checkpoint; use small
paged reads. Create PLAN.md, source, tests, and CHECKPOINT.md with
`repo_write_file`; edit existing files with `code_apply_patch`. Keep each
write/patch around 1200 characters so it fits a 512-token tool response.
Use `sandbox_run` for Python programs and self-contained test scripts. It
invokes the Playground's `sandbox/run.py` and checks the script's JSON result,
including failures even when the wrapper exits successfully. This runner
provides path/time limits, not OS-level isolation of arbitrary Python code.

Only writes and executions inside `work/<goal>/` receive the scheduled
worker's authorization. Other repositories, Playground control files, shell
commands, secrets paths, and restricted directories remain protected. Normal
interactive coding still asks for approval; conscience checks and explicit
confirmation preferences remain in effect. Completion notifications and
publishing are not automatically authorized by a local coding session; save
an email draft when those tools are unavailable.

Generated source stays on disk. Completed tool-call arguments are compacted
instead of repeatedly carrying whole files; context trimming removes linked
tool calls/results together and retains the task. Resume the next bounded
session from CHECKPOINT.md rather than trying to fit an entire project in 8K.

Validation (no live inference server needed):

```bash
python -m pytest tests/unit/test_playground_worker.py tests/unit/test_worker_mode.py \
  tests/unit/test_agentic_context_fit.py tests/unit/test_coding_safety_retries.py \
  tests/unit/test_think_routing.py tests/unit/test_schedule_crud.py -q
```

The scripted-model test exercises real file creation, sandbox subprocesses,
and checkpointing while checking every assembled request against 8192 tokens.
It tests execution plumbing and budgeting; live model quality still requires
an actual model server and a real goal.

Normal interactive tool approvals are one-shot: the saved action is consumed
before execution, including when the tool reports failure. A failure may follow
an external side effect, so retry by making a fresh request after checking the
outcome. Calls receive unique resumable run IDs, and failed approval storage
reports that nothing executed. The scheduled worker's scoped authorization
remains separate from these interactive approvals.

Experience-promoted workflows retain sequential dependencies. Promotion uses
atomic file replacement and a shared write lock, preserves invalid existing
playbooks for repair, and reuses a source experience's existing plan rather
than creating duplicates on a retried promotion.
