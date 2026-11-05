# Settings Studio

Open **Settings** in Aiko's WebUI, or `/studio/settings/`. Sign in as the
creator or an administrator: these settings affect the whole running instance.
Ordinary chat accounts cannot read or write this API.

The page follows Aidon's soft cards, serif heading, compact tabs and light/dark
palette. Search spans every category, including advanced settings. Each field
shows its environment key, YAML source, and when changes take effect. Only
changed fields are submitted; switching tabs preserves unsaved edits.

## Coverage

All leaves of the nine runtime YAML mappings in `config/index.yaml` appear in
the editor. The existing curated fields retain their bounds and descriptions;
additional fields derive their types from YAML and display nearby comments.
Lists use a JSON list of strings. Multiline text has a text editor. New scalar
settings in these files are discovered automatically, without frontend changes.
File paths are fixed on the server and symlinked config files are rejected.

The page also exposes previously environment-only controls for recall timeouts,
per-source recall token caps, agent request budget, embedding/intent caches,
Fly propagation budget/hops/seeds/cadence, and RAM release cadence. Defaults are
shown with a **code default** badge and are only written when edited. They are
useful for an 8 GB Jetson; smaller caches and Fly budgets trade throughput or
coverage for lower resource use. No model size, hardware, or budget defaults
are changed merely by opening or saving another field.

`index.yaml` and `tools.yaml` are listed under Configuration sources. They remain
on-disk edits: they define bootstrap load order and executable tool definitions.
State-root and age-identity locations are visible but read-only because changing
them requires migrating state or key material. This page does not edit user
profiles, personality documents, model artifacts, or arbitrary repository YAML.

## Applying changes

These six controls apply automatically to **the next chat request** after Save:

| Control | Purpose |
|---|---|
| `TEMPERATURE` | Sampling randomness |
| `TOP_P` | Nucleus sampling |
| `TOP_K` | Candidate sampling limit |
| `LLM_MAX_TOKENS` | Maximum reply length |
| `REPEAT_PENALTY` | Repetition penalty |
| `REPEAT_LAST_N` | Repetition window |

Their consumers already read the environment for each request. An in-progress
reply continues with the parameters it received. Saving does not unload models
or call the bootstrap loader again. Other changes require restarting Aiko,
including model endpoints, credentials, ports, device paths, context assembly,
cached subsystem settings and Fly modes. This conservative policy avoids
partially rebuilt clients or incompatible in-memory model state.

A **restart pending** badge distinguishes a saved value from the running one.
It does not lock the field. Actual deployment environment overrides stay locked,
even when their current value matches the file; the deployment environment must
be changed at its source. Live controls are process-local: restart other Aiko
processes if running more than one instance against the same configuration.

## Secrets and persistence

The encrypted `.env.age` editor shows key names and set/unset state only.
Blank password inputs preserve the current value. New values for YAML credential
fields are encrypted into `.env.age`, and the corresponding YAML value is
cleared so bootstrap can load the encrypted value. Existing plaintext YAML
credentials are migrated only when explicitly changed. Keep the age identity
available to Aiko; the editor does not create or replace it.

YAML takes precedence over `.env.age`. An encrypted entry shadowed by YAML is
marked **edit in YAML group**. New secrets are also checked for this condition.
Passwords, tokens and API/private keys discovered in runtime YAML are masked.
The production backend decrypts in memory and verifies re-encryption before
replacing the ciphertext. It never writes decrypted `.env.age` content to disk.

Saves validate the entire batch, stage and parse-check YAML replacements, and
serialize in-process writers. Failed batches restore the original files. A
revision check rejects stale browser saves, including edits made on disk after
the page loaded. Reload before retrying a conflict. This is not a filesystem
transaction across a power failure; keep normal configuration/key backups.
