# Configuration and deployment notes

Aiko loads process environment overrides, then indexed YAML defaults, then local
secret values for keys YAML did not define. Empty YAML values allow a secret or
code default to supply the value. `override=True` retains its explicit reload
behavior: later secret values can replace YAML and process values. Failed loads
publish no partial configuration.

Dotenv values are literal. `${NAME}` in `.env.age` or the development `.env` is
stored and read as that exact text; supply the resolved value when you want a
URL or credential assembled from other values. Settings preserves backslashes,
quotes and multiline secrets. NaN/infinity numeric settings use the code default.

Use `util/edit_dotenv.sh` to edit encrypted secrets with `age` and `nvim`. The
editor uses a private `/dev/shm` directory and disables persistent editor state.
`AGE_KEY`, `AGE_KEY_PUB` and `ENV_AGE_PATH` accept explicit paths; relative paths
resolve below `USER_SPACE_ROOT` (default `~/.aiko`). The previous encrypted file
is preserved if encryption or verification fails.

`util/sync_app_server_url.sh --print-only` resolves `AIKO_PUBLIC_BASE_URL` with the
same loader. Without `--print-only`, it atomically updates `local.properties` in
the configured `GAMES_DIR`, `LINGO_DIR` and `ONMYOJI_DIR` checkouts. Rebuild APKs to
apply a URL change. No decrypted temporary dotenv is created.

User directories inside `USER_SPACE_ROOT` must be actual directories, rather
than symlinks to another user's state or another location. A mount or symlink for
the configured root itself remains usable. Display names do not link accounts or
grant the built-in owner identity; that mapping uses the verified provider ID.
Legacy owner sessions without that verified ID require login again.

The Modal scripts under `backend/modal/` require their documented remote secrets
and SDK/model dependencies. Request limits prevent oversized workloads; they do
not constitute a live GPU quality or capacity measurement. Deploy the updated
scripts to apply their validation changes. OSRM uses the MLD
extract/partition/customize pipeline, followed by `osrm-routed --algorithm mld`.

The backup and physical deployment checks in `BUG_AUDIT.md` remain the required
live validation record; automated tests do not substitute for a restore trial.
