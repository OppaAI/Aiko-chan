#!/usr/bin/env bash
# Resolve the URL with Aiko's actual config loader; never write decrypted
# secrets to a temporary file. Rebuild APKs after changing local.properties.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export GAMES_DIR="${GAMES_DIR:-$(dirname "$REPO")/Aiko-Games}"
export LINGO_DIR="${LINGO_DIR:-$(dirname "$REPO")/Aiko-Lingo}"
export ONMYOJI_DIR="${ONMYOJI_DIR:-$(dirname "$REPO")/Aiko-Onmyoji}"
PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}" python3 - "$@" <<'PY'
import os
import sys
import tempfile
from pathlib import Path
from urllib.parse import urlsplit
from system.config import load_config

if sys.argv[1:] not in ([], ['--print-only']):
    raise SystemExit('usage: sync_app_server_url.sh [--print-only]')
load_config()
url = os.getenv('AIKO_PUBLIC_BASE_URL', '').strip()
parts = urlsplit(url)
if parts.scheme not in {'http', 'https'} or not parts.hostname or any(c in url for c in '\r\n\\'):
    raise SystemExit('AIKO_PUBLIC_BASE_URL must be a single HTTP(S) URL')
if sys.argv[1:] == ['--print-only']:
    print(url)
    raise SystemExit(0)
for name in ('GAMES_DIR', 'LINGO_DIR', 'ONMYOJI_DIR'):
    directory = Path(os.environ[name])
    if not (directory / 'app').is_dir():
        print(f'skip {directory} (not found)', file=sys.stderr)
        continue
    target = directory / 'local.properties'
    lines = target.read_text().splitlines() if target.exists() else []
    # Values are data, never a sed replacement expression or shell program.
    lines = [line for line in lines if line.split('=', 1)[0].strip() != 'aikoServerUrl']
    lines.append('aikoServerUrl=' + url)
    fd, tmp = tempfile.mkstemp(dir=directory, prefix='.local-properties-')
    try:
        with os.fdopen(fd, 'w') as stream:
            stream.write('\n'.join(lines) + '\n')
        os.replace(tmp, target)
    finally:
        Path(tmp).unlink(missing_ok=True)
    print(f'wrote aikoServerUrl -> {target}')
PY
