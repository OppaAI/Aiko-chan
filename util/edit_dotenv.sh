#!/usr/bin/env bash
# Edit encrypted dotenv on tmpfs; keep plaintext and editor state private.
# AGE_KEY defaults to <USER_SPACE_ROOT>/age-key.txt; AGE_KEY_PUB defaults to
# ${AGE_KEY}.pub. Pass the encrypted path or set ENV_AGE_PATH.
set -euo pipefail
umask 077
STATE_ROOT="${USER_SPACE_ROOT:-$HOME/.aiko}"
ENC="${1:-${ENV_AGE_PATH:-.env.age}}"
KEY="${AGE_KEY:-age-key.txt}"
[[ "$ENC" = /* ]] || ENC="$STATE_ROOT/$ENC"
[[ "$KEY" = /* ]] || KEY="$STATE_ROOT/$KEY"
PUB="${AGE_KEY_PUB:-${KEY}.pub}"
[[ "$PUB" = /* ]] || PUB="$STATE_ROOT/$PUB"
if [[ ! -f "$KEY" || ! -f "$PUB" ]]; then
    echo "age identity or recipient file not found" >&2
    exit 1
fi
WORKDIR="$(mktemp -d /dev/shm/aiko-env.XXXXXX)"
TMP="$WORKDIR/env"
CIPHER_TMP=""
cleanup() {
    rm -rf -- "$WORKDIR"
    [[ -z "$CIPHER_TMP" ]] || rm -f -- "$CIPHER_TMP"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
if [[ -f "$ENC" ]]; then
    age -d -i "$KEY" "$ENC" > "$TMP"
else
    : > "$TMP"
fi
# Disable swap, backup, persistent undo, and shada (which can retain file text).
nvim -n -i NONE -c "set nobackup nowritebackup noundofile" "$TMP"
CIPHER_TMP="$(mktemp "${ENC}.XXXXXX")"
age -R "$PUB" "$TMP" > "$CIPHER_TMP"
# Verify decryption without writing another plaintext file.
age -d -i "$KEY" "$CIPHER_TMP" | cmp -s -- "$TMP" -
mv -f -- "$CIPHER_TMP" "$ENC"
CIPHER_TMP=""
echo "re-encrypted -> $ENC"
