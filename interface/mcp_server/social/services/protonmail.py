from typing import Optional, List, Dict
import asyncio
import os
import sys
import time
import warnings
from contextlib import redirect_stdout
from pathlib import Path
from social.services import env, err
from system.userspace import user_state_path

# Suppress CryptographyDeprecationWarning for TripleDES (used by protonmail-api-client)
# Note: CryptographyDeprecationWarning subclasses UserWarning, not DeprecationWarning.
warnings.filterwarnings(
    "ignore",
    category=Warning,
    message="TripleDES has been moved to cryptography.hazmat.decrepit.ciphers.algorithms.TripleDES",
)

# Global client cache (kept alive across tool calls)
_client_cache = None
_cache_username = None

def _session_file() -> str:
    """Resolve session path lazily (per-call, not import-time).

    user_state_path() depends on the active user id (AIKO_USER_ID). Resolving
    at import time freezes the path to whatever user (often guest) was active
    during the first import. Resolving per call keeps OppaAI's session under
    ~/.aiko/OppaAI/profile/ even if the module was first imported pre-login.
    """
    try:
        return str(user_state_path("profile/protonmail_session.pickle"))
    except Exception:
        # Fallback to the import-time constant if userspace resolution fails.
        return _SESSION_FILE_FALLBACK

# protonmail-api-client stores sessions as a binary pickle, not JSON.
# Kept as fallback only; new code must call _session_file().
_SESSION_FILE_FALLBACK = None
try:
    _SESSION_FILE_FALLBACK = str(user_state_path("profile/protonmail_session.pickle"))
except Exception:
    _SESSION_FILE_FALLBACK = os.path.expanduser("~/.aiko/OppaAI/profile/protonmail_session.pickle")
# Legacy import-time constant (kept so external imports don't break).
_SESSION_FILE = _SESSION_FILE_FALLBACK


def _is_invalid_refresh_error(exc: Exception) -> bool:
    """True when the exception looks like an expired/invalid refresh token.

    protonmail-api-client raises e.g.:
      "Can't update tokens, status: 422 json: {'Error': 'Invalid refresh token', 'Code': 10013}"
    """
    msg = f"{exc}".lower()
    return (
        "invalid refresh token" in msg
        or "code" in msg and "10013" in msg
        or ("422" in msg and "refresh" in msg)
    )


def _is_captcha_or_abuse_error(exc: Exception) -> bool:
    """True when Proton is gating the login behind human verification.

    Observed server-side (Sept 2026): POST core/v4/auth answers
      Code 9001 "For security reasons, please complete CAPTCHA ..."
    and the client's CAPTCHA auto-solver then dies inside its own token
    refresh with 422/Code 10013 — so the 422 below *masks* the CAPTCHA
    gate. Match both the direct and the masked shape here.
    """
    msg = f"{exc}".lower()
    kind = type(exc).__name__.lower()
    return (
        "captcha" in msg
        or "captcha" in kind
        or "9001" in msg
        or "appeal-abuse" in msg
        or "human verification" in msg
        or "humanverification" in msg
        or "too many recent logins" in msg
        or "2028" in msg
    )


def _non_interactive_2fa_code() -> str:
    """Supply a 2FA/TOTP code without touching stdin.

    The MCP server runs on stdio transport: the library's default
    getter (input("enter 2FA code:")) would consume JSON-RPC bytes as a
    "code" and hang/corrupt the wire. Prefer an explicit one-shot code,
    else a TOTP secret, else fail with an actionable message.
    """
    one_shot = (env("PROTONMAIL_2FA_CODE") or "").strip().replace(" ", "")
    if one_shot:
        return one_shot
    secret = (env("PROTONMAIL_TOTP_SECRET") or "").strip().replace(" ", "")
    if secret:
        try:
            import pyotp
        except ImportError:
            raise RuntimeError(
                "ProtonMail requires 2FA and PROTONMAIL_TOTP_SECRET is set "
                "but the 'pyotp' package is not installed"
            )
        return pyotp.TOTP(secret).now()
    raise RuntimeError(
        "ProtonMail account requires two-factor authentication, but neither "
        "PROTONMAIL_2FA_CODE nor PROTONMAIL_TOTP_SECRET is set "
        "(interactive stdin prompt is disabled on the MCP stdio transport)"
    )


# --- Anti-hammering backoff -------------------------------------------------
# Proton flags accounts/IPs that log in too often (Code 9001 CAPTCHA gate,
# Code 2028 "too many recent logins"). The scheduler retries failed email
# polls, so without a backoff every poll burns another login attempt and
# deepens the flag. After an abuse-class failure, fresh password logins are
# refused for PROTONMAIL_AUTH_COOLDOWN_S (default 600s). The marker lives on
# disk next to the session pickle so concurrent MCP processes back off too;
# delete the marker file to retry immediately. Set the env var to 0 to
# disable.
_last_auth_failure_at = 0.0
_last_auth_failure_reason = ""


def _auth_cooldown_seconds() -> int:
    try:
        return max(0, int(env("PROTONMAIL_AUTH_COOLDOWN_S", "600") or "600"))
    except (TypeError, ValueError):
        return 600


def _cooldown_file() -> str:
    try:
        base = os.path.dirname(_session_file())
    except Exception:
        base = os.path.expanduser("~/.aiko/OppaAI/profile")
    return os.path.join(base, "protonmail_login_cooldown")


def _auth_backoff_remaining() -> float:
    """Seconds left on the login backoff, 0 when logins are allowed."""
    cooldown = _auth_cooldown_seconds()
    if cooldown <= 0:
        return 0.0
    newest = _last_auth_failure_at
    try:
        marker = _cooldown_file()
        if os.path.exists(marker):
            newest = max(newest, os.path.getmtime(marker))
    except OSError:
        pass
    return max(0.0, (newest + cooldown) - time.time())


def _record_auth_failure(reason: object = "") -> None:
    """Start/refresh the login backoff (memory + on-disk marker)."""
    global _last_auth_failure_at, _last_auth_failure_reason
    _last_auth_failure_at = time.time()
    _last_auth_failure_reason = str(reason)[:200]
    try:
        marker = _cooldown_file()
        Path(marker).parent.mkdir(parents=True, exist_ok=True)
        Path(marker).write_text(f"{_last_auth_failure_at}\n{_last_auth_failure_reason}\n")
    except OSError as e:
        print(f"[PROTONMAIL] Could not write cooldown marker: {e}", file=sys.stderr, flush=True)


def _clear_auth_backoff() -> None:
    """A login succeeded — lift any backoff."""
    global _last_auth_failure_at, _last_auth_failure_reason
    _last_auth_failure_at = 0.0
    _last_auth_failure_reason = ""
    try:
        marker = _cooldown_file()
        if os.path.exists(marker):
            os.remove(marker)
    except OSError:
        pass


def _cooldown_error() -> Dict:
    remaining = int(_auth_backoff_remaining())
    reason = _last_auth_failure_reason
    try:
        if not reason and os.path.exists(_cooldown_file()):
            reason = Path(_cooldown_file()).read_text().splitlines()[1] if len(Path(_cooldown_file()).read_text().splitlines()) > 1 else ""
    except (OSError, IndexError):
        pass
    detail = f" (last failure: {reason})" if reason else ""
    return {
        "ok": False,
        "provider": "protonmail",
        "error": (
            f"ProtonMail password login paused for ~{remaining}s to avoid "
            f"deepening Proton's anti-abuse/CAPTCHA flag{detail}. Wait for the "
            f"backoff to expire, or delete {_cooldown_file()} to retry now "
            f"(risks re-triggering the flag). Set PROTONMAIL_AUTH_COOLDOWN_S=0 "
            f"to disable the backoff."
        ),
    }

# stderr-bound print for ProtonMail's internal logger.
# ProtonMail.__init__ defaults logging_func=print (stdout). Since the MCP
# server runs on stdio transport any write to stdout corrupts the JSON-RPC
# wire, causing the client to hang waiting for a valid response.
def _stderr_print(*args, **kwargs):
    kwargs.setdefault("file", sys.stderr)
    kwargs.setdefault("flush", True)
    print(*args, **kwargs)


def _run_client_call(method, *args, **kwargs):
    """Keep third-party client output (including tqdm progress bars) off MCP stdout."""
    with redirect_stdout(sys.stderr):
        return method(*args, **kwargs)


# protonmail-api-client pins its x-pm-appversion headers at release time
# (currently web-mail@5.0.66.5 in 2.4.3). Proton rejects stale versions on
# the mail API with HTTP 422 Code 5003 ("out of date, please refresh"),
# whose payload carries no Messages/Counts key — surfacing here as
# KeyError 'Messages'/'Counts'. Override with the live web-client version
# (scraped from mail.proton.me's bundle, Sep 2026) after auth, for both
# fresh logins and loaded sessions (the pickle stores the stale header).
# Env override so the next Proton bump is a config change, not a release.
def _mail_app_version() -> str:
    return env("PROTONMAIL_MAIL_APP_VERSION", "web-mail@5.0.133.5") or "web-mail@5.0.133.5"


def _apply_mail_app_version(client) -> None:
    try:
        client.session.headers["x-pm-appversion"] = _mail_app_version()
    except Exception as e:
        print(f"[PROTONMAIL] Could not set app version header: {e}", file=sys.stderr, flush=True)


def _fetch_all_pages_no_count(client, page_size: int = 150, max_pages: int = 20) -> list:
    """Fetch every message page without touching the /count endpoint.

    Stops at the first short/empty page. Bounded by max_pages so a lying
    server can't page forever.
    """
    seen: list = []
    for page in range(max_pages):
        batch = _run_client_call(client.get_messages_by_page, page, page_size)
        if not batch:
            break
        seen.extend(batch)
        if len(batch) < page_size:
            break
    return seen


def _get_client():
    """Return an authenticated client using the documented session flow."""
    global _client_cache, _cache_username
    try:
        from protonmail import ProtonMail
    except ImportError:
        return None, {"ok": False, "error": "protonmail-api-client not installed", "provider": "protonmail"}
    username = env("PROTONMAIL_USERNAME")
    password = env("PROTONMAIL_PASSWORD")
    if not username:
        return None, {"ok": False, "error": "PROTONMAIL_USERNAME not set", "provider": "protonmail"}
    if _client_cache is not None and _cache_username == username:
        print("[PROTONMAIL] Using cached client", file=sys.stderr, flush=True)
        _apply_mail_app_version(_client_cache)
        return _client_cache, None
    session_file = _session_file()
    if not os.path.exists(session_file) and not password:
        return None, {"ok": False, "error": "PROTONMAIL_PASSWORD not set for first login", "provider": "protonmail"}
    if not os.path.exists(session_file) and _auth_backoff_remaining() > 0:
        # No usable session and we recently hit Proton's anti-abuse gate:
        # refuse to burn another login attempt.
        print(f"[PROTONMAIL] Login backoff active ({int(_auth_backoff_remaining())}s left); skipping password login", file=sys.stderr, flush=True)
        return None, _cooldown_error()
    print(f"[PROTONMAIL] Authenticating as {username[:3]}{chr(42) * max(0, len(username) - 3)}...", file=sys.stderr, flush=True)
    print(f"[PROTONMAIL] Session file exists: {os.path.exists(session_file)} ({session_file})", file=sys.stderr, flush=True)
    try:
        client = ProtonMail(logging_func=_stderr_print)
        if os.path.exists(session_file):
            print(f"[PROTONMAIL] Loading session: {session_file}", file=sys.stderr, flush=True)
            try:
                _run_client_call(client.load_session, session_file, auto_save=True)
            except Exception as load_err:
                # Stale/expired refresh token (422 Code 10013): the saved
                # pickle can never refresh again. Delete it and fall through
                # to a fresh password login instead of failing forever.
                if _is_invalid_refresh_error(load_err) and password:
                    if _auth_backoff_remaining() > 0:
                        print(f"[PROTONMAIL] Saved session expired but login backoff is active ({int(_auth_backoff_remaining())}s left); not re-logging in", file=sys.stderr, flush=True)
                        raise
                    print(f"[PROTONMAIL] Saved session expired ({load_err}); deleting stale file and re-logging in...", file=sys.stderr, flush=True)
                    try:
                        os.remove(session_file)
                    except OSError:
                        pass
                    _client_cache = None
                    _cache_username = None
                    client = ProtonMail(logging_func=_stderr_print)
                    _password_login(client, username, password)
                    Path(session_file).parent.mkdir(parents=True, exist_ok=True)
                    _run_client_call(client.save_session, session_file)
                    print(f"[PROTONMAIL] Session re-saved: {session_file}", file=sys.stderr, flush=True)
                else:
                    raise
            else:
                print("[PROTONMAIL] Session loaded successfully", file=sys.stderr, flush=True)
        else:
            print("[PROTONMAIL] No saved session; performing login...", file=sys.stderr, flush=True)
            _password_login(client, username, password)
            Path(session_file).parent.mkdir(parents=True, exist_ok=True)
            _run_client_call(client.save_session, session_file)
            print(f"[PROTONMAIL] Session saved: {session_file}", file=sys.stderr, flush=True)
        _clear_auth_backoff()
        _apply_mail_app_version(client)
        _client_cache = client
        _cache_username = username
        return client, None
    except Exception as e:
        # Never cache a failed client; next call retries (e.g. after the user
        # fixes credentials or deletes the stale session manually).
        _client_cache = None
        _cache_username = None
        print(f"[PROTONMAIL] Authentication failed: {e}", file=sys.stderr, flush=True)
        hint = ""
        if _is_invalid_refresh_error(e) and not password:
            hint = " (saved session expired and PROTONMAIL_PASSWORD is not set, so automatic re-login is impossible — set the password or delete the session pickle)"
        if _is_abuse_gate(e, session_existed=os.path.exists(session_file)):
            # Proton is demanding human/CAPTCHA verification (or we are rate
            # limited). Retrying the login immediately only deepens the flag,
            # so start the backoff and explain what to do.
            _record_auth_failure(e)
            hint = (
                " — Proton is gating logins behind human/CAPTCHA verification "
                "(anti-abuse) or rate-limiting this account/IP, so automatic "
                "password login cannot proceed. What helps: (1) stop automated "
                "retries for a while so the flag can decay (this client now "
                "backs off automatically); (2) log in once in a real browser "
                "from this network and solve the CAPTCHA; (3) if it persists, "
                "appeal at https://proton.me/support/appeal-abuse"
            )
        return None, {"ok": False, "error": f"authentication failed: {e}{hint}", "provider": "protonmail"}


def _is_abuse_gate(exc: Exception, session_existed: bool) -> bool:
    """True when the error means 'Proton blocked the login attempt itself'.

    A 422/Code 10013 with NO session file can never be a genuinely stale
    saved session (there is nothing to refresh) — it is the masked shape of
    the CAPTCHA/anti-abuse gate (proven Sept 2026: the client's CAPTCHA
    auto-solver dies in its own token refresh). Any direct CAPTCHA/rate-limit
    signal counts regardless of session state.
    """
    if _is_captcha_or_abuse_error(exc):
        return True
    return _is_invalid_refresh_error(exc) and not session_existed


def _password_login(client, username: str, password: str) -> None:
    """Run the library login with a non-interactive 2FA getter and verify it.

    The library logs "login failure" but does NOT raise when SRP
    verification fails (e.g. wrong password) — it just continues into the
    fork/cookies flow, which later explodes as a confusing 422. Check
    authenticated() explicitly so bad credentials surface as bad
    credentials.
    """
    _run_client_call(client.login, username, password, _non_interactive_2fa_code)
    user = getattr(client, "user", None)
    authenticated = user.authenticated() if user is not None else False
    if not authenticated:
        raise RuntimeError(
            "ProtonMail SRP verification failed (server rejected the login "
            "proof — most likely a wrong PROTONMAIL_PASSWORD)"
        )


def clear_cached_client() -> None:
    """Drop the in-memory client so the next call re-authenticates."""
    global _client_cache, _cache_username
    _client_cache = None
    _cache_username = None


def _drop_stale_session(reason: object = "") -> None:
    """Delete the saved pickle and clear the in-memory client.

    Called when ProtonMail reports an invalid/expired refresh token
    (HTTP 422 Code 10013). The pickle can never refresh again, so keeping
    it only guarantees the same failure on every subsequent call.
    """
    clear_cached_client()
    try:
        session_file = _session_file()
    except Exception:
        return
    try:
        if os.path.exists(session_file):
            os.remove(session_file)
            print(f"[PROTONMAIL] Deleted stale session ({reason}): {session_file}", file=sys.stderr, flush=True)
    except OSError as e:
        print(f"[PROTONMAIL] Could not delete stale session: {e}", file=sys.stderr, flush=True)


async def _reauth_fresh_client():
    """Force a fresh password login after dropping the stale session.

    Returns (client, error_dict) like _get_client(). Requires
    PROTONMAIL_PASSWORD to be set; without it automatic recovery is
    impossible and the caller must surface a clear error.
    """
    _drop_stale_session("invalid refresh token")
    if not env("PROTONMAIL_PASSWORD"):
        return None, {"ok": False, "error": "saved ProtonMail session expired and PROTONMAIL_PASSWORD is not set, so automatic re-login is impossible — set the password or log in manually", "provider": "protonmail"}
    if _auth_backoff_remaining() > 0:
        return None, _cooldown_error()
    return _get_client()


def get_client():
    """Get ProtonMail client. Returns (client, error_dict) or (client, None)."""
    return _get_client()


# ProtonMail label ids. Messages can carry several labels at once (e.g.
# Inbox + a custom label), and the API gives no ordering guarantee over
# msg.labels, so folder matching below checks membership across the full
# list rather than assuming the target label sits at index 0.
_FOLDER_LABEL_MAP = {
    "inbox": "0",
    "spam": "4",
    "trash": "3",
}


async def read_messages(client, folder: str, unread: bool, max_results: int, query: str, list_only: bool, message_id: str = "") -> Dict:
    """Read messages using ProtonMail client."""
    try:
        return await _read_messages_inner(client, folder, unread, max_results, query, list_only, message_id)
    except Exception as e:
        # Token refresh happens lazily on the first API call, not inside
        # load_session — so an expired pickle surfaces HERE as 422/10013.
        # Drop it, fresh-login once, and retry so one stale file doesn't
        # break every poll until manual intervention.
        if _is_invalid_refresh_error(e):
            print(f"[PROTONMAIL] Read hit expired session ({e}); retrying with fresh login...", file=sys.stderr, flush=True)
            fresh, err_resp = await _reauth_fresh_client()
            if err_resp:
                return err_resp
            try:
                return await _read_messages_inner(fresh, folder, unread, max_results, query, list_only, message_id)
            except Exception as retry_e:
                if _is_invalid_refresh_error(retry_e):
                    _drop_stale_session(retry_e)
                return {"ok": False, "error": f"authentication failed: {retry_e}", "provider": "protonmail"}
        return {"ok": False, "error": f"read failed: {e}", "provider": "protonmail"}


async def _read_messages_inner(client, folder: str, unread: bool, max_results: int, query: str, list_only: bool, message_id: str = "") -> Dict:
    """Read messages using ProtonMail client (no retry; wrapper handles re-auth)."""
    try:
        # Translate generic folder name to a ProtonMail label id.
        folder_lower = folder.lower()
        protonmail_label = _FOLDER_LABEL_MAP.get(folder_lower, folder_lower)

        # Get all messages. NOTE: client.get_messages() calls the
        # mail/v4/messages/count endpoint first, which currently raises
        # KeyError('Counts') (Proton response drift, Sep 2026). Fall back to
        # count-free paging via get_messages_by_page() instead of failing
        # the whole inbox read.
        try:
            all_messages = await asyncio.to_thread(_run_client_call, client.get_messages)
        except KeyError:
            all_messages = await asyncio.to_thread(_fetch_all_pages_no_count, client)

        # Filter by folder (0=inbox, 3=trash, 4=spam).
        messages = []
        folder_counts = {}

        for msg in all_messages:
            label_list = []
            if getattr(msg, "labels", None):
                raw_labels = msg.labels if isinstance(msg.labels, list) else [msg.labels]
                label_list = [str(l) for l in raw_labels]

            # Tally every label the message carries (not just the first)
            # so folder_counts actually reflects what's on the account.
            for lbl in label_list or [""]:
                folder_counts[lbl] = folder_counts.get(lbl, 0) + 1

            # Match if the resolved target label id is anywhere in this
            # message's label list - don't assume position/order.
            in_target_folder = protonmail_label in label_list
            is_unread = getattr(msg, "unread", False)

            if in_target_folder and (not unread or is_unread):
                messages.append(msg)

        print(f"[EMAIL] Provider=protonmail label={protonmail_label} Folder counts: {folder_counts}, filtered={len(messages)}", file=sys.stderr, flush=True)

        # If message_id provided, return full message
        if message_id:
            target_msg = None
            for msg in messages:
                if getattr(msg, "id", "") == message_id:
                    target_msg = msg
                    break
            if target_msg is None:
                return {"ok": False, "error": f"message not found: {message_id}", "provider": "protonmail"}

            full = await asyncio.to_thread(_run_client_call, client.read_message, target_msg)
            return {
                "ok": True,
                "provider": "protonmail",
                "id": message_id,
                "from": full.sender.address if full.sender else "",
                "subject": full.subject or "",
                "date": str(full.time) if full.time else "",
                "body": full.body or "",
            }

        # Filter by query
        if query:
            q = query.lower()
            messages = [m for m in messages if q in (m.subject or "").lower() or q in (m.sender.address if m.sender else "").lower()]

        results = []
        for i, msg in enumerate(messages[:max_results]):
            if list_only:
                results.append({
                    "id": getattr(msg, "id", ""),
                    "from": msg.sender.address if getattr(msg, "sender", None) else "",
                    "subject": getattr(msg, "subject", "") or "",
                    "date": str(getattr(msg, "time", "") or ""),
                    "snippet": "",
                })
                continue
            try:
                print(f"[EMAIL] Reading message {i+1}/{max_results} ({getattr(msg, 'id', 'unknown')[:10]}...)...", file=sys.stderr, flush=True)
                full = await asyncio.to_thread(_run_client_call, client.read_message, msg)
                print(f"[EMAIL] Read message {i+1}/{max_results} OK", file=sys.stderr, flush=True)
                results.append({
                    "id": getattr(msg, "id", ""),
                    "from": full.sender.address if full.sender else "",
                    "subject": full.subject or "",
                    "date": str(full.time) if full.time else "",
                    "snippet": (full.body or ""),
                })
            except Exception:
                continue

        return {"ok": True, "provider": "protonmail", "count": len(results), "messages": results}
    except Exception as e:
        # Let expired-session errors bubble to the wrapper for fresh-login retry.
        if _is_invalid_refresh_error(e):
            raise
        return {"ok": False, "error": f"read failed: {e}", "provider": "protonmail"}


async def _send_message_inner(client, recipients: List[str], subject: str, body: str, cc: List[str], bcc: List[str], attachments: List[Dict]) -> Dict:
    """Send email using ProtonMail client (no retry; wrapper handles re-auth)."""
    # protonmail-api-client requires two-step: create_message then send_message
    new_message = await asyncio.to_thread(
        _run_client_call,
        client.create_message,
        recipients=recipients,
        subject=subject,
        body=body,
        cc=cc if cc else [],
        bcc=bcc if bcc else [],
    )

    # Send the created message
    sent_message = await asyncio.to_thread(_run_client_call, client.send_message, new_message)

    return {
        "ok": True,
        "provider": "protonmail",
        "message_id": getattr(sent_message, "id", "unknown"),
        "status": "sent"
    }


async def send_message(client, recipients: List[str], subject: str, body: str, cc: List[str], bcc: List[str], attachments: List[Dict]) -> Dict:
    """Send email using ProtonMail client (retries once on expired session)."""
    if not recipients:
        return {"ok": False, "error": "recipients required", "provider": "protonmail"}

    try:
        return await _send_message_inner(client, recipients, subject, body, cc, bcc, attachments)
    except Exception as e:
        if _is_invalid_refresh_error(e):
            print(f"[PROTONMAIL] Send hit expired session ({e}); retrying with fresh login...", file=sys.stderr, flush=True)
            fresh, err_resp = await _reauth_fresh_client()
            if err_resp:
                return err_resp
            try:
                return await _send_message_inner(fresh, recipients, subject, body, cc, bcc, attachments)
            except Exception as retry_e:
                if _is_invalid_refresh_error(retry_e):
                    _drop_stale_session(retry_e)
                return {"ok": False, "error": f"authentication failed: {retry_e}", "provider": "protonmail"}
        return {"ok": False, "error": f"send failed: {e}", "provider": "protonmail"}


async def _delete_message_inner(client, message_id: str) -> Dict:
    """Delete email using ProtonMail client (no retry; wrapper handles re-auth)."""
    messages = await asyncio.to_thread(_run_client_call, client.get_messages)
    target = next((msg for msg in messages if getattr(msg, "id", "") == message_id), None)
    if target is None:
        return {"ok": False, "error": f"message not found: {message_id}", "provider": "protonmail"}
    await asyncio.to_thread(_run_client_call, client.delete_messages, [target])
    return {"ok": True, "provider": "protonmail", "message_id": message_id, "status": "deleted"}


async def delete_message(client, message_id: str) -> Dict:
    """Delete email using ProtonMail client (retries once on expired session)."""
    if not message_id:
        return {"ok": False, "error": "message_id required", "provider": "protonmail"}

    try:
        return await _delete_message_inner(client, message_id)
    except Exception as e:
        if _is_invalid_refresh_error(e):
            print(f"[PROTONMAIL] Delete hit expired session ({e}); retrying with fresh login...", file=sys.stderr, flush=True)
            fresh, err_resp = await _reauth_fresh_client()
            if err_resp:
                return err_resp
            try:
                return await _delete_message_inner(fresh, message_id)
            except Exception as retry_e:
                if _is_invalid_refresh_error(retry_e):
                    _drop_stale_session(retry_e)
                return {"ok": False, "error": f"authentication failed: {retry_e}", "provider": "protonmail"}
        return {"ok": False, "error": f"delete failed: {e}", "provider": "protonmail"}