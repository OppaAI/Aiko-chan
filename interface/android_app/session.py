"""Shared authentication policy for companion HTTP clients."""
import os

from fastapi import HTTPException, Request


async def require_companion_session(request: Request) -> dict:
    from interface.webui import auth
    try:
        session = await auth.require_session(request)
    except HTTPException as exc:
        # Legacy clients have no cookie jar. Opt in only on a trusted network;
        # invalid supplied credentials must never become an owner session.
        enabled = os.getenv('AIKO_COMPANION_OWNER_FALLBACK', '0').lower() in {'1', 'true', 'yes', 'on'}
        owner = (os.getenv('AIKO_USER_ID') or '').strip()
        if exc.status_code == 401 and not request.cookies.get('session_id') and enabled and owner:
            return {'user_id': owner, 'username': owner}
        raise
    return await auth.require_accepted_session(session)
