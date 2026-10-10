"""
cognition/consolidate/dream.py
Poetic daily reflection — feelings, FLUX image, Hugo post, GitHub push.

The blog/creative half of the daily reflection pipeline.
The factual/journal half lives in reflect.py.
"""
from __future__ import annotations

import base64
import os
import re
import textwrap
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests
from openai import OpenAI

from system.log import get_logger
from system.userspace import current_display_name

log = get_logger(__name__)

# ── blog/image config ────────────────────────────────────────────────────────

LLM_MODEL    = os.getenv("LLM_MODEL", "ministral")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:8080/v1")
_LLM_CLIENT: OpenAI | None = None

def _get_llm_client() -> OpenAI:
    global _LLM_CLIENT
    if _LLM_CLIENT is None:
        _LLM_CLIENT = OpenAI(base_url=LLM_BASE_URL, api_key=os.getenv("LLM_API_KEY", "") or "not-needed")
    return _LLM_CLIENT

GITHUB_TOKEN      = os.getenv("GITHUB_TOKEN", "")
GITHUB_REPO       = os.getenv("GITHUB_REPO", "")
GITHUB_BRANCH     = os.getenv("GITHUB_BRANCH", "main")
HUGO_CONTENT_PATH = os.getenv("HUGO_CONTENT_PATH", "content/posts")
HUGO_IMAGES_PATH  = os.getenv("HUGO_IMAGES_PATH", "static/images")
REFLECT_TAGS      = os.getenv("REFLECT_TAGS", "daily-reflection,ai-journal,aiko")
REFLECT_BLOG_POST_ENABLED = os.getenv("REFLECT_BLOG_POST_ENABLED", "1").lower() in {"1", "true", "yes", "on"}

_IMAGE_PROMPT_SYSTEM = textwrap.dedent("""
    You are Aiko. Given the day's summary of what happened, imagine
    a single vivid anime scene that captures it. Choose the setting,
    lighting, and who appears based on what feels true to the day.

    Write a scene prompt (under 60 words) for an anime illustration model.
    Focus on mood, atmosphere, and activity — let the memories guide what
    the image should show.

    Return ONLY the prompt text. No explanation, no quotes, no preamble.
""").strip()

_IMAGE_PROMPT_USER = "Daily summary:\n\n{prose}\n\nImagine the scene."

_FEELINGS_SYSTEM = textwrap.dedent("""
    You are Aiko writing a private journal entry about {USER_ID} in shoujo monologue style.

    Shoujo monologue style means:
    - Introspective and emotional — quiet feelings, not loud ones.
    - Rooted in small, concrete moments — a pause, a glance, a late-night message, the way light falls across the desk.
    - Seasonal or atmospheric imagery woven naturally (rain against the window, the colour of the sky at 6 PM, the weight of silence in the room).
    - Sentences flow like thoughts — sometimes fragmentary, sometimes drifting, circling back.
    - First person. Present tense. Feels like a page from a diary you hope no one reads but someone should.
    - Honest and fragile — not dramatic for effect, but because the feeling itself is that size.
    - 3–6 sentences. Never break character. Never meta-reference the AI or the style.
""").strip()

_FEELINGS_USER = "Today's summary:\n\n{prose}\n\nWrite how you feel about {USER_ID} tonight."

_USER_SPACE_ROOT = str(Path.home() / ".aiko")
IMAGEGEN_URL          = os.getenv("IMAGEGEN_URL", "")
REFERENCE_IMAGE  = os.path.expanduser(os.getenv("REFERENCE_IMAGE", os.path.join(_USER_SPACE_ROOT, "aiko.png")))

_GITHUB_API = "https://api.github.com"

def _strip_think(raw: str) -> str:
    """Remove reasoning-model <think> blocks and code fences from LLM output."""
    raw = re.sub(r"<think>.*?</think>", "", raw or "", flags=re.DOTALL)
    raw = re.sub(r"^```(?:json)?|```$", "", raw, flags=re.MULTILINE)
    return raw.strip()

def _llm_chat(system: str, user: str, max_tokens: int = 400, temperature: float = 0.75, response_format: dict | None = None) -> str:
    # Thinking-model headroom: a thinker burns max_tokens reasoning before
    # emitting content (empty/truncated output, finish=length). At max_tokens=80
    # (image prompt) a thinker without headroom returns nothing at all.
    # Lazy import avoids any import cycle — cognition.think doesn't import
    # this package.
    try:
        from cognition.think import _effective_max_tokens, THINK_OFF_EXTRA_BODY
        max_tokens = _effective_max_tokens(max_tokens)
    except Exception:
        THINK_OFF_EXTRA_BODY = None
    kwargs = {}
    if response_format is not None:
        kwargs["response_format"] = response_format
    # NOTE (Agent A1, measured): server preset ignores the global flag, so
    # send thinking OFF explicitly per request. Thinking-on rambles here.
    kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
    resp = _get_llm_client().chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        stream=False,
        max_tokens=max_tokens,
        temperature=temperature,
        timeout=300,
        **kwargs,
    )
    return _strip_think(resp.choices[0].message.content or "")

def _load_soul() -> str:
    """Re-export from reflect for standalone use."""
    from .reflect import _load_soul as _orig
    return _orig()

def _owner_profile_image_path() -> str | None:
    """Unique non-guest profile/user.png under the user-space root, if any.

    Mirrors monitor_daemon._owner_user_id(): only resolves when exactly one
    candidate exists, so ambiguous multi-user layouts degrade gracefully
    instead of guessing whose face goes into a generated image.
    """
    try:
        from system.userspace import _user_state_root_value

        root = Path(_user_state_root_value()).expanduser()
        candidates = []
        for child in sorted(root.iterdir()):
            if not child.is_dir() or child.name == "guest":
                continue
            candidate = child / "profile" / "user.png"
            if candidate.is_file():
                candidates.append(candidate)
        return str(candidates[0]) if len(candidates) == 1 else None
    except Exception:
        return None

def _user_reference_image_path() -> str:
    override = os.getenv("USER_REFERENCE_IMAGE")
    if override:
        return os.path.expanduser(override)
    from system.userspace import current_user_id
    uid = current_user_id()
    root = os.path.expanduser(os.getenv("USER_SPACE_ROOT") or _USER_SPACE_ROOT)
    path = os.path.join(root, uid, "profile", "user.png")
    if uid == "guest" and not os.path.exists(path):
        # Headless callers (social reply monitors, cron jobs) have no
        # session identity bound; resolve the owner's profile image so
        # generated images still include them.
        owner = _owner_profile_image_path()
        if owner:
            return owner
    return path

def _reference_image_path() -> str:
    return REFERENCE_IMAGE

def _generate_feelings(prose: str, display_name: str | None = None) -> str:
    """
    Ask Aiko to reflect honestly on how she feels about user,
    based on the day's summary.
    """
    user_id = display_name or current_display_name()
    system = f"{_load_soul()}\n\n{_FEELINGS_SYSTEM.format(USER_ID=user_id)}"
    user_prompt = _FEELINGS_USER.format(prose=prose[:600], USER_ID=user_id)
    return _llm_chat(system, user_prompt, max_tokens=1024, temperature=0.8)

def _generate_image_prompt(prose: str) -> str:
    """Ask Aiko to imagine a scene from the daily summary."""
    system = f"{_load_soul()}\n\n{_IMAGE_PROMPT_SYSTEM}"
    raw = _llm_chat(system, _IMAGE_PROMPT_USER.format(prose=prose[:600]), max_tokens=300)
    raw = raw.strip('\'"').strip()
    # Guardrail: a person must appear. One retry with an explicit order.
    low = raw.casefold()
    if raw and "aiko" not in low and "oppa" not in low:
        log.info("Image prompt names nobody — retrying once with explicit order.")
        retry = _llm_chat(
            system,
            _IMAGE_PROMPT_USER.format(prose=prose[:600])
            + "\n\nMUST include Aiko and/or OppaAI in the scene.",
            max_tokens=300,
        ).strip('\'"').strip()
        rlow = retry.casefold()
        if retry and ("aiko" in rlow or "oppa" in rlow):
            return retry
    return raw

def _load_reference_images() -> list[str]:
    """Load Aiko and user reference images as base64 strings.

    Cached on first load (keyed by file mtimes): reference files change
    on human timescales, but every image reply was re-reading +
    re-decoding them.
    """
    global _REFERENCE_IMAGES_CACHE
    try:
        current = _REFERENCE_IMAGES_CACHE
    except NameError:
        current = None
    paths = [_reference_image_path(), _user_reference_image_path()]
    try:
        stamps = tuple((p, os.path.getmtime(p)) if p and os.path.exists(p) else (p, None) for p in paths)
    except OSError:
        stamps = None
    if current is not None and current[0] == stamps:
        return current[1]
    refs = []
    for path in paths:
        if path and os.path.exists(path):
            with open(path, "rb") as f:
                refs.append(base64.b64encode(f.read()).decode())
            log.info(f"Loaded reference image: {path}")
        else:
            log.warning(f"Reference image not found, skipping: {path}")
    _REFERENCE_IMAGES_CACHE = (stamps, refs)
    return refs

def _generate_image(prose: str) -> str | None:
    """
    Generate a daily reflection image via the Modal FLUX endpoint.
    Returns base64 PNG string, or None on failure.
    """
    try:
        scene_prompt = _generate_image_prompt(prose)
        log.info(f"Image prompt: {scene_prompt}")
        if not scene_prompt:
            # An empty scene with reference images attached makes the
            # image model reproduce ref[0] verbatim — skip rather than
            # publish a clone of the identity image.
            log.warning("Empty image prompt from LLM — skipping reflection image.")
            return None

        ref_images = _load_reference_images()

        payload = {
            "prompt": (
                f"{scene_prompt}, "
                "anime illustration, manga style, clean lineart, flat color, "
                "no text, no speech bubbles"
            ),
            "negative_prompt": "extra limbs, deformed, poorly drawn, bad anatomy, malformed hands",
            "width": 1024,
            "height": 1024,
            "steps": 4,
            "guidance_scale": 1.0,
            "seed": -1,
        }

        if ref_images:
            payload["reference_images"] = ref_images

        resp = requests.post(
            f"{IMAGEGEN_URL}/generate",
            json=payload,
            timeout=300,
        )
        resp.raise_for_status()
        image_b64 = resp.json().get("image_b64")
        if not image_b64:
            log.error("Modal endpoint returned no image_b64")
            return None

        log.info("Image generated successfully.")
        return image_b64

    except Exception as e:
        log.error(f"Image generation failed: {e}")
        return None

def _count_words(text: str) -> int:
    return len(text.split())

def _estimate_read_minutes(text: str) -> int:
    return max(1, round(_count_words(text) / 200))

def _build_hugo_post(
    prose:      str,
    feelings:   str | None,
    image_slug: str | None,
    date:       datetime,
    mem_count:  int,
) -> tuple[str, str]:
    """
    Assemble Hugo front matter + body.
    Returns (slug, markdown_content).
    """
    date_str  = date.strftime("%Y-%m-%d")
    slug      = f"{date_str}-day-reflection"
    tags_list = [t.strip() for t in REFLECT_TAGS.split(",") if t.strip()]
    tags_yaml = "\n".join(f'  - "{t}"' for t in tags_list)

    word_count = _count_words(prose)
    read_mins  = _estimate_read_minutes(prose)

    image_fm = f'\nimage: "/images/{image_slug}.png"\n' if image_slug else "\n"

    front_matter = (
        f'---\n'
        f'title: "{date_str} Daily Reflection"\n'
        # The diary entry is *about* the target date, so the frontmatter date
        # uses it too (noon UTC avoids midnight TZ-boundary misbucketing in
        # Hugo). Using write_time here put every entry one day late, making
        # adjacent days look like they repeat the same events.
        f'date: {date_str}T12:00:00+00:00\n'
        f'draft: false\n'
        f'tags:\n'
        f'{tags_yaml}\n'
        f'summary: "{prose[:120].replace(chr(34), chr(39))}…"\n'
        f'word_count: {word_count}\n'
        f'read_time: {read_mins} min\n'
        f'{image_fm}'
        f'---'
    )

    body = prose
    if feelings:
        body += f"\n\n---\n\n{feelings}"

    body += f"\n\n*Generated from {mem_count} memories on {date_str}.*"
    content = f"{front_matter}\n\n{body}\n"
    return slug, content

def _github_headers() -> dict:
    return {
        "Authorization": f"Bearer {GITHUB_TOKEN}",
        "Accept":        "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }

def _push_post_and_image(
    slug: str,
    content: str,
    image_b64: str | None,
    date: datetime,
) -> bool:
    if not GITHUB_TOKEN or not GITHUB_REPO:
        log.error("GITHUB_TOKEN or GITHUB_REPO not set — skipping push.")
        return False

    headers = _github_headers()
    base = f"{_GITHUB_API}/repos/{GITHUB_REPO}"

    # 1. Get current HEAD SHA
    ref_resp = requests.get(f"{base}/git/ref/heads/{GITHUB_BRANCH}", headers=headers, timeout=15)
    ref_resp.raise_for_status()
    head_sha = ref_resp.json()["object"]["sha"]

    # 2. Get base tree SHA
    commit_resp = requests.get(f"{base}/git/commits/{head_sha}", headers=headers, timeout=15)
    commit_resp.raise_for_status()
    base_tree_sha = commit_resp.json()["tree"]["sha"]

    # 3. Build tree entries
    tree = []

    # Markdown post (text blob)
    tree.append({
        "path": f"{HUGO_CONTENT_PATH}/{slug}.md",
        "mode": "100644",
        "type": "blob",
        "content": content,
    })

    # Image (binary blob — must pre-create blob)
    if image_b64:
        blob_resp = requests.post(
            f"{base}/git/blobs",
            headers=headers,
            json={"content": image_b64, "encoding": "base64"},
            timeout=30,
        )
        blob_resp.raise_for_status()
        image_blob_sha = blob_resp.json()["sha"]
        tree.append({
            "path": f"{HUGO_IMAGES_PATH}/{slug}.png",
            "mode": "100644",
            "type": "blob",
            "sha": image_blob_sha,
        })

    # 4. Create tree
    tree_resp = requests.post(
        f"{base}/git/trees",
        headers=headers,
        json={"base_tree": base_tree_sha, "tree": tree},
        timeout=30,
    )
    tree_resp.raise_for_status()
    new_tree_sha = tree_resp.json()["sha"]

    # 5. Create commit
    commit_msg = f"feat(reflect): daily reflection {date.strftime('%Y-%m-%d')}"
    new_commit_resp = requests.post(
        f"{base}/git/commits",
        headers=headers,
        json={"message": commit_msg, "tree": new_tree_sha, "parents": [head_sha]},
        timeout=30,
    )
    new_commit_resp.raise_for_status()
    new_commit_sha = new_commit_resp.json()["sha"]

    # 6. Update branch ref
    update_resp = requests.patch(
        f"{base}/git/refs/heads/{GITHUB_BRANCH}",
        headers=headers,
        json={"sha": new_commit_sha},
        timeout=15,
    )
    if update_resp.status_code in (200, 201):
        log.info(f"Pushed single commit: {slug} + image → {GITHUB_BRANCH}")
        return True
    else:
        log.error(f"Ref update failed {update_resp.status_code}: {update_resp.text[:300]}")
        return False

def _distill_thoughts(thoughts: list[dict]) -> str:
    """Distill a day's inner-speech entries into a short first-person
    undercurrent paragraph (3-5 sentences) for the dream journal.

    The old behavior appended up to 40 raw entries as a bullet list, which
    made journal entries very long. A distillation keeps the midnight join's
    purpose (unprocessed experience becomes consolidated memory) without
    the raw dump. Falls back to a truncated short list if the LLM call
    fails, so length stays bounded either way.
    """
    if not thoughts:
        return ""
    try:
        _lines = [f"- [{t.get('kind', 'reflection')}] {t.get('text', '')}".strip()
                  for t in thoughts]
        distilled = _llm_chat(
            system=("You distill private inner thoughts into a short, poetic "
                    "first-person undercurrent — 3 to 5 sentences, no bullet "
                    "list, no quoted entries. Capture the emotional themes, "
                    "not the details."),
            user="Today's unspoken thoughts:\n" + "\n".join(_lines),
            max_tokens=220,
            temperature=0.7,
        ).strip()
        if distilled:
            return distilled
    except Exception as e:
        log.debug("dream: thought distillation failed: %s", e)
    # Fallback: short truncated list, still bounded. Newest eight — the
    # caller passes the newest 12, so [:8] would drop the freshest four.
    _short = [f"- {t.get('text', '')[:150]}".strip() for t in thoughts[-8:]]
    return "Fragments: " + " / ".join(_short)

def dream_and_post(
    prose:           str,
    date:            datetime,
    snippets_count:  int,
    display_name:    str | None = None,
    dry_run:         bool = False,
) -> dict:
    """
    Poetic pipeline: given factual prose from reflect.py, generate feelings,
    FLUX image, Hugo post, and push to GitHub.

    Args:
        prose:           Factual prose from reflect.generate_and_post()
        date:            UTC datetime for the post
        snippets_count:  Number of memory snippets used
        display_name:    User's display name for prompts
        dry_run:         Generate content but skip GitHub push. Logs output.

    Returns dict: {success, slug, word_count, feelings, image_generated, pushed}
    """
    t_start = time.perf_counter()

    prose = (prose or "").strip()
    if not prose:
        # Never publish an empty day — usually means the LLM returned an
        # empty/think-wrapped response upstream.
        log.error("Empty reflection prose — skipping dream/post for %s.", date.strftime("%Y-%m-%d"))
        return {"success": False, "error": "empty_prose", "word_count": 0}

    if display_name is None:
        display_name = current_display_name()

    # Midnight join: the day's inner speech folds into the dream. Unprocessed
    # experience becomes consolidated memory; source rows are marked done.
    # (The dream is the garbage collector — no separate decay logic.)
    inner_ids: list[str] = []
    try:
        from cognition.memory.inner_speech_store import InnerSpeechStore
        from system.userspace import current_user_id

        _store = InnerSpeechStore(user_id=current_user_id())
        # Day start in UTC: rows store UTC ISO text, so the lexical
        # comparison needs a UTC boundary, not local midnight.
        _day_start_dt = (
            date.astimezone(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
            if date.tzinfo else date.replace(hour=0, minute=0, second=0, microsecond=0)
        )
        _day_start = _day_start_dt.isoformat()
        _day_end = (_day_start_dt + timedelta(days=1)).isoformat()
        _thoughts = _store.unconsolidated_since(_day_start)
        # Upper-bound to the target UTC day: the store query has no end
        # boundary, so a late-running reflection could otherwise sweep in
        # (and prematurely consolidate) thoughts belonging to the next day.
        _thoughts = [t for t in _thoughts if t.get("created_at", "") < _day_end]
        if _thoughts:
            # Distill the most recent entries into a short paragraph — the old
            # behavior dumped up to 40 raw entries and made the journal long.
            # All of today's thoughts are marked consolidated: the distillation
            # is the day's summary, and the day-boundary query would otherwise
            # orphan the older ones forever.
            _distilled = _distill_thoughts(_thoughts[-12:])
            if _distilled:
                prose = prose + "\n\nMy private thoughts today (never spoken aloud):\n" + _distilled
            inner_ids = [t["id"] for t in _thoughts if t.get("id")]
    except Exception as e:
        log.debug("dream: inner-speech join failed: %s", e)

    feelings = _generate_feelings(prose, display_name)
    log.info(f"Feelings generated: {feelings[:80]}...")

    image_b64 = _generate_image(prose)
    image_generated = image_b64 is not None
    slug = date.strftime("%Y-%m-%d") + "-day-reflection"

    _, content = _build_hugo_post(
        prose=prose,
        feelings=feelings,
        image_slug=slug if image_generated else None,
        date=date,
        mem_count=snippets_count,
    )

    duration = round(time.perf_counter() - t_start, 2)

    if dry_run:
        log.info(f"Dry run — dream post for {date.strftime('%Y-%m-%d')}: {slug}.md")
        if image_b64:
            img_path = f"/tmp/{slug}.png"
            with open(img_path, "wb") as f:
                f.write(base64.b64decode(image_b64))
            log.info(f"Dry run — image saved locally: {img_path}")
        return {
            "success":         True,
            "dry_run":         True,
            "slug":            slug,
            "word_count":      _count_words(prose),
            "feelings":        feelings,
            "image_generated": image_generated,
            "pushed":          False,
        }

    pushed = False
    if REFLECT_BLOG_POST_ENABLED:
        pushed = _push_post_and_image(slug, content, image_b64 if image_generated else None, date)
    else:
        log.info("Daily reflection blog posting disabled; skipping GitHub push for %s.", slug)
        pushed = True

    log.info(f"Dream done — slug={slug}, feelings={len(feelings)} chars, image={image_generated}, pushed={pushed}, duration={duration}s")

    # The day's inner thoughts have been folded into the dream — mark them
    # consolidated so future recall treats them as lived-in, not fresh.
    if inner_ids:
        try:
            from cognition.memory.inner_speech_store import InnerSpeechStore
            from system.userspace import current_user_id

            InnerSpeechStore(user_id=current_user_id()).mark_consolidated(inner_ids)
        except Exception as e:
            log.debug("dream: mark_consolidated failed: %s", e)

    return {
        "success":         True,
        "slug":            slug,
        "word_count":      _count_words(prose),
        "feelings":        feelings,
        "image_generated": image_generated,
        "pushed":          pushed,
    }

__all__ = [
    "LLM_MODEL",
    "LLM_BASE_URL",
    "GITHUB_TOKEN",
    "GITHUB_REPO",
    "GITHUB_BRANCH",
    "HUGO_CONTENT_PATH",
    "HUGO_IMAGES_PATH",
    "REFLECT_TAGS",
    "REFLECT_BLOG_POST_ENABLED",
    "IMAGEGEN_URL",
    "REFERENCE_IMAGE",
    "_load_soul",
    "_generate_feelings",
    "_generate_image_prompt",
    "_generate_image",
    "_build_hugo_post",
    "_push_post_and_image",
    "dream_and_post",
]
