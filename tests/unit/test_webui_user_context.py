from __future__ import annotations

import asyncio
import base64
from concurrent.futures import ThreadPoolExecutor
import os
import queue
import threading

import pytest

from interface.webui.webui import AikoWeb, _note_visual_frame_bounded, _validate_image_data_uri, _vision_base_url
from interface.webui.studio.session_binding import _relative_path
from system.prepare import run_post_auth
from system.userspace import current_display_name, current_user_id, reset_current_display_name, reset_current_user_id


def test_image_motion_runs_in_thread_with_source(monkeypatch):
    from cognition.flysense import pathways
    from interface.webui import webui

    web = AikoWeb.__new__(AikoWeb)
    web._broadcast = lambda *_args, **_kwargs: None
    web._infer_image = lambda *_args: "description"
    monkeypatch.setattr(webui, "observe_fly_runtime", lambda *_args, **_kwargs: None)
    seen = []
    caller_thread = threading.get_ident()

    def note_frame(uid, image, source):
        seen.append((uid, image, source, threading.get_ident()))

    monkeypatch.setattr(pathways, "note_visual_frame", note_frame)

    asyncio.run(web._handle_image_input("image", "question", "alice", source="screen"))

    assert len(seen) == 1
    assert seen[0][:3] == ("alice", "image", "screen")
    assert seen[0][3] != caller_thread


class DummyMemorize:
    def __init__(self) -> None:
        self.switched: list[str] = []
        self.display_names: list[str] = []

    def switch_user(self, uid: str) -> None:
        self.switched.append(uid)

    def set_display_name(self, name: str) -> None:
        self.display_names.append(name)


@pytest.fixture(autouse=True)
def clean_user_env(monkeypatch):
    """Establish baseline user context before each test and reset after.

    This prevents context leakage between tests by ensuring both the environment
    variables and context tokens are reset to known values.
    """
    from system.userspace import set_current_user_id, set_current_display_name

    monkeypatch.delenv("AIKO_USER_ID", raising=False)
    monkeypatch.delenv("CURRENT_DISPLAY_NAME", raising=False)

    # Establish baseline context values
    user_token = set_current_user_id("guest")
    display_token = set_current_display_name(None)

    yield

    # Reset context after test
    reset_current_user_id(user_token)
    reset_current_display_name(display_token)


@pytest.mark.asyncio
async def test_bind_user_context_is_request_local(monkeypatch):
    web = AikoWeb.__new__(AikoWeb)
    web._memorize = DummyMemorize()

    user_token, display_token, display_name = await web._bind_user_context(
        "alice", {"username": "Alice"}
    )
    try:
        assert current_user_id() == "alice"
        assert current_display_name() == "Alice"
        assert display_name == "Alice"
        assert os.getenv("AIKO_USER_ID") is None
        assert web._memorize.switched == []
        assert web._memorize.display_names == []
        assert not hasattr(web, "_current_user_id")
        assert not hasattr(web, "_current_display_name")
    finally:
        reset_current_display_name(display_token)
        reset_current_user_id(user_token)

    assert current_user_id() == "guest"


def test_get_input_uses_queued_identity_not_shared_state(monkeypatch):
    web = AikoWeb.__new__(AikoWeb)
    web._input_q = queue.Queue()
    web._input_q.put(("hello", "bob", "Bobby"))
    web._broadcast = lambda payload: None
    web._push_vitals = lambda: None

    assert web.get_input() == "hello"
    assert current_user_id() == "bob"
    assert current_display_name() == "Bobby"


@pytest.mark.parametrize(("active_until", "expected"), [
    (float("inf"), []),
    (0.0, ["leanIn"]),
])
def test_text_only_get_input_respects_reactive_gesture_state(active_until, expected):
    web = AikoWeb.__new__(AikoWeb)
    web._no_voice = True
    web._gesture_active_until = active_until
    web._input_q = queue.Queue()
    web._input_q.put(("hello", "bob", "Bobby"))
    web._broadcast = lambda payload, **_kwargs: None
    web._push_vitals = lambda: None
    played = []
    web.play_gesture = played.append

    assert web.get_input() == "hello"
    assert played == expected


def test_text_only_get_input_plays_lean_in_once_after_reactive_gesture_ends():
    class EmptyThenInputQueue:
        def __init__(self):
            self._items = [queue.Empty, queue.Empty, ("hello", "bob", "Bobby")]

        def get(self, timeout):
            item = self._items.pop(0)
            if item is queue.Empty:
                raise queue.Empty
            return item

    web = AikoWeb.__new__(AikoWeb)
    web._no_voice = True
    web._input_q = EmptyThenInputQueue()
    web._broadcast = lambda payload, **_kwargs: None
    web._push_vitals = lambda: None
    checks = 0

    def gesture_is_active():
        nonlocal checks
        checks += 1
        return checks == 1

    web._gesture_is_active = gesture_is_active
    played = []
    web.play_gesture = played.append

    assert web.get_input() == "hello"
    assert played == ["leanIn"]


def _image_header(mime: str, width: int, height: int) -> bytes:
    if mime == "image/jpeg":
        return b"\xff\xd8\xff\xc0\x00\x0b\x08" + height.to_bytes(2, "big") + width.to_bytes(2, "big") + b"\x01\x01\x11\x00"
    if mime == "image/png":
        return b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR" + width.to_bytes(4, "big") + height.to_bytes(4, "big")
    return b"RIFF\x16\x00\x00\x00WEBPVP8X\x0a\x00\x00\x00\x00\x00\x00\x00" + (width - 1).to_bytes(3, "little") + (height - 1).to_bytes(3, "little")


def test_camera_image_validation_accepts_small_jpeg_data_uri():
    image = f"data:image/jpeg;base64,{base64.b64encode(_image_header('image/jpeg', 1, 1)).decode()}"

    assert _validate_image_data_uri(image) == image


@pytest.mark.parametrize("mime", ["image/png", "image/webp"])
def test_camera_image_validation_accepts_matching_image_signatures(mime):
    image = f"data:{mime};base64,{base64.b64encode(_image_header(mime, 1, 1)).decode()}"

    assert _validate_image_data_uri(image) == image


@pytest.mark.parametrize(("mime", "raw"), [
    ("image/jpeg", b"\x89PNG\r\n\x1a\n"),
    ("image/png", b"RIFF\x04\x00\x00\x00WEBP"),
    ("image/webp", b"\xff\xd8\xff\xd9"),
])
def test_camera_image_validation_rejects_mismatched_image_signatures(mime, raw):
    encoded = base64.b64encode(raw).decode()

    assert _validate_image_data_uri(f"data:{mime};base64,{encoded}") is None


@pytest.mark.parametrize("image", [
    "https://example.test/image.jpg",
    "data:image/gif;base64,R0lGODlh",
    "data:image/jpeg;base64,not valid base64!",
])
def test_camera_image_validation_rejects_unsafe_or_invalid_payloads(image):
    assert _validate_image_data_uri(image) is None


@pytest.mark.parametrize("mime", ["image/jpeg", "image/png", "image/webp"])
def test_camera_image_validation_rejects_oversized_pixel_count(mime):
    image = f"data:{mime};base64,{base64.b64encode(_image_header(mime, 4096, 2049)).decode()}"

    assert _validate_image_data_uri(image) is None


def test_visual_decode_budget_is_shared_across_connections(monkeypatch):
    from cognition.flysense import pathways

    started = threading.Event()
    release = threading.Event()
    seen = []

    def note_frame(uid, image, source):
        seen.append(uid)
        if uid == "alice":
            started.set()
            release.wait(timeout=5)

    monkeypatch.setattr(pathways, "note_visual_frame", note_frame)
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(_note_visual_frame_bounded, "alice", "image", "camera")
        try:
            assert started.wait(timeout=2)
            pool.submit(_note_visual_frame_bounded, "bob", "image", "screen").result(timeout=2)
        finally:
            release.set()
        first.result(timeout=2)

    _note_visual_frame_bounded("carol", "image", "camera")
    assert seen == ["alice", "carol"]


@pytest.mark.parametrize(("webui_url", "vision_url", "llm_url", "expected"), [
    ("https://webui.test/v1", "https://vision.test/v1", "https://llm.test/v1", "https://webui.test/v1"),
    ("", "https://vision.test/v1", "https://llm.test/v1", "https://vision.test/v1"),
    (" ", "", "https://llm.test/v1", "https://llm.test/v1"),
    ("", "", "", "http://localhost:8080/v1"),
])
def test_vision_base_url_uses_first_non_blank_value(monkeypatch, webui_url, vision_url, llm_url, expected):
    monkeypatch.setenv("WEBUI_VISION_BASE_URL", webui_url)
    monkeypatch.setenv("VISION_BASE_URL", vision_url)
    monkeypatch.setenv("LLM_BASE_URL", llm_url)

    assert _vision_base_url() == expected


def test_studio_api_path_is_relative_to_its_mount():
    """Mounted studio middleware must recognize its /api routes."""
    assert _relative_path({
        "path": "/studio/memory/ltm/api/graph",
        "root_path": "/studio/memory/ltm",
    }) == "/api/graph"
    assert _relative_path({"path": "/api/graph", "root_path": ""}) == "/api/graph"


def test_post_auth_binds_memory_to_logged_in_user():
    class Memorize:
        def __init__(self):
            self.switched_to = []
            self.cleanup_user = None

        def switch_user(self, uid):
            self.switched_to.append(uid)

        def cleanup(self):
            self.cleanup_user = current_user_id()

        def persona_context(self):
            return ""

        def get_all(self):
            return []

    memorize = Memorize()
    run_post_auth("github_alice", memorize=memorize)

    assert memorize.switched_to == ["github_alice"]
    assert memorize.cleanup_user == "github_alice"
    assert current_user_id() == "guest"


def test_concurrent_user_active_calls_run_post_auth_once(monkeypatch):
    """Two concurrent _on_user_active calls for the same user should only run post-auth once."""
    class Memorize:
        def __init__(self):
            self.switch_calls = []
            self.cleanup_calls = 0

        def switch_user(self, uid):
            self.switch_calls.append(uid)

        def cleanup(self):
            self.cleanup_calls += 1

        def persona_context(self):
            return ""

        def get_all(self):
            return []

    memorize = Memorize()
    from system.wakeup import BootResult
    boot_result = BootResult(memorize=memorize, think=None, speak=None, perceive=None, observe=None, navigate=None)
    web = AikoWeb(boot_result=boot_result, defer_servers=True)

    barrier = threading.Barrier(2)
    def active_with_barrier(uid):
        barrier.wait()
        web._on_user_active(uid)

    t1 = threading.Thread(target=active_with_barrier, args=("github_alice",))
    t2 = threading.Thread(target=active_with_barrier, args=("github_alice",))
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    assert memorize.switch_calls == ["github_alice"]
    assert memorize.cleanup_calls == 1


def test_concurrent_user_active_different_users(monkeypatch):
    """Two concurrent _on_user_active calls for different users should both run post-auth."""
    class Memorize:
        def __init__(self):
            self.switch_calls = []
            self.cleanup_calls = 0

        def switch_user(self, uid):
            self.switch_calls.append(uid)

        def cleanup(self):
            self.cleanup_calls += 1

        def persona_context(self):
            return ""

        def get_all(self):
            return []

    memorize = Memorize()
    from system.wakeup import BootResult
    boot_result = BootResult(memorize=memorize, think=None, speak=None, perceive=None, observe=None, navigate=None)
    web = AikoWeb(boot_result=boot_result, defer_servers=True)

    barrier = threading.Barrier(2)
    def active_with_barrier(uid):
        barrier.wait()
        web._on_user_active(uid)

    t1 = threading.Thread(target=active_with_barrier, args=("github_alice",))
    t2 = threading.Thread(target=active_with_barrier, args=("github_bob",))
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    assert set(memorize.switch_calls) == {"github_alice", "github_bob"}
    assert memorize.cleanup_calls == 2
