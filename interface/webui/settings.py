"""Authenticated settings editor for YAML and encrypted environment values."""
from __future__ import annotations
import io, json, os, re, subprocess, tempfile
from pathlib import Path
from typing import Any
from fastapi import APIRouter, Depends, HTTPException, Request
from system.config import _decrypt_env, _load_yaml_mapping

async def _require_settings_access(request: Request) -> dict:
    from interface.webui.auth import require_session
    session = await require_session(request)
    if not session.get("accepted_terms"):
        raise HTTPException(status_code=403, detail="Guidelines not yet accepted")
    allowed = {item.strip() for item in os.getenv("SETTINGS_ALLOWED_USERS", "OppaAI").split(",") if item.strip()}
    if session.get("user_id") not in allowed:
        raise HTTPException(status_code=403, detail="Settings are restricted to the configured owner")
    return session

router = APIRouter(prefix="/api/settings", tags=["settings"], dependencies=[Depends(_require_settings_access)])
_NAME = re.compile(r"^[A-Za-z0-9_.-]+$")

def _root() -> Path: return Path(__file__).resolve().parents[2]
def _state_paths() -> tuple[Path, Path]:
    root = Path(os.environ.get("USER_SPACE_ROOT", str(Path.home() / ".aiko"))).expanduser()
    def resolve(name: str, default: str) -> Path:
        value = Path(os.environ.get(name, default)).expanduser()
        return value if value.is_absolute() else root / value
    return resolve("AGE_KEY", "age-key.txt"), resolve("ENV_AGE_PATH", ".env.age")
def _yaml_names() -> list[str]:
    data = _load_yaml_mapping(_root() / "config" / "index.yaml")
    return [str(n) for n in data.get("configs", []) if _NAME.fullmatch(str(n)) and str(n).endswith((".yaml", ".yml"))]
def _check_name(name: str) -> Path:
    if name == ".env.age": return _state_paths()[1]
    if name not in _yaml_names(): raise HTTPException(status_code=404, detail="Unknown settings file")
    return (_root() / "config" / name).resolve()

def _yaml_metadata(path: Path) -> dict[str, dict[str, Any]]:
    """Collect friendly groups, descriptions, choices, and ranges from comments."""
    metadata: dict[str, dict[str, Any]] = {}
    group = "General"
    pending: list[str] = []
    stack: list[tuple[int, str]] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        stripped = raw.strip()
        if stripped.startswith("#"):
            comment = stripped[1:].strip()
            if comment.startswith("--") or (comment and comment.endswith("--")):
                group = comment.strip("- ").strip()
                pending = []
            elif comment:
                pending.append(comment)
            continue
        match = re.match(r"^(\s*)([A-Za-z_][A-Za-z0-9_-]*):\s*(.*)$", raw)
        if not match:
            continue
        indent, key, remainder = len(match.group(1)), match.group(2), match.group(3)
        while stack and indent <= stack[-1][0]: stack.pop()
        setting_path = ".".join(item[1] for item in stack + [(indent, key)])
        inline = ""
        if "#" in remainder: remainder, inline = remainder.split("#", 1)
        description = " ".join(pending + ([inline.strip()] if inline.strip() else []))
        options: list[str] = []
        choice_match = re.search(r"(?:options?|choices?)\s*:?\s*([^.;]+)", description, re.I)
        if choice_match:
            options = [x.strip(" `\"") for x in re.split(r"\s*[|/,]\s*", choice_match.group(1)) if x.strip()]
        if not options:
            pipe_match = re.search(r"([A-Za-z][\w-]*(?:\s*\|\s*[A-Za-z][\w-]*)+)", description)
            if pipe_match:
                options = [x.strip() for x in pipe_match.group(1).split("|")]
        range_match = re.search(r"(?:range|clamp(?:ed)?)\s*[: ]\s*(-?\d+(?:\.\d+)?)\s*(?:-|\.\.|to)\s*(-?\d+(?:\.\d+)?)", description, re.I)
        metadata[setting_path] = {"group": group, "description": description, "options": options, "range": [float(range_match.group(1)), float(range_match.group(2))] if range_match else None}
        if remainder.strip() == "": stack.append((indent, key))
        pending = []
    return metadata
def _friendly_group(path: str, section: str = "") -> str:
    """Map detailed YAML sections into a small, stable settings taxonomy."""
    name = path.upper()
    rules = (
        ("Vision", ("VISION", "CAMERA", "IMAGE")),
        ("Voice", ("ASR", "LISTEN", "VAD", "TTS", "SPEAK", "AUDIO", "BARGE")),
        ("Interface", ("WEBUI", "HTTP_", "WS_", "SSL", "RENDER", "STREAM", "LATENCY", "FLY_", "PYTHONPATH")),
        ("Memory", ("MEMORY", "KNOWLEDGE", "RECALL", "EPISOD", "ENTITY", "VECTOR", "LORE")),
        ("Social", ("SOCIAL", "THREADS", "DISCORD", "TELEGRAM", "MASTODON", "BLUESKY", "PIXELFED", "PROTON")),
        ("Games", ("SHOGI", "YANEURAOU", "SELFPLAY", "GO_", "KOI_")),
        ("Conscience", ("CCC", "CONSCIENCE", "CANON", "MORAL")),
        ("Identity", ("AI_NAME", "USER_SPACE", "AGE_KEY", "ENV_AGE", "TIMEZONE", "OWNER")),
        ("Agentic", ("WIKI", "SKILL", "AGENT", "SEARCH", "JOB_", "ROUTER", "PLAYBOOK", "TOOL")),
    )
    for group, prefixes in rules:
        if any(token in name for token in prefixes):
            return group
    section_name = section.upper()
    if any(token in section_name for token in ("MODEL", "THINK", "REFLECT", "INNER SPEECH", "COGNITION")):
        return "Agentic"
    if any(token in section_name for token in ("SYSTEM", "SCHEDULE", "LOGGING", "BACKUP")):
        return "System"
    return "System"

def _setting_entries(data: dict[str, Any], prefix: str = "", metadata: dict[str, dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    entries = []
    for key, value in data.items():
        path = prefix + "." + str(key) if prefix else str(key)
        if isinstance(value, dict):
            entries.extend(_setting_entries(value, path, metadata))
            continue
        kind = "text"
        if isinstance(value, bool) or (isinstance(value, str) and value.lower() in ("0", "1", "true", "false", "yes", "no", "on", "off")): kind = "toggle"
        elif isinstance(value, (int, float)) or (isinstance(value, str) and re.fullmatch(r"-?\d+(\.\d+)?", value.strip())): kind = "scale" if 0 <= float(value) <= 1 else "number"
        info = (metadata or {}).get(path, {})
        numeric = isinstance(value, (int, float)) or (isinstance(value, str) and re.fullmatch(r"-?\d+(\.\d+)?", value.strip()))
        if info.get("options"): kind = "select"
        minimum, maximum = (info.get("range") or [None, None])
        if numeric and minimum is None:
            number = float(value); minimum = 0 if number >= 0 else number * 2; maximum = max(number * 2, 1) if number >= 0 else 0
        entries.append({"path": path, "key": str(key), "value": value, "kind": kind, "group": _friendly_group(path, info.get("group", "")), "description": info.get("description", ""), "options": info.get("options", []), "min": minimum, "max": maximum})
    return entries

def _set_path(data: dict[str, Any], path: str, value: Any) -> None:
    target = data
    parts = path.split(".")
    for part in parts[:-1]: target = target.setdefault(part, {})
    target[parts[-1]] = value

def _coerce(value: Any, original: Any) -> Any:
    if isinstance(original, bool): return bool(value)
    if isinstance(original, int) and not isinstance(original, bool): return int(float(value))
    if isinstance(original, float): return float(value)
    if isinstance(original, str) and original.lower() in ("0", "1", "true", "false", "yes", "no", "on", "off"): return "1" if str(value).lower() in ("1", "true", "yes", "on") else "0"
    if isinstance(original, list): return json.loads(value) if isinstance(value, str) else value
    return str(value) if isinstance(original, str) else value

def _read_env() -> dict[str, str]:
    identity, encrypted = _state_paths()
    return {} if not encrypted.exists() else _decrypt_env(encrypted, identity)
def _dotenv(values: dict[str, Any]) -> bytes:
    lines = ["# Managed by Aiko Settings Studio"]
    for key in sorted(values):
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key): raise HTTPException(status_code=400, detail=f"Invalid environment variable name: {key}")
        lines.append(f"{key}={json.dumps('' if values[key] is None else str(values[key]), ensure_ascii=False)}")
    return ("\n".join(lines) + "\n").encode()
def _encrypt_env(plaintext: bytes) -> None:
    identity, encrypted = _state_paths()
    if not identity.exists(): raise HTTPException(status_code=400, detail=f"Age identity not found: {identity}")
    try:
        recipient = subprocess.run(["age-keygen", "-y", str(identity)], capture_output=True, check=True, timeout=10).stdout.strip()
        result = subprocess.run(["age", "-r", recipient.decode("ascii"), "-a"], input=plaintext, capture_output=True, check=True, timeout=10)
    except FileNotFoundError as exc: raise HTTPException(status_code=500, detail="The age and age-keygen binaries are required") from exc
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, UnicodeError) as exc: raise HTTPException(status_code=400, detail=f"Could not encrypt .env.age: {exc}") from exc
    encrypted.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("wb", dir=encrypted.parent, prefix=".env.age.", delete=False) as handle:
        temporary = Path(handle.name); handle.write(result.stdout); handle.flush(); os.fsync(handle.fileno())
    os.replace(temporary, encrypted)

@router.get("/files")
async def list_settings_files() -> dict[str, Any]:
    identity, encrypted = _state_paths()
    return {"files": [{"name": n, "kind": "yaml"} for n in _yaml_names()] + [{"name": ".env.age", "kind": "env", "exists": encrypted.exists(), "key_exists": identity.exists()}]}
@router.get("/file/{name}")
async def read_settings_file(name: str) -> dict[str, Any]:
    path = _check_name(name)
    if name == ".env.age":
        try: return {"name": name, "kind": "env", "values": _read_env()}
        except (FileNotFoundError, RuntimeError) as exc: raise HTTPException(status_code=400, detail=str(exc)) from exc
    if not path.exists(): raise HTTPException(status_code=404, detail="Settings file is missing")
    return {"name": name, "kind": "yaml", "settings": _setting_entries(_load_yaml_mapping(path), metadata=_yaml_metadata(path))}
@router.put("/file/{name}")
async def write_settings_file(name: str, payload: dict[str, Any]) -> dict[str, str]:
    path = _check_name(name)
    if name == ".env.age":
        values = payload.get("values")
        if not isinstance(values, dict): raise HTTPException(status_code=400, detail="Environment settings must be an object")
        _encrypt_env(_dotenv(values)); return {"status": "saved"}
    settings = payload.get("settings")
    if not isinstance(settings, list): raise HTTPException(status_code=400, detail="Settings must be a list")
    try:
        data = _load_yaml_mapping(path)
        originals = {entry["path"]: entry.get("value") for entry in _setting_entries(data)}
        for entry in settings:
            if not isinstance(entry, dict) or entry.get("path") not in originals: raise ValueError("unknown setting path")
            _set_path(data, entry["path"], _coerce(entry.get("value"), originals[entry["path"]]))
        import yaml
        path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")
    except Exception as exc: raise HTTPException(status_code=400, detail=f"Could not save settings: {exc}") from exc
    return {"status": "saved"}
def _load_yaml_mapping_from_text(content: str) -> dict[str, Any]:
    try:
        import yaml
        return yaml.safe_load(io.StringIO(content)) or {}
    except ImportError:
        from system.config import _simple_yaml_load
        return _simple_yaml_load(content)
