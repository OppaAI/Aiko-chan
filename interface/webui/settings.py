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
    return {"name": name, "kind": "yaml", "content": path.read_text(encoding="utf-8")}
@router.put("/file/{name}")
async def write_settings_file(name: str, payload: dict[str, Any]) -> dict[str, str]:
    path = _check_name(name)
    if name == ".env.age":
        values = payload.get("values")
        if not isinstance(values, dict): raise HTTPException(status_code=400, detail="Environment settings must be an object")
        _encrypt_env(_dotenv(values)); return {"status": "saved"}
    content = payload.get("content")
    if not isinstance(content, str) or not content.strip(): raise HTTPException(status_code=400, detail="YAML content cannot be empty")
    try: parsed = _load_yaml_mapping_from_text(content)
    except Exception as exc: raise HTTPException(status_code=400, detail=f"Invalid YAML: {exc}") from exc
    if not isinstance(parsed, dict): raise HTTPException(status_code=400, detail="YAML must contain a mapping")
    path.write_text(content, encoding="utf-8"); return {"status": "saved"}
def _load_yaml_mapping_from_text(content: str) -> dict[str, Any]:
    try:
        import yaml
        return yaml.safe_load(io.StringIO(content)) or {}
    except ImportError:
        from system.config import _simple_yaml_load
        return _simple_yaml_load(content)
