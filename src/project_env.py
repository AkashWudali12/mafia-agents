from __future__ import annotations

import ast
import os
import re
from pathlib import Path


def load_project_env(path: str | Path | None = None, *, override: bool = False) -> Path | None:
    env_path = _resolve_env_path(path)
    if env_path is None or not env_path.is_file():
        return None
    try:
        values = _parse_env_file(env_path)
    except OSError:
        return None
    for key, value in values.items():
        if override or key not in os.environ:
            os.environ[key] = value
    return env_path.resolve()


def _resolve_env_path(path: str | Path | None) -> Path | None:
    if path is not None:
        return Path(path)
    current = Path.cwd().resolve()
    for directory in (current, *current.parents):
        candidate = directory / ".env"
        if candidate.is_file():
            return candidate
    return None


def _parse_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue
        key, raw_value = line.split("=", 1)
        key = key.strip()
        if not key:
            continue
        values[key] = _parse_env_value(raw_value.strip())
    return values


def _parse_env_value(raw_value: str) -> str:
    if not raw_value:
        return ""
    value = re.split(r"\s+#", raw_value, maxsplit=1)[0].strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        try:
            parsed = ast.literal_eval(value)
        except (SyntaxError, ValueError):
            return value[1:-1]
        return parsed if isinstance(parsed, str) else str(parsed)
    return value
