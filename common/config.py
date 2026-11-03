"""YAML config loading with environment-variable expansion.

Any string value in a config may reference environment variables such as
``${EKKA_ROOT}`` or ``${EKKA_ACTIVATION_DIR}``; they are expanded when the config
is loaded. An unset variable is an error, so a typo does not silently produce a
literal ``${...}`` path.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Union

import yaml

from .paths import setup_env

_VAR_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}|\$([A-Za-z_][A-Za-z0-9_]*)")


def expand_vars(obj: Any) -> Any:
    """Recursively expand ``$VAR`` / ``${VAR}`` in strings inside dicts and lists."""
    if isinstance(obj, str):
        def repl(m: "re.Match[str]") -> str:
            name = m.group(1) or m.group(2)
            if name not in os.environ:
                raise KeyError(f"Environment variable '{name}' is not set (referenced in config value {obj!r})")
            return os.environ[name]

        return _VAR_RE.sub(repl, obj)
    if isinstance(obj, dict):
        return {k: expand_vars(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [expand_vars(v) for v in obj]
    return obj


def load_config(path: Union[str, Path]) -> dict:
    """Load a YAML config and expand environment variables in its values."""
    setup_env()
    with open(path, "r") as f:
        cfg = yaml.safe_load(f) or {}
    return expand_vars(cfg)
