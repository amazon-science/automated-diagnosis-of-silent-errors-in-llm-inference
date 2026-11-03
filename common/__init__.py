"""Shared helpers for Ekka: repository paths, config loading, and LLM construction."""

from .paths import REPO_ROOT, setup_env
from .config import load_config, expand_vars

__all__ = ["REPO_ROOT", "setup_env", "load_config", "expand_vars"]
