"""Repository paths and environment defaults.

Ekka configs refer to two locations through environment variables:

* ``EKKA_ROOT``: the root of this repository (defaults to the checkout location).
* ``EKKA_ACTIVATION_DIR``: where collected activation tensors are written and read
  (defaults to ``$EKKA_ROOT/activations``). Activations can be tens of GB, so point
  this at a large local disk.

Values from a ``.env`` file at the repository root are loaded if python-dotenv is
installed. Variables that are already set in the environment take precedence.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
AGENT_ROOT = REPO_ROOT / "agent-langgraph"

_ENV_READY = False


def setup_env() -> None:
    """Load ``.env`` (if present) and fill in default values for Ekka variables."""
    global _ENV_READY
    if _ENV_READY:
        return
    env_file = REPO_ROOT / ".env"
    if env_file.exists():
        try:
            from dotenv import load_dotenv  # type: ignore

            load_dotenv(env_file, override=False)
        except ImportError:
            pass
    os.environ.setdefault("EKKA_ROOT", str(REPO_ROOT))
    os.environ.setdefault("EKKA_ACTIVATION_DIR", os.path.join(os.environ["EKKA_ROOT"], "activations"))
    _ENV_READY = True


def add_repo_to_syspath() -> None:
    """Make ``import common``, ``import scripts...``, ``import my_agent...`` work from any entry point."""
    for p in (str(REPO_ROOT), str(AGENT_ROOT)):
        if p not in sys.path:
            sys.path.append(p)
