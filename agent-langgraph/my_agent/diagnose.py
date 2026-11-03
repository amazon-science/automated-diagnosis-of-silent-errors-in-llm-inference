#!/usr/bin/env python3
"""Run the Ekka diagnosis workflow on a bug case.

Usage:
    python agent-langgraph/my_agent/diagnose.py --config_path bug-cases/vllm/vllm-26812/config_with_kb.yaml

The implementation lives in the ``my_agent.diagnosis`` package (see its __init__.py for the layout).
"""

import sys
from pathlib import Path

# Make the repository root (common/, scripts/) and agent-langgraph/ (my_agent/) importable.
_REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(_REPO_ROOT), str(_REPO_ROOT / "agent-langgraph")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from my_agent.diagnosis.cli import main  # noqa: E402

if __name__ == "__main__":
    main()
