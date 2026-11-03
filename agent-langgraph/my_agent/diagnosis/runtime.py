"""Process-wide state shared by the diagnosis stages and the LLM tools.

Values here are assigned at run time (see ``cli.run_diagnosis``). Always access them as
``runtime.MODEL`` / ``runtime.KNOWLEDGE_BASE_PATH`` so that later assignments are visible."""

from pathlib import Path
from typing import Dict, Optional

import torch
# Registers DTensor with torch.load so activations saved from tensor-parallel runs can be loaded.
import torch.distributed.tensor  # noqa: F401

# Chat model shared by every agent stage. Set by run_diagnosis() from the config.
MODEL = None
# Activation tensors loaded for the current alignment trial, keyed by a readable name.
ACTIVATION_STORE: Dict[str, torch.Tensor] = {}
# Root of the alignment-example knowledge base. Populated from config["knowledge_base_path"]; None disables it.
KNOWLEDGE_BASE_PATH: Optional[Path] = None


def register_tensor(key: str, tensor: torch.Tensor):
    ACTIVATION_STORE[key] = tensor


def get_tensor(key: str) -> torch.Tensor:
    return ACTIVATION_STORE[key]
