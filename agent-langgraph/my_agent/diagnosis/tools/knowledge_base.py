"""LLM tools for browsing the alignment-example knowledge base (agent-training/knowledge_base)."""

from typing import List, Dict, Any, Optional
from langchain.tools import tool
from pathlib import Path
import logging

from my_agent.diagnosis import runtime


def _collect_alignment_examples(root: Optional[Path]) -> Dict[str, List[str]]:
    """Internal helper to build an index of alignment example files."""
    index: Dict[str, List[str]] = {}
    if root is None or not root.exists():
        return index

    for model_dir in sorted(root.iterdir()):
        if not model_dir.is_dir():
            continue
        files = sorted(
            [
                path.name
                for path in model_dir.iterdir()
                if path.is_file() and path.suffix == ".py"
            ]
        )
        if files:
            index[model_dir.name] = files
    return index


@tool("list_alignment_examples")
def list_alignment_examples(model: Optional[str] = None) -> Dict[str, Any]:
    """List available activation-alignment example snippets grouped by model.
    
    Args:
        model (str, optional): Filter to a specific model directory name (e.g., "Qwen2.5-7B-Instruct").
    """
    logging.info(f"[Tool][list_alignment_examples] Listing examples for model={model}")
    root = runtime.KNOWLEDGE_BASE_PATH
    index = _collect_alignment_examples(root)
    if not index:
        return {
            "error": "No alignment examples found.",
            "hint": "Ensure knowledge_base_path is configured.",
        }

    if model:
        if model not in index:
            return {
                "error": f"Model '{model}' not found under {root}.",
                "available_models": list(index.keys()),
            }
        return {
            "root": str(root),
            "model": model,
            "files": index[model],
            "note": "Use get_alignment_example_code(model, filename) to fetch code.",
        }

    return {
        "root": str(root),
        "models": index,
        "note": "Use get_alignment_example_code(model, filename) to fetch code.",
    }


@tool("get_alignment_example_code")
def get_alignment_example_code(model: str, filename: str) -> Dict[str, Any]:
    """Return the code for a specific activation-alignment example."""
    logging.info(f"[Tool][get_alignment_example_code] Fetching {filename} for model={model}")
    if runtime.KNOWLEDGE_BASE_PATH is None:
        return {"error": "knowledge_base_path is not configured."}

    root = runtime.KNOWLEDGE_BASE_PATH.resolve()
    model_dir = (root / model).resolve()

    if not model_dir.exists() or not model_dir.is_dir() or root not in model_dir.parents and model_dir != root:
        return {"error": f"Model directory '{model_dir}' does not exist under {root}."}

    target_path = (model_dir / filename).resolve()
    if model_dir not in target_path.parents and model_dir != target_path:
        return {"error": "Invalid file path requested."}
    if not target_path.exists() or not target_path.is_file():
        return {"error": f"File '{filename}' not found in {model_dir}."}

    try:
        code = target_path.read_text()
    except Exception as exc:
        return {"error": f"Failed to read file: {exc}"}

    return {
        "model": model,
        "filename": filename,
        "path": str(target_path),
        "code": code,
    }
