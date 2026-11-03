#!/usr/bin/env python3
"""
extract_forward_from_index.py

Reuse a prebuilt class inventory (classes.json / used.json) to resolve and extract
the effective `forward` method for one or more classes, without rescanning the repo.
Now with a **PyTorch (torch.nn) fallback**: if a class isn't found in your index,
we attempt to import `torch.nn.<Class>` and extract its `forward` via `inspect`.

Inputs (by default):
  --index-dir /path/to/target_index
    - expects:
        classes.json  # produced by scan_vllm_classes.py
        used.json     # optional

Output: JSON mapping each requested class -> {
  "decl_class": "...",        # class where forward is declared
  "file": "/abs/path.py",
  "lineno": <int>,
  "end_lineno": <int>,
  "source": "<function source code>",
}

Example:
  python extract_forward_from_index.py \
    --index-dir /path/to/target_index \
    --classes RowParallelLinear Embedding Linear \
    --pretty
"""

import argparse
import ast
import json
import inspect
from pathlib import Path
from typing import Dict, List, Optional, Any

# ---------------------------
# Data model
# ---------------------------

class ClassInfo:
    __slots__ = ("name", "file", "lineno", "end_lineno", "bases")
    def __init__(self, name: str, file: str, lineno: int, end_lineno: int, bases: List[str]):
        self.name = name
        self.file = Path(file)
        self.lineno = lineno
        self.end_lineno = end_lineno
        self.bases = bases

# ---------------------------
# Utils
# ---------------------------

def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding='utf-8', errors='ignore')
    except Exception:
        return ""

def load_index(index_dir: Path) -> Dict[str, List[ClassInfo]]:
    classes_path = index_dir / "classes.json"
    if not classes_path.exists():
        raise SystemExit(f"Missing {classes_path} - generate it with scan_vllm_classes.py")

    data = json.loads(read_text(classes_path))
    # classes.json format: { "total_unique_classes": N, "by_class": { Name: [ {file, bases, lineno...}, ... ] } }
    by_class = data.get("by_class", {})
    index: Dict[str, List[ClassInfo]] = {}
    for name, metas in by_class.items():
        lst = []
        for m in metas:
            lst.append(ClassInfo(
                name=name,
                file=m["file"],
                lineno=m.get("lineno") or 1,
                end_lineno=m.get("end_lineno") or (m.get("lineno") or 1),
                bases=m.get("bases", []),
            ))
        index[name] = lst
    return index

def parse_python_file_for_class(path: Path, classname: str):
    """Minimal parser for a specific file to find a class definition by name.
    Returns (class_node, source_text) or (None, source_text)."""
    src = read_text(path)
    if not src.strip():
        return None, src
    try:
        tree = ast.parse(src, filename=str(path))
    except SyntaxError:
        return None, src
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == classname:
            src = ast.unparse(node)
            return node, src
    return None, src

def find_forward_in_classnode(cls_node: ast.ClassDef) -> Optional[ast.FunctionDef]:
    for body_node in cls_node.body:
        if isinstance(body_node, ast.FunctionDef) and body_node.name == "forward":
            return body_node
    return None

def extract_source_segment(src: str, node: ast.AST) -> str:
    start = getattr(node, "lineno", None)
    end = getattr(node, "end_lineno", None)
    if start is None:
        return ""
    lines = src.splitlines(True)
    end = end or start
    frag = "".join(lines[start-1:end])
    return frag

def resolve_forward(class_name: str,
                    index: Dict[str, List[ClassInfo]],
                    visited: Optional[set] = None) -> Optional[dict]:
    """Resolve the effective forward for class_name using only the prebuilt index.
    If not present, return None."""
    visited = visited or set()
    if class_name in visited:
        return None
    visited.add(class_name)

    candidates = index.get(class_name, [])
    # Prefer definition with the earliest occurrence (by file path then line)
    candidates = sorted(candidates, key=lambda ci: (str(ci.file), ci.lineno))

    for ci in candidates:
        # Parse only this file to check for forward
        cls_node, src = parse_python_file_for_class(ci.file, ci.name)
        if cls_node is not None:
            fwd = find_forward_in_classnode(cls_node)
            if fwd is not None:
                return {
                    "decl_class": ci.name,
                    "file": str(ci.file),
                    "lineno": getattr(fwd, "lineno", None),
                    "end_lineno": getattr(fwd, "end_lineno", None),
                    "source": extract_source_segment(src, fwd),
                }

            # walk bases: trailing identifier only, relying on index names
            for base_expr in [b if isinstance(b, str) else str(b) for b in ci.bases]:
                base_name = base_expr.split(".")[-1]
                if base_name == ci.name:
                    continue
                if base_name in index:
                    resolved = resolve_forward(base_name, index, visited)
                    if resolved:
                        return resolved

    return None

# ---------------------------
# torch.nn fallback
# ---------------------------

def _normalize_torch_name(name: str) -> str:
    """Accept 'Embedding', 'nn.Embedding', or 'torch.nn.Embedding' and return 'Embedding'."""
    if name.startswith("torch.nn."):
        return name.split(".")[-1]
    if name.startswith("nn."):
        return name.split(".")[-1]
    return name

def resolve_torch_forward(class_name: str) -> Optional[dict]:
    """Try to resolve a torch.nn class by name (e.g., 'Embedding', 'Linear')."""
    base = _normalize_torch_name(class_name)
    try:
        import torch.nn as nn  # import locally to avoid hard dep if not needed
        cls = getattr(nn, base, None)
        if cls is None:
            return None
        # Prefer forward; fallback to __call__ only if needed
        func = getattr(cls, "forward", None) or getattr(cls, "__call__", None)
        if func is None:
            return {"error": f"torch.nn.{base} has no forward or __call__"}
        try:
            src = inspect.getsource(func)
            file = inspect.getsourcefile(func)
            lines, start = inspect.getsourcelines(func)
            return {
                "decl_class": f"torch.nn.{base}",
                "file": file,
                "lineno": start,
                "end_lineno": start + len(lines) - 1,
                "source": "".join(lines),
            }
        except Exception as e:
            return {"error": f"Cannot retrieve source for torch.nn.{base}: {e}"}
    except Exception as e:
        return {"error": f"Cannot import torch.nn: {e}"}
    
def resolve_torch_class_definition(class_name: str) -> Optional[Dict]:
    """Return the full class definition (source code) for a torch.nn class by name."""
    def _normalize_torch_name(name: str) -> str:
        return name.strip().split('.')[-1]  # e.g., handle "torch.nn.Linear"

    base = _normalize_torch_name(class_name)

    try:
        import torch.nn as nn
        cls = getattr(nn, base, None)
        if cls is None:
            return {"error": f"torch.nn has no attribute '{base}'"}

        try:
            # Get full class definition
            src = inspect.getsource(cls)
            file = inspect.getsourcefile(cls)
            lines, start = inspect.getsourcelines(cls)
            return {
                "decl_class": f"torch.nn.{base}",
                "file": file,
                "lineno": start,
                "end_lineno": start + len(lines) - 1,
                "source": "".join(lines),
            }
        except Exception as e:
            return {"error": f"Cannot retrieve class source for torch.nn.{base}: {e}"}

    except Exception as e:
        return {"error": f"Cannot import torch.nn: {e}"}

# ---------------------------
# CLI + Public API
# ---------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index-dir", type=str, required=True,
                    help="Directory containing classes.json (and optionally used.json)")
    ap.add_argument("--classes", nargs="+", required=True,
                    help="Class names to resolve (e.g., RowParallelLinear FusedRMSNorm Embedding)")
    ap.add_argument("--pretty", action="store_true", help="Pretty-print JSON")
    args = ap.parse_args()

    out = extract_forward_from_index(args.index_dir, args.classes)
    if args.pretty:
        print(json.dumps(out, indent=2, ensure_ascii=False))
    else:
        print(json.dumps(out, ensure_ascii=False))

def extract_forward_from_index(index_dir: str, classes: List[str]) -> Dict[str, Dict[str, Any]]:
    """Resolve forwards for classes using the prebuilt index, with torch.nn fallback.

    Parameters
    ----------
    index_dir : str
        Directory containing classes.json
    classes : List[str]
        Class names such as ['RowParallelLinear', 'Embedding', 'nn.Linear']

    Returns
    -------
    Dict[str, Dict[str, Any]]
        Mapping class_name -> result dict with forward metadata OR {error: ...}
    """
    index_dir = Path(index_dir).resolve()
    index = load_index(index_dir)
    out: Dict[str, Dict[str, Any]] = {}

    for cls in classes:
        # 1) Try local index (project classes)
        res = resolve_forward(cls, index)
        if res is not None:
            out[cls] = res
            continue
        # 2) Try torch.nn fallback
        torch_res = resolve_torch_forward(cls)
        if torch_res is not None:
            out[cls] = torch_res
            continue
        # 3) Not found
        raise ValueError("class not found in index or torch.nn")

    return out

if __name__ == "__main__":
    main()
