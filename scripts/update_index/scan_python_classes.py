#!/usr/bin/env python3
"""
scan_python_classes.py

Statically enumerate class definitions under a path (e.g. vLLM/vllm/model_executor),
with special attention to torch.nn.Module subclasses (e.g., RowParallelLinear).
Optionally, parse a specific model implementation file to guess which classes are
likely used (via constructor calls / references).

Also provides an OPTIONAL runtime helper (when imported as a module) to introspect
an *already-instantiated* model and list the concrete Python classes of submodules.

Usage examples:

  # 1) Static scan of all classes under model_executor and write CSV/JSON reports
  python scan_python_classes.py \
      --root /path/to/vllm/vllm/model_executor \
      --out-dir /path/to/class_index

  # 2) Additionally, point to a model file (e.g., qwen3.py) to rank likely-used classes
  python scan_python_classes.py \
      --root /path/to/vllm/vllm/model_executor \
      --model-file /path/to/vllm/vllm/model_executor/models/qwen3.py \
      --out-dir /path/to/class_index
"""

import argparse
import ast
import json
import csv
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional


def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding='utf-8', errors='ignore')
    except Exception:
        return ""


def get_class_defs(py_path: Path) -> List[dict]:
    src = read_text(py_path)
    if not src.strip():
        return []
    try:
        tree = ast.parse(src, filename=str(py_path))
    except SyntaxError:
        return []

    classes = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            bases = []
            for b in node.bases:
                try:
                    bases.append(ast.unparse(b))  # py3.9+
                except Exception:
                    if isinstance(b, ast.Name):
                        bases.append(b.id)
                    else:
                        bases.append(type(b).__name__)
            doc = ast.get_docstring(node) or ""
            classes.append({
                "name": node.name,
                "bases": bases,
                "lineno": getattr(node, "lineno", None),
                "col_offset": getattr(node, "col_offset", None),
                "doc_firstline": doc.strip().splitlines()[0] if doc else "",
                "file": str(py_path),
            })
    return classes


def is_nn_module(class_meta: dict) -> bool:
    bases = " ".join(class_meta.get("bases", []))
    patterns = [
        r"\bnn\.Module\b",
        r"\btorch\.nn\.Module\b",
        r"\bModule\b",
        r"\bTorchModule\b",
    ]
    return any(re.search(p, bases) for p in patterns)


def scan_root(root: Path) -> Dict[str, List[dict]]:
    all_defs: Dict[str, List[dict]] = {}
    for py in root.rglob("*.py"):
        defs = get_class_defs(py)
        for c in defs:
            key = c["name"]
            all_defs.setdefault(key, []).append(c)
    return all_defs


def extract_identifiers_from_calls(src: str) -> List[str]:
    cand = re.findall(r"\b([A-Z][A-Za-z0-9_]+)\s*\(", src)
    cand += re.findall(r"\b([A-Z][A-Za-z0-9_]+)\.", src)
    seen = set()
    out = []
    for t in cand:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


def rank_likely_used(model_file: Path, class_index: Dict[str, List[dict]]) -> List[dict]:
    src = read_text(model_file)
    ids = extract_identifiers_from_calls(src)
    results = []
    for name in ids:
        metas = class_index.get(name, [])
        if not metas:
            continue
        metas_sorted = sorted(metas, key=lambda m: (not is_nn_module(m), m["file"]))
        results.append({
            "name": name,
            "defs": metas_sorted,
        })
    return results


def collect_classes_from_model(model) -> List[str]:
    try:
        import torch
    except Exception:
        torch = None

    classes = set()
    if hasattr(model, "modules"):
        it = model.modules()
    else:
        it = []
        stack = [model]
        seen = set()
        while stack:
            obj = stack.pop()
            if id(obj) in seen:
                continue
            seen.add(id(obj))
            it.append(obj)
            for v in vars(obj).values():
                if hasattr(v, "__dict__") or hasattr(v, "modules"):
                    stack.append(v)

    for m in it or []:
        try:
            cls = m.__class__
            classes.add(f"{cls.__module__}.{cls.__name__}")
        except Exception:
            pass
    return sorted(classes)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, required=True,
                    help="Path to scan (e.g., /path/to/vllm/vllm/model_executor)")
    ap.add_argument("--model-file", type=str, default=None,
                    help="Optional: model .py file to rank likely used classes (e.g., qwen3.py)")
    ap.add_argument("--out-dir", type=str, default=None,
                    help="Output directory for JSON/CSV files. If omitted, prints to stdout.")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    if not root.exists():
        raise SystemExit(f"--root does not exist: {root}")

    class_index = scan_root(root)

    flat_rows = []
    for name, metas in class_index.items():
        for m in metas:
            flat_rows.append({
                "class": name,
                "file": m["file"],
                "lineno": m["lineno"],
                "bases": " ; ".join(m["bases"]),
                "is_nn_module": str(is_nn_module(m)),
                "doc_firstline": m["doc_firstline"],
            })

    out = {
        "total_unique_classes": len(class_index),
        "by_class": class_index,
    }

    used_ranked = None
    if args.model_file:
        model_file = Path(args.model_file).resolve()
        if model_file.exists():
            used_ranked = rank_likely_used(model_file, class_index)

    if args.out_dir:
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        json_path = out_dir / "classes.json"
        csv_path = out_dir / "classes.csv"
        with json_path.open("w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["class", "file", "lineno", "bases", "is_nn_module", "doc_firstline"])
            writer.writeheader()
            for r in flat_rows:
                writer.writerow(r)

        if used_ranked is not None:
            used_json = out_dir / "used.json"
            with used_json.open("w", encoding="utf-8") as f:
                json.dump(used_ranked, f, indent=2, ensure_ascii=False)
            print(f"Wrote: {used_json}")

        print(f"Wrote: {json_path}")
        print(f"Wrote: {csv_path}")
    else:
        print(json.dumps(out, indent=2, ensure_ascii=False))
        if used_ranked is not None:
            print("\nLikely used in model file:")
            print(json.dumps(used_ranked, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()