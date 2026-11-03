"""
evaluator.py

Run evaluation from a single YAML config.

Stages:
  1) Component mapping coverage
  2) Call-sequence prefix alignment quality
  3) Final prediction top-k hit (exact match)

Config flags:
  - print_details: bool
  - save_report: bool
  - report_name: str (default: eval_report.json)
"""

from __future__ import annotations

import argparse
import itertools
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, FrozenSet, Iterable, List, Optional, Tuple


# ----------------------------
# Types
# ----------------------------

MappingKey = Tuple[FrozenSet[str], FrozenSet[str]]


# ----------------------------
# Config
# ----------------------------

@dataclass(frozen=True)
class EvalConfig:
    gt_component_mapping: Path
    gt_buggy_component: Path
    agent_component_mapping: Path
    agent_call_sequence: Path
    agent_detection_result: Path
    error_ratio_threshold: float = 1.5

    print_details: bool = False
    save_report: bool = False
    report_name: str = "eval_report.json"


def _read_yaml(path: Path) -> Dict[str, Any]:
    """Read a YAML config, expanding ${EKKA_ROOT} and other environment variables."""
    import sys

    repo_root = str(Path(__file__).resolve().parents[1])
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    from common.config import load_config as _load_cfg

    return _load_cfg(path)


def load_config(config_path: str | Path, result_dir_override: str | Path | None = None) -> EvalConfig:
    cfg_path = Path(config_path).expanduser().resolve()
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")

    cfg = _read_yaml(cfg_path)

    # gt_dir defaults to the directory holding the eval config (bug-cases/<engine>/<case>/evaluation).
    gt_dir = Path(cfg.get("gt_dir", cfg_path.parent)).expanduser().resolve()
    if result_dir_override is not None:
        result_dir = Path(result_dir_override).expanduser().resolve()
    elif cfg.get("result_dir"):
        result_dir = Path(cfg["result_dir"]).expanduser().resolve()
    else:
        raise ValueError("No result directory given: pass --result-dir <diagnosis_output_dir> or set result_dir in the config.")

    gt_component_mapping = gt_dir / cfg.get("gt_component_mapping", "gt_component_mapping.json")
    gt_buggy_component = gt_dir / cfg.get("gt_buggy_component", "gt_buggy_component.json")

    agent_component_mapping = result_dir / cfg.get("agent_component_mapping", "component_mapping.json")
    agent_call_sequence = result_dir / cfg.get("agent_call_sequence", "call_sequence_aligned.json")
    agent_detection_result = result_dir / cfg.get("agent_detection_result", "detection_result.json")

    threshold = float(cfg.get("error_ratio_threshold", 1.5))

    print_details = bool(cfg.get("print_details", False))
    save_report = bool(cfg.get("save_report", False))
    report_name = str(cfg.get("report_name", "eval_report.json"))

    return EvalConfig(
        gt_component_mapping=gt_component_mapping,
        gt_buggy_component=gt_buggy_component,
        agent_component_mapping=agent_component_mapping,
        agent_call_sequence=agent_call_sequence,
        agent_detection_result=agent_detection_result,
        error_ratio_threshold=threshold,
        print_details=print_details,
        save_report=save_report,
        report_name=report_name,
    )


# ----------------------------
# IO helpers
# ----------------------------

def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


# ----------------------------
# Stage 1: mapping coverage
# ----------------------------

def as_set(items: Iterable[str]) -> FrozenSet[str]:
    return frozenset(items)

def mapping_entry_key(entry: List[List[str]]) -> MappingKey:
    hf_list, vllm_list = entry
    return (as_set(hf_list), as_set(vllm_list))

def pretty_mapping_key(key: MappingKey) -> str:
    hf, vllm = key
    return f"HF={sorted(hf)}  <->  vLLM={sorted(vllm)}"

def load_mapping_keys(path: Path) -> set[MappingKey]:
    data = read_json(path)
    return {mapping_entry_key(p) for p in data["component_mapping"]}

def mapping_key_to_jsonable(key: MappingKey) -> Dict[str, List[str]]:
    """Convert (frozenset, frozenset) -> JSON-friendly dict of sorted lists."""
    hf, vllm = key
    return {"HF": sorted(hf), "vLLM": sorted(vllm)}


def score_mapping_coverage(gt: set[MappingKey], agent: set[MappingKey]) -> Dict[str, Any]:
    missing_keys = sorted(gt - agent, key=lambda x: (sorted(x[0]), sorted(x[1])))
    extra_keys   = sorted(agent - gt, key=lambda x: (sorted(x[0]), sorted(x[1])))

    total = len(gt)
    covered = total - len(missing_keys)
    coverage = covered / total if total else 1.0

    return {
        "gt_total": total,
        "covered": covered,
        "coverage": coverage,
        "coverage_pct": coverage * 100.0,

        # JSON-friendly versions:
        "missing": [mapping_key_to_jsonable(k) for k in missing_keys],
        "extra":   [mapping_key_to_jsonable(k) for k in extra_keys],

        # If you still want pretty strings for printing/debugging:
        "missing_pretty": [pretty_mapping_key(k) for k in missing_keys],
        "extra_pretty":   [pretty_mapping_key(k) for k in extra_keys],
    }



# ----------------------------
# Stage 2: call-sequence prefix alignment
# ----------------------------

def expand_with_variables(pattern: str, variables: Dict[str, List[str]]) -> List[str]:
    vars_in = re.findall(r"\{(\w+)\}", pattern)
    if not vars_in:
        return [pattern]

    values = [variables[v] for v in vars_in]
    out: List[str] = []
    for combo in itertools.product(*values):
        s = pattern
        for v, val in zip(vars_in, combo):
            s = s.replace(f"{{{v}}}", val)
        out.append(s)
    return out

def list_mappable_components(gt_component_mapping: Dict[str, Any], side: str = "vLLM") -> set[str]:
    assert side in ("HF", "vLLM")
    variables = gt_component_mapping.get("variables", {})
    pairs = gt_component_mapping["component_mapping"]

    out: set[str] = set()
    for hf_list, vllm_list in pairs:
        chosen = hf_list if side == "HF" else vllm_list
        for item in chosen:
            out.update(expand_with_variables(item, variables))
    return out

def score_call_sequence_prefix(
    *,
    gt_buggy_components: List[str],
    call_sequence: List[List[Any]],
    mappable_components: set[str],
    error_ratio_threshold: float,
) -> Dict[str, Any]:
    """
    Find earliest GT buggy component in the call sequence.
    Evaluate the prefix before it; only consider entries that are "mappable".

    Returns full lists (not samples):
      - skipped_not_mappable_components: all prefix component names skipped because not mappable
      - incorrect_components: all prefix components considered but not correct (False or >= threshold)
    """
    # Record first occurrence for each buggy component
    first_pos: Dict[str, int] = {}
    for idx, (name, _val) in enumerate(call_sequence):
        if name in gt_buggy_components and name not in first_pos:
            first_pos[name] = idx

    if not first_pos:
        return {
            "score": 0.0,
            "error": "No ground-truth buggy component found in call_sequence_aligned.json",
            "buggy_components": gt_buggy_components,
        }

    earliest_name, earliest_idx = min(first_pos.items(), key=lambda kv: kv[1])
    prefix = call_sequence[:earliest_idx]

    considered = 0
    correct = 0

    skipped_not_mappable_components: List[str] = []
    incorrect_components: List[Dict[str, Any]] = []

    for name, val in prefix:
        if name not in mappable_components:
            skipped_not_mappable_components.append(name)
            continue

        considered += 1
        ok = (val is not False) and isinstance(val, (int, float)) and (val < error_ratio_threshold)
        if ok:
            correct += 1
        else:
            incorrect_components.append({
                "component": name,
                "error_ratio": val,
            })

    score = (correct / considered) if considered else 0.0

    return {
        "score": score,
        "error_ratio_threshold": error_ratio_threshold,
        "earliest_buggy_component": earliest_name,
        "earliest_buggy_index": earliest_idx,
        "prefix_len": len(prefix),
        "considered_mappable": considered,
        "correct_mappable": correct,
        "skipped_not_mappable": len(skipped_not_mappable_components),
        "skipped_not_mappable_components": skipped_not_mappable_components,
        "incorrect_components": incorrect_components,
        "positions_of_buggy_components": first_pos,
    }



# ----------------------------
# Stage 3: final prediction (exact match)
# ----------------------------

def score_final_prediction_exact(
    gt_buggy_components: List[str],
    suspected_components: List[str],
) -> Dict[str, Any]:
    buggy = set(gt_buggy_components)
    suspected = list(suspected_components)

    matched = sorted(set(suspected).intersection(buggy))
    hit = len(matched) > 0

    return {
        "hit": hit,
        "topk": len(suspected),
        "num_buggy": len(buggy),
        "matched_components": matched,
    }


# ----------------------------
# Full evaluation
# ----------------------------

def run_evaluation(cfg: EvalConfig) -> Dict[str, Any]:
    stage1 = None
    stage2 = None
    stage3 = None

    gt_buggy_components: List[str] | None = None
    gt_map: Dict[str, Any] | None = None

    if cfg.gt_buggy_component.exists():
        gt_bug = read_json(cfg.gt_buggy_component)
        gt_buggy_components = gt_bug.get("buggy_component", [])

    if cfg.gt_component_mapping.exists():
        gt_map = read_json(cfg.gt_component_mapping)

    # Stage 1
    if cfg.gt_component_mapping.exists() and cfg.agent_component_mapping.exists():
        gt_keys = load_mapping_keys(cfg.gt_component_mapping)
        agent_keys = load_mapping_keys(cfg.agent_component_mapping)
        stage1 = score_mapping_coverage(gt_keys, agent_keys)

    # Stage 2
    if (
        gt_buggy_components is not None
        and gt_map is not None
        and cfg.agent_call_sequence.exists()
    ):
        call_seq = read_json(cfg.agent_call_sequence)
        mappable_vllm = list_mappable_components(gt_map, side="vLLM")
        stage2 = score_call_sequence_prefix(
            gt_buggy_components=gt_buggy_components,
            call_sequence=call_seq,
            mappable_components=mappable_vllm,
            error_ratio_threshold=cfg.error_ratio_threshold,
        )

    # Stage 3
    if gt_buggy_components is not None and cfg.agent_detection_result.exists():
        det = read_json(cfg.agent_detection_result)
        suspected_components: List[str] = det.get("suspected_components", [])
        stage3 = score_final_prediction_exact(
            gt_buggy_components=gt_buggy_components,
            suspected_components=suspected_components,
        )

    return {
        "inputs": {
            "gt_component_mapping": str(cfg.gt_component_mapping),
            "gt_buggy_component": str(cfg.gt_buggy_component),
            "agent_component_mapping": str(cfg.agent_component_mapping),
            "agent_call_sequence": str(cfg.agent_call_sequence),
            "agent_detection_result": str(cfg.agent_detection_result),
        },
        "stage1_mapping_coverage": stage1,
        "stage2_call_sequence_prefix": stage2,
        "stage3_final_prediction": stage3,
    }


# ----------------------------
# CLI
# ----------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="YAML config path")
    ap.add_argument("--result-dir", help="Override result_dir in the config")
    args = ap.parse_args()

    cfg = load_config(args.config, result_dir_override=args.result_dir)
    report = run_evaluation(cfg)

    print(json.dumps(report, indent=2))

    if cfg.print_details:
        s1 = report["stage1_mapping_coverage"]
        if s1["missing_pretty"]:
            print("\n[Stage 1] Missing GT mapping entries:")
            for s in s1["missing_pretty"]:
                print("  -", s)
        if s1["extra_pretty"]:
            print("\n[Stage 1] Extra mapping entries (agent-only):")
            for s in s1["extra_pretty"]:
                print("  -", s)


    if cfg.save_report:
        out_path = cfg.agent_component_mapping.parent / cfg.report_name
        out_path.write_text(json.dumps(report, indent=2))
        print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
