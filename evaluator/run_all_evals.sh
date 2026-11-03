#!/usr/bin/env bash
set -u
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
export EKKA_ROOT="${EKKA_ROOT:-$REPO_ROOT}"
BUG_ROOT="$REPO_ROOT/bug-cases"
EVAL_PY="$SCRIPT_DIR/evaluator.py"

if [[ ! -f "$EVAL_PY" ]]; then
  echo "Missing evaluator: $EVAL_PY" >&2
  exit 1
fi

total_runs=0
total_fail=0
total_skipped=0

for group in sglang vllm; do
  group_dir="$BUG_ROOT/$group"
  if [[ ! -d "$group_dir" ]]; then
    continue
  fi

  for case_dir in "$group_dir"/*; do
    if [[ ! -d "$case_dir" ]]; then
      continue
    fi

    eval_dir="$case_dir/evaluation"
    if [[ ! -d "$eval_dir" ]]; then
      ((total_skipped++))
      continue
    fi

    config="$eval_dir/eval_config.yaml"
    if [[ ! -f "$config" ]]; then
      echo "Missing config: $config" >&2
      ((total_skipped++))
      continue
    fi

    mapfile -t result_dirs < <(
      # Every run of agent-langgraph/my_agent/diagnose.py writes <case>/diagnosis_output_<timestamp>/
      find "$case_dir" -maxdepth 1 -type d -name "diagnosis_output_*" | sort
    )

    if [[ ${#result_dirs[@]} -eq 0 ]]; then
      ((total_skipped++))
      continue
    fi

    for result_dir in "${result_dirs[@]}"; do
      ((total_runs++))
      echo "==> $(basename "$case_dir") | $(basename "$result_dir")"
      if ! python3 "$EVAL_PY" --config "$config" --result-dir "$result_dir"; then
        ((total_fail++))
      fi
    done
  done
done

echo "Done. runs=$total_runs failed=$total_fail skipped=$total_skipped"
