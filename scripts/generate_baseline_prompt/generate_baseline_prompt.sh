#!/usr/bin/env bash
# Regenerate baseline/for_docker/bug-cases/*/*/prompt_top{1,3}.md from their prompt_top{1,3}.yaml configs.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${EKKA_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
export EKKA_ROOT="$ROOT"
BUILD_PROMPT_PY="${ROOT}/baseline/build_prompt.py"
BUG_CASES_DIR="${ROOT}/baseline/for_docker/bug-cases"

if [[ ! -f "${BUILD_PROMPT_PY}" ]]; then
  echo "Missing build_prompt.py at ${BUILD_PROMPT_PY}" >&2
  exit 1
fi

case_count=0
prompt_count=0

for case_dir in "${BUG_CASES_DIR}"/*/*; do
  if [[ ! -d "${case_dir}" ]]; then
    continue
  fi

  case_count=$((case_count + 1))

  for top_k in 1 3; do
    config_path="${case_dir}/prompt_top${top_k}.yaml"
    output_path="${case_dir}/prompt_top${top_k}.md"

    if [[ ! -f "${config_path}" ]]; then
      continue
    fi

    python3 "${BUILD_PROMPT_PY}" \
      --config "${config_path}" \
      --output "${output_path}"

    prompt_count=$((prompt_count + 1))
  done
done

echo "Processed ${case_count} cases and built ${prompt_count} prompt files."
