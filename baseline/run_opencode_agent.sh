#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EKKA_ROOT="${EKKA_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"

# Host directory mounted into the container at /docker_mount (agent outputs land here).
MOUNT_HOST="${EKKA_DOCKER_MOUNT:-$EKKA_ROOT/docker_mount}"
RESULTS_ROOT="${MOUNT_HOST}/results"
IMAGE="${EKKA_BASELINE_IMAGE:-ekka-baseline:latest}"

PROMPTS_ROOT="$EKKA_ROOT/baseline/for_docker/bug-cases"
CONTAINER_PROMPTS_ROOT="/workspace/bug-cases"

OPENCODE_BIN="/root/.opencode/bin/opencode"
OPENCODE_MODEL="${OPENCODE_MODEL:-litellm/bedrock-claude-sonnet-4.5}"
RUN_PROMPT="Resolve the task in the attached file"

TIMESTAMP="$(date +"%Y%m%d_%H%M%S")"

run_case_prompt() {
  local case_dir="$1"
  local prompt_path="$2"
  local prompt_file
  local variant
  local case_id
  local case_slug
  local relative_prompt
  local container_prompt
  local output_filename
  local stdout_filename
  local result_dir
  local container_cmd

  prompt_file="$(basename "$prompt_path")"
  case_id="$(basename "$case_dir")"
  case_slug="${case_id//-/_}"
  variant="${prompt_file#prompt_}"
  variant="${variant%.md}"

  relative_prompt="${prompt_path#${PROMPTS_ROOT}/}"
  container_prompt="${CONTAINER_PROMPTS_ROOT}/${relative_prompt}"

  output_filename="output_${case_slug}_${variant}.json"
  stdout_filename="stdout_${case_slug}_${variant}.log"
  result_dir="${RESULTS_ROOT}/${case_id}/${TIMESTAMP}"

  mkdir -p "${result_dir}"
  rm -f "${MOUNT_HOST:?}/${output_filename}" "${MOUNT_HOST:?}/${stdout_filename}"

  echo "Running ${case_id} ${variant} ..."
  container_cmd="${OPENCODE_BIN} run \"${RUN_PROMPT}\" -f \"${container_prompt}\" --model \"${OPENCODE_MODEL}\""

  if ! docker run --network host --gpus all -it --shm-size=2g -e HF_TOKEN -v "${MOUNT_HOST}:/docker_mount" "${IMAGE}" \
    bash -lc "${container_cmd}" 2>&1 | tee "${MOUNT_HOST}/${stdout_filename}"; then
    echo "Docker run failed for ${case_id} ${variant}" >&2
    return
  fi

  if [[ -f "${MOUNT_HOST}/${output_filename}" ]]; then
    cp "${MOUNT_HOST}/${output_filename}" "${result_dir}/${output_filename}"
  else
    echo "Missing output: ${MOUNT_HOST}/${output_filename}" >&2
  fi

  if [[ -f "${MOUNT_HOST}/${stdout_filename}" ]]; then
    cp "${MOUNT_HOST}/${stdout_filename}" "${result_dir}/${stdout_filename}"
  else
    echo "Missing stdout: ${MOUNT_HOST}/${stdout_filename}" >&2
  fi
}

# Options (environment variables):
#   TRIALS   number of runs per prompt (default 5)
#   CASES    space-separated case ids to run, e.g. "sglang-7936 vllm-26812" (default: all)
#   TOP_K    space-separated prompt variants to run (default: "1 3")
TRIALS="${TRIALS:-5}"
CASES="${CASES:-}"
TOP_K="${TOP_K:-1 3}"
mkdir -p "${MOUNT_HOST}"

for category_dir in "${PROMPTS_ROOT}"/sglang "${PROMPTS_ROOT}"/vllm; do
  [[ -d "${category_dir}" ]] || continue
  for case_dir in "${category_dir}"/*; do
    [[ -d "${case_dir}" ]] || continue
    if [[ -n "${CASES}" && " ${CASES} " != *" $(basename "${case_dir}") "* ]]; then
      continue
    fi
    for k in ${TOP_K}; do
      prompt_path="${case_dir}/prompt_top${k}.md"
      [[ -f "${prompt_path}" ]] || continue
      for i in $(seq 1 "${TRIALS}"); do
        TIMESTAMP="$(date +"%Y%m%d_%H%M%S")"
        echo "Running ${case_dir} ${prompt_path} ${i} ..."
        run_case_prompt "${case_dir}" "${prompt_path}"
      done
    done
  done
done
