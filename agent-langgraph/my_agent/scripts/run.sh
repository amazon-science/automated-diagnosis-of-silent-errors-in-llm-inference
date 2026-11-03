#!/usr/bin/env bash
# Run the Ekka diagnosis workflow on one or more configs.
#
# Usage:
#   agent-langgraph/my_agent/scripts/run.sh bug-cases/vllm/vllm-26812/config_with_kb.yaml [more configs...]
#
# EKKA_ROOT defaults to the repository root; EKKA_ACTIVATION_DIR defaults to $EKKA_ROOT/activations.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export EKKA_ROOT="${EKKA_ROOT:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
export EKKA_ACTIVATION_DIR="${EKKA_ACTIVATION_DIR:-$EKKA_ROOT/activations}"

if [[ $# -eq 0 ]]; then
  echo "Usage: $0 <config.yaml> [more configs...]" >&2
  exit 1
fi

for config in "$@"; do
  echo "==> Running diagnosis with $config"
  python3 "$EKKA_ROOT/agent-langgraph/my_agent/diagnose.py" --config_path "$config"
done
