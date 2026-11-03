#!/usr/bin/env bash
# Build a class index for a framework source tree and copy it into bug-case directories.
#
# The diagnosis agent looks up class definitions through two indexes inside each case directory:
#   <case>/reference_index  -> the reference implementation (Hugging Face transformers / diffusers)
#   <case>/target_index     -> the engine under test (vLLM or SGLang)
#
# Usage:
#   scripts/update_index/build_index.sh <reference|target> <framework_source_root> <case_dir> [<case_dir> ...]
#
# Examples:
#   # transformers installed in the HF environment
#   scripts/update_index/build_index.sh reference \
#       "$(python -c 'import transformers, os; print(os.path.dirname(transformers.__file__))')" \
#       bug-cases/vllm/vllm-26812
#   # vLLM checkout used to reproduce the bug
#   scripts/update_index/build_index.sh target /path/to/vllm/vllm bug-cases/vllm/vllm-26812
#   # SGLang checkout
#   scripts/update_index/build_index.sh target /path/to/sglang/python bug-cases/sglang/sglang-4807
set -euo pipefail

if [[ $# -lt 3 ]]; then
  sed -n '2,20p' "$0" >&2
  exit 1
fi

KIND="$1"; SRC_ROOT="$2"; shift 2
case "$KIND" in
  reference|target) ;;
  *) echo "First argument must be 'reference' or 'target', got '$KIND'" >&2; exit 1 ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EKKA_ROOT="${EKKA_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
TMP_OUT="$(mktemp -d)"
trap 'rm -rf "$TMP_OUT"' EXIT

python3 "$SCRIPT_DIR/scan_python_classes.py" --root "$SRC_ROOT" --out-dir "$TMP_OUT/${KIND}_index"

for case_dir in "$@"; do
  [[ "$case_dir" = /* ]] || case_dir="$EKKA_ROOT/$case_dir"
  rm -rf "$case_dir/${KIND}_index"
  cp -r "$TMP_OUT/${KIND}_index" "$case_dir/${KIND}_index"
  echo "Wrote $case_dir/${KIND}_index"
done
