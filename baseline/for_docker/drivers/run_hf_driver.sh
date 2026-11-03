#!/bin/bash
# HF driver can use either vLLM or SGLang venv, defaulting to vLLM
# Supports tensor parallelism via --tp or --nproc-per-node argument

source /workspace/venvs/vllm/bin/activate
export PYTHONPATH="/workspace/frameworks/vllm:$PYTHONPATH"
cd /workspace/drivers

# Parse TP argument
TP=""
ARGS=()

while [[ $# -gt 0 ]]; do
    case $1 in
        --tp)
            TP="$2"
            shift 2
            ;;
        --nproc-per-node)
            TP="$2"
            shift 2
            ;;
        *)
            ARGS+=("$1")
            shift
            ;;
    esac
done

# If TP is specified, use torchrun; otherwise default to TP=1 (single process)
if [ -n "$TP" ]; then
    torchrun --nproc-per-node "$TP" driver_hf_v2_baseline.py "${ARGS[@]}"
else
    # Default to single process (TP=1) using torchrun
    torchrun --nproc-per-node 1 driver_hf_v2_baseline.py "${ARGS[@]}"
fi

