#!/bin/bash
source /workspace/venvs/vllm/bin/activate
export PYTHONPATH="/workspace/frameworks/vllm:$PYTHONPATH"
cd /workspace/drivers
python driver_vllm_v2_baseline.py "$@"

