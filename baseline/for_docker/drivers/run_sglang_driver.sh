#!/bin/bash
source /workspace/venvs/sglang/bin/activate
export PYTHONPATH="/workspace/frameworks/sglang:$PYTHONPATH"
cd /workspace/drivers
python driver_sglang_baseline.py "$@"

