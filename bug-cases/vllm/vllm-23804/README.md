
# vLLM issue 23804
Qwen3-reranker accuracy issue
issue link: https://github.com/vllm-project/vllm/issues/23804
fix PR: https://github.com/vllm-project/vllm/pull/23836


## vLLM Build

marlin generation failed
https://github.com/vllm-project/vllm/issues/18380

python3 -m pip install jinja2
conda install mkl mkl-devel 


CUDA::nvToolsExt not found on install workaround in CMake for local build
https://github.com/vllm-project/vllm/issues/20892

Update main CMakelist according to the issue description 


## reproduction
HF TP=4 scores
scores: [0.99951171875, 0.99951171875]

vLLM TP=4 scores
fixed
scores [0.9994843006134033, 0.9993563294410706]

original
scores [0.3294568359851837, 0.25639307498931885]
