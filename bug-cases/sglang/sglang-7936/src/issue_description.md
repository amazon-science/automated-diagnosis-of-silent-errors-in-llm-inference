[Bug] garbage output when page size > 1 for llama4 #7936
Closed
Closed
[Bug] garbage output when page size > 1 for llama4
#7936
@Ying1123
Description
Ying1123
opened on Jul 10, 2025
Contributor
Checklist

1. I have searched related issues but cannot get the expected help.

2. The bug has not been fixed in the latest version.

3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.

4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.

5. Please use English, otherwise it will be closed.
Describe the bug
If not supported, we need to assert with hint.

Reproduction
python3 -m sglang.launch_server --model-path meta-llama/Llama-4-Scout-17B-16E-Instruct  --tp 8 --mem-fraction-static 0.8 --context-length 100000 --cuda-graph-bs 1 2 4 8 16 32 --page-size 4
python -m sglang.test.send_one
Environment
Python: 3.10.12 (main, Feb 4 2025, 14:57:36) [GCC 11.4.0]
CUDA available: True
GPU 0,1,2,3,4,5,6,7: NVIDIA H100 80GB HBM3
GPU 0,1,2,3,4,5,6,7 Compute Capability: 9.0
CUDA_HOME: /usr/local/cuda
NVCC: Cuda compilation tools, release 12.8, V12.8.93
CUDA Driver Version: 550.54.15
PyTorch: 2.7.1+cu126
sglang: 0.4.9
sgl_kernel: 0.2.4
flashinfer_python: 0.2.7.post1
triton: 3.3.1
transformers: 4.53.0
torchao: 0.9.0
numpy: 1.26.4
aiohttp: 3.11.14
fastapi: 0.116.0
hf_transfer: 0.1.9
huggingface_hub: 0.33.2
interegular: 0.3.3
modelscope: 1.27.1
orjson: 3.10.18
outlines: 0.1.11
packaging: 24.2
psutil: 7.0.0
pydantic: 2.11.7
python-multipart: 0.0.20
pyzmq: 27.0.0
uvicorn: 0.35.0
uvloop: 0.21.0
vllm: Module Not Found
xgrammar: 0.1.19
openai: 1.93.2
tiktoken: 0.9.0
anthropic: 0.57.1
litellm: 1.74.0.post1
decord: 0.6.0

