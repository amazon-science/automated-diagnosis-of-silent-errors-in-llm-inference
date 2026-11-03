[Bug] the model result is wrong when using sglang to serve qwen3-vl-8b-instruct #17887
Closed
#18024
Closed
[Bug] the model result is wrong when using sglang to serve qwen3-vl-8b-instruct
#17887
#18024
@jet-yangqs
Description
jet-yangqs
opened on Jan 28 · edited by jet-yangqs
Checklist

I searched related issues but found no solution.

The bug persists in the latest version.

Issues without environment info and a minimal reproducible demo are hard to resolve and may receive no feedback.

If this is not a bug report but a general question, please start a discussion at https://github.com/sgl-project/sglang/discussions. Otherwise, it will be closed.

Please use English. Otherwise, it will be closed.
Describe the bug
device info:

Image
sglang version: 0.5.8

unexpected model output:

Image
Reproduction
conmand for reproduction: CUDA_VISIBLE_DEVICES=3 python3 tmp_sglang.py

code for reproduction:

`import time
from PIL import Image
from sglang import Engine
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, AutoConfig
import torch
if name == "main":
# TODO: change to your own checkpoint path
checkpoint_path = "/data1/noelyang/offline_models/Qwen/Qwen3-VL-8B-Instruct"
processor = AutoProcessor.from_pretrained(checkpoint_path)

messages = [
    {
        "role": "user",
        "content": [
          {"type": "text", "text": "who are you?"},
        ],
    }
]

text = processor.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

image_inputs, _ = process_vision_info(messages, image_patch_size=processor.image_processor.patch_size)

llm = Engine(
    model_path=checkpoint_path,
    enable_multimodal=True,
    mem_fraction_static=0.8,
    tp_size=torch.cuda.device_count(),
    attention_backend="fa3"
)

start = time.time()
sampling_params = {"max_new_tokens": 1024}
response = llm.generate(prompt=text, image_data=image_inputs, sampling_params=sampling_params)
print(f"Response costs: {time.time() - start:.2f}s")
print(f"Generated text: {response['text']}")`
Environment
python3 -m sglang.check_env
Python: 3.12.3 (main, Jan 8 2026, 11:30:50) [GCC 13.3.0]
CUDA available: True
GPU 0,1,2,3,4,5,6,7: NVIDIA L40
GPU 0,1,2,3,4,5,6,7 Compute Capability: 8.9
CUDA_HOME: /usr/local/cuda
NVCC: Cuda compilation tools, release 12.9, V12.9.86
CUDA Driver Version: 535.161.07
PyTorch: 2.9.1+cu129
sglang: 0.5.8
sgl_kernel: 0.3.21
flashinfer_python: 0.6.1
flashinfer_cubin: 0.6.1
flashinfer_jit_cache: 0.6.1+cu129
triton: 3.5.1
transformers: 4.57.1
torchao: 0.9.0
numpy: 2.4.1
aiohttp: 3.13.3
fastapi: 0.128.0
hf_transfer: 0.1.9
huggingface_hub: 0.36.0
interegular: 0.3.3
modelscope: 1.34.0
orjson: 3.11.5
outlines: 0.1.11
packaging: 26.0
psutil: 7.2.1
pydantic: 2.12.5
python-multipart: 0.0.21
pyzmq: 27.1.0
uvicorn: 0.40.0
uvloop: 0.22.1
vllm: Module Not Found
xgrammar: 0.1.27
openai: 2.6.1
tiktoken: 0.12.0
anthropic: 0.76.0
litellm: Module Not Found
decord2: 3.0.0
NVIDIA Topology:
GPU0 GPU1 GPU2 GPU3 GPU4 GPU5 GPU6 GPU7 CPU Affinity NUMA Affinity GPU NUMA ID
GPU0 X NODE NODE NODE SYS SYS SYS SYS 0-191 0 N/A
GPU1 NODE X PIX NODE SYS SYS SYS SYS 0-191 0 N/A
GPU2 NODE PIX X NODE SYS SYS SYS SYS 0-191 0 N/A
GPU3 NODE NODE NODE X SYS SYS SYS SYS 0-191 0 N/A
GPU4 SYS SYS SYS SYS X NODE NODE NODE 192-383 1 N/A
GPU5 SYS SYS SYS SYS NODE X PIX NODE 192-383 1 N/A
GPU6 SYS SYS SYS SYS NODE PIX X NODE 192-383 1 N/A
GPU7 SYS SYS SYS SYS NODE NODE NODE X 192-383 1 N/A

Legend:

X = Self
SYS = Connection traversing PCIe as well as the SMP interconnect between NUMA nodes (e.g., QPI/UPI)
NODE = Connection traversing PCIe as well as the interconnect between PCIe Host Bridges within a NUMA node
PHB = Connection traversing PCIe as well as a PCIe Host Bridge (typically the CPU)
PXB = Connection traversing multiple PCIe bridges (without traversing the PCIe Host Bridge)
PIX = Connection traversing at most a single PCIe bridge
NV# = Connection traversing a bonded set of # NVLinks

Hypervisor vendor:: KVM
ulimit soft: 1000000

