You are an expert ML engineer diagnosing **accuracy divergences** between two inference engines running the **same model**.

## Goal
Given a buggy target framework and a correct reference framework, identify which **torch submodule implementation in the target framework** is most likely the root cause of the accuracy differences.

## Issue
[Bug] Qwen3 w8a8 generate garbage output with all MoE backend #10138
Closed
#10144
Closed
[Bug] Qwen3 w8a8 generate garbage output with all MoE backend
#10138
#10144
@rainj-me
Description
rainj-me
opened on Sep 7
Collaborator
Checklist

1. I have searched related issues but cannot get the expected help.

2. The bug has not been fixed in the latest version.

3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.

4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.

5. Please use English, otherwise it will be closed.
Describe the bug
When serving the Qwen3 w8a8 MoE model, none of the MoE backend generate correct result. Before this PR, only the cutlass MoE backend generate correct result.

Reproduction
Without the MoE Refactor PR
Step 1. reset commit to dbb1235
Step 2. serving Qwen3 model with the cutlass MoE backend, which generate the correct result

SGLANG_CUTLASS_MOE=1 SGL_ENABLE_JIT_DEEPGEMM=0 TORCH_CUDA_ARCH_LIST=10.0  CUDA_VISIBLE_DEVICES=2 python3 -m sglang.launch_server --model-path /data00/models/Qwen3-30B-A3B-Instruct-2507-FP8 --trust-remote-code  --port 28000 --mem-fraction-static 0.9  --tp 1 --disable-radix-cache

{"id":"c1026b20f44547ca9b6a9150375dead4","object":"chat.completion","created":1757290123,"model":"/data01/models/Qwen3-30B-A3B-Instruct-2507-FP8","choices":[{"index":0,"message":{"role":"assistant","content":"A large language model (LLM) is a type of artificial intelligence system trained on vast amounts of text data to understand and generate human-like language. These models use deep learning techniques, particularly transformer architectures, to recognize patterns in language and produce coherent, contextually relevant responses. LLMs can perform a wide range of tasks, such as answering questions, writing essays, translating languages, summarizing text, and even coding. Examples include models like GPT, BERT, and Llama. While powerful, they rely on their training data and can sometimes generate inaccurate or biased information, so careful use and evaluation are essential.","reasoning_content":null,"tool_calls":null},"logprobs":null,"finish_reason":"stop","matched_stop":151645}],"usage":{"prompt_tokens":18,"total_tokens":144,"completion_tokens":126,"prompt_tokens_details":null,"reasoning_tokens":0},"metadata":{"weight_version":"default"}}
Step3, serving Qwen3 model with the triton MoE backend, which generate garbage result

 SGLANG_CUTLASS_MOE=0 SGL_ENABLE_JIT_DEEPGEMM=0 TORCH_CUDA_ARCH_LIST=10.0  CUDA_VISIBLE_DEVICES=2 python3 -m sglang.launch_server --model-path /data00/models/Qwen3-30B-A3B-Instruct-2507-FP8 --trust-remote-code  --port 28000 --mem-fraction-static 0.9  --tp 1 --disable-radix-cache --moe-runner-backend triton

{"id":"c604bc905a1c41f9baee663bcfa1c6e8","object":"chat.completion","created":1757290261,"model":"/data01/models/Qwen3-30B-A3B-Instruct-2507-FP8","choices":[{"index":0,"message":{"role":"assistant","content":"\n  1.  2.  3.  4.  5.  6.  7.  8.  9.  10.  11.  12.  13.  14.  15.  16.  17.  18.  19.  20.  21.  22.  23.  24.  25.  26.  27.  28.  29.  30.  31.  32.  33.  34.  35.  36.  37.  38.  39.  40.  41.  4","reasoning_content":null,"tool_calls":null},"logprobs":null,"finish_reason":"length","matched_stop":null}],"usage":{"prompt_tokens":18,"total_tokens":218,"completion_tokens":200,"prompt_tokens_details":null,"reasoning_tokens":0},"metadata":{"weight_version":"default"}}
With the MoE Refactor PR
Step 1. git reset commit to 3fa62da
Step 2. serving Qwen3 model with the cutlass MoE backend, which generate the garbage result

SGLANG_CUTLASS_MOE=1 SGL_ENABLE_JIT_DEEPGEMM=0 TORCH_CUDA_ARCH_LIST=10.0  CUDA_VISIBLE_DEVICES=2 python3 -m sglang.launch_server --model-path /data00/models/Qwen3-30B-A3B-Instruct-2507-FP8 --trust-remote-code  --port 28000 --mem-fraction-static 0.9  --tp 1 --disable-radix-cache

{"id":"d75c83687dc3421cb1f59a4dbcadf6a0","object":"chat.completion","created":1757290413,"model":"/data01/models/Qwen3-30B-A3B-Instruct-2507-FP8","choices":[{"index":0,"message":{"role":"assistant","content":".0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000","reasoning_content":null,"tool_calls":null},"logprobs":null,"finish_reason":"length","matched_stop":null}],"usage":{"prompt_tokens":18,"total_tokens":218,"completion_tokens":200,"prompt_tokens_details":null,"reasoning_tokens":0},"metadata":{"weight_version":"default"}}
Step3, serving Qwen3 model with the triton MoE backend, which generate garbage result

SGLANG_CUTLASS_MOE=0 SGL_ENABLE_JIT_DEEPGEMM=0 TORCH_CUDA_ARCH_LIST=10.0  CUDA_VISIBLE_DEVICES=2 python3 -m sglang.launch_server --model-path /data00/models/Qwen3-30B-A3B-Instruct-2507-FP8 --trust-remote-code  --port 28000 --mem-fraction-static 0.9  --tp 1 --disable-radix-cache --moe-runner-backend triton

{"id":"8c74de5a7de54120ae70ce4556429d43","object":"chat.completion","created":1757290481,"model":"/data01/models/Qwen3-30B-A3B-Instruct-2507-FP8","choices":[{"index":0,"message":{"role":"assistant","content":"\n  1.  2.  3.  4.  5.  6.  7.  8.  9.  10.  11.  12.  13.  14.  15.  16.  17.  18.  19.  20.  21.  22.  23.  24.  25.  26.  27.  28.  29.  30.  31.  32.  33.  34.  35.  36.  37.  38.  39.  40.  41.  4","reasoning_content":null,"tool_calls":null},"logprobs":null,"finish_reason":"length","matched_stop":null}],"usage":{"prompt_tokens":18,"total_tokens":218,"completion_tokens":200,"prompt_tokens_details":null,"reasoning_tokens":0},"metadata":{"weight_version":"default"}}


## Resources (inside the provided docker container)
### Model definitions
- Reference framework model definition path: /workspace/venvs/sglang/lib/python3.10/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py
- Target framework model definition path: /workspace/frameworks/sglang/python/sglang/srt/models/qwen3_moe.py

### Engine source trees
- Reference framework source code root: /workspace/venvs/sglang/lib/python3.10/site-packages/transformers
- Target framework source code root: /workspace/frameworks/sglang/python

### Reproduction (same prompt, greedy decoding, same seed)
- Reference framework:   /workspace/drivers/run_hf_driver.sh --config /workspace/bug-cases/sglang/sglang-10138/hf_config.yaml --tp 4
- Target framework: /workspace/drivers/run_sglang_driver.sh --config /workspace/bug-cases/sglang/sglang-10138/sglang_config.yaml


### Model architecture tree (JSON)
You are given a JSON file that represents the torch model architecture as a **tree**:
- Path: /workspace/bug-cases/sglang/sglang-10138/model_tree_sglang.json

Each node has:
- "type": module class/type name
- "names": one or more attribute names (or indices for ModuleList like "0", "1", ...)
- "children": submodules

## What counts as a “component”
A component is a **torch.nn.Module** that appears in the tree (reachable via `named_modules()` / model graph).

### Full path requirement (STRICT)
You MUST select suspected components ONLY from this tree and return their **FULL module paths**, including the "root" prefix.

### Variable abstraction (recommended when applicable)
If the same suspicion pattern repeats across layers, you MAY use variables to avoid redundancy.

- Use variables like {i}, {j} to represent repeated indices in ModuleList paths (e.g., layers).
- You MUST define each variable with the complete list of values it can take.
- Every templated path MUST expand to valid concrete paths that exist in the tree.

Example:
- template: "root.language_model.model.layers.{i}.self_attn.qkv_proj"
- variable: {i} in [0..27]
This means the component set includes every expanded concrete path for i=0..27.

## What you should do
1. **Reproduce** the mismatch using the scripts above.
2. **Localize** the divergence to a small set of candidate submodules in the target framework.
3. For each candidate, provide **concrete evidence**:
   - the exact **file path** in the target framework,
   - the relevant **line number range(s)** for the suspected logic,
   - why this logic can plausibly cause the observed accuracy issue,
   - (optional but strong) what experiment/log check would falsify it.
   
## What you must not do
1. Search for the original Github issue online.

## Output requirements
- Provide **top-1** suspected components.
- Each component must be either:
  (a) a concrete full module path from the tree, OR
  (b) a templated path that expands (via variables you define) to valid concrete paths in the tree.
- Avoid overly generic blame (e.g., “attention” with no specific mechanism). Point to a **specific class** and **specific code lines**.
        
        Output format:
        ```json
        {
            "variables": Dict[str, List[str]] = Field(description="The variables that you use to represent the repeated indices in model tree paths."),
            "suspected_components": List[str] = Field(description="The component class names that you suspect are the root cause components of the accuracy differences. They must be either a concrete full module path from the tree or a templated path that expands to valid concrete paths in the tree."),
            "reasoning": Dict[str, str] = Field(description="The reasoning for why the components are suspected to be the root cause of the issue, key is the component name in suspected_components and value is the reasoning for why the component is suspected to be the root cause of the issue. You should mention code line numbers in the reasoning for each component.")
        }
        ```
        
        Output the result json in /docker_mount/output_sglang_10138_top1.json in side the docker container