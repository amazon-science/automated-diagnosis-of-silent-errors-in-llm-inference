You are an expert ML engineer diagnosing **accuracy divergences** between two inference engines running the **same model**.

## Goal
Given a buggy target framework and a correct reference framework, identify which **torch submodule implementation in the target framework** is most likely the root cause of the accuracy differences.

## Issue
[Bug] qwen 2 accuracy drop #10344
Closed
Closed
[Bug] qwen 2 accuracy drop
#10344
@zhyncs
Description
zhyncs
opened on Sep 11, 2025
Collaborator
Checklist

1. I have searched related issues but cannot get the expected help.

2. The bug has not been fixed in the latest version.

3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.

4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.

5. Please use English, otherwise it will be closed.
Describe the bug
https://github.com/sgl-project/sglang/actions/runs/17659574671

Qwen/Qwen2-57B-A14B-Instruct	0.58	0.86
Reproduction
N/A

Environment
N/A

## Resources (inside the provided docker container)
### Model definitions
- Reference framework model definition path: /workspace/frameworks/vllm/vllm/model_executor/models/qwen2.py
- Target framework model definition path: /workspace/frameworks/sglang/python/sglang/srt/models/qwen2.py

### Engine source trees
- Reference framework source code root: /workspace/frameworks/vllm/vllm
- Target framework source code root: /workspace/frameworks/sglang/python

### Reproduction (same prompt, greedy decoding, same seed)
- Reference framework:   /workspace/drivers/run_vllm_driver.sh --config /workspace/bug-cases/sglang/sglang-10344/vllm_config.yaml --tp 4
- Target framework: /workspace/drivers/run_sglang_driver.sh --config /workspace/bug-cases/sglang/sglang-10344/sglang_config.yaml


### Model architecture tree (JSON)
You are given a JSON file that represents the torch model architecture as a **tree**:
- Path: /workspace/bug-cases/sglang/sglang-10344/model_tree_sglang.json

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
- Provide **top-3** suspected components.
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
        
        Output the result json in /docker_mount/output_sglang_10344_top3.json in side the docker container