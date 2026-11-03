#!/usr/bin/env python3
"""
Script to build prompts from a template using configuration files.
Takes a config file, fills in the template, and outputs the final prompt to a file.
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
from common.config import load_config  # expands ${EKKA_ROOT} in config values


def build_prompt(config: Dict[str, Any]) -> str:
    """
    Build the prompt from the template using configuration values.
    
    Expected config keys:
    - problem_description: str
    - reference_framework_model_path: str
    - target_framework_model_path: str
    - reference_framework_code_path: str
    - target_framework_code_path: str
    - reference_framework_reproduction_script: str
    - reference_framework_config_path: str
    - target_framework_reproduction_script: str
    - target_framework_config_path: str
    - model_tree_json_path: str
    - top_k: int (optional, defaults to 3)
    - output_path: str (optional, for specifying where to output results)
    """
    problem_description_path = config.get('problem_description_path', '')
    reference_framework_model_path = config.get('reference_framework_model_path', '')
    target_framework_model_path = config.get('target_framework_model_path', '')
    reference_framework_code_path = config.get('reference_framework_code_path', '/workspace/venvs/vllm/lib/python3.10/site-packages/transformers')
    target_framework_code_path = config.get('target_framework_code_path', '/workspace/frameworks/vllm/vllm')
    reference_framework_reproduction_script = config.get('reference_framework_reproduction_script', '/workspace/drivers/run_hf_driver.sh')
    reference_framework_config_path = config.get('reference_framework_config_path', '')
    target_framework_reproduction_script = config.get('target_framework_reproduction_script', '/workspace/drivers/run_vllm_driver.sh')
    target_framework_config_path = config.get('target_framework_config_path', '')
    model_tree_json_path = config.get('model_tree_json_path', '')
    top_k = config.get('top_k', 3)
    output_path = config.get('output_path', '/docker_mount/output.json')
    tp = config.get('tp', 1)
    
    with open(problem_description_path, 'r') as f:
        problem_description = f.read()
        
    prompt = f"""
You are an expert ML engineer diagnosing **accuracy divergences** between two inference engines running the **same model**.

## Goal
Given a buggy target framework and a correct reference framework, identify which **torch submodule implementation in the target framework** is most likely the root cause of the accuracy differences.

## Issue
{problem_description}

## Resources (inside the provided docker container)
### Model definitions
- Reference framework model definition path: {reference_framework_model_path}
- Target framework model definition path: {target_framework_model_path}

### Engine source trees
- Reference framework source code root: {reference_framework_code_path}
- Target framework source code root: {target_framework_code_path}

### Reproduction (same prompt, greedy decoding, same seed)
- Reference framework:   {reference_framework_reproduction_script} --config {reference_framework_config_path} --tp {tp}
- Target framework: {target_framework_reproduction_script} --config {target_framework_config_path}


### Model architecture tree (JSON)
You are given a JSON file that represents the torch model architecture as a **tree**:
- Path: {model_tree_json_path}

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

- Use variables like {{i}}, {{j}} to represent repeated indices in ModuleList paths (e.g., layers).
- You MUST define each variable with the complete list of values it can take.
- Every templated path MUST expand to valid concrete paths that exist in the tree.

Example:
- template: "root.language_model.model.layers.{{i}}.self_attn.qkv_proj"
- variable: {{i}} in [0..27]
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
- Provide **top-{top_k}** suspected components.
- Each component must be either:
  (a) a concrete full module path from the tree, OR
  (b) a templated path that expands (via variables you define) to valid concrete paths in the tree.
- Avoid overly generic blame (e.g., “attention” with no specific mechanism). Point to a **specific class** and **specific code lines**.
        
        Output format:
        ```json
        {{
            "variables": Dict[str, List[str]] = Field(description="The variables that you use to represent the repeated indices in model tree paths."),
            "suspected_components": List[str] = Field(description="The component class names that you suspect are the root cause components of the accuracy differences. They must be either a concrete full module path from the tree or a templated path that expands to valid concrete paths in the tree."),
            "reasoning": Dict[str, str] = Field(description="The reasoning for why the components are suspected to be the root cause of the issue, key is the component name in suspected_components and value is the reasoning for why the component is suspected to be the root cause of the issue. You should mention code line numbers in the reasoning for each component.")
        }}
        ```
        """
    
    # Add output path instruction if provided
    if output_path:
        prompt += f"\n        Output the result json in {output_path} in side the docker container"
    
    return prompt.strip()


def main():
    parser = argparse.ArgumentParser(
        description='Build prompts from a template using configuration files'
    )
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to the YAML configuration file'
    )
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Path to the output file where the final prompt will be written'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Build prompt
    prompt = build_prompt(config)
    
    # Write output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        f.write(prompt)
    
    print(f"Prompt successfully written to {output_path}")


if __name__ == '__main__':
    main()
