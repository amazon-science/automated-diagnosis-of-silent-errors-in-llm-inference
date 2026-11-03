"""Helpers for locating activation files on disk, loading them into the runtime store, and gathering module source used in prompts."""

from typing import List, Dict, Any, Optional
import os
import re
from glob import glob
import torch
import torch.distributed.tensor
import logging

from my_agent.output_structure import ComponentMappingOutput
from my_agent.scripts.extract_forward_from_index import extract_forward_from_index
from my_agent.diagnosis.runtime import register_tensor


class ActivationIOMixin:
    """Locate, load, register and summarise activation files and module source snippets."""

    def expand_component_name(self, component: str, component_mappings: ComponentMappingOutput) -> str:
        def replace_variable(match: re.Match) -> str:
            var = match.group(1)
            values = component_mappings.variables.get(var)
            if not values:
                raise ValueError(f"Variable {var} not found for component {component}")
            # we always use the first value for now
            return values[0]

        return re.sub(r"\{([^{}]+)\}", replace_variable, component)

    def sanitize_component_name(self, component: str) -> str:
        return re.sub(r"[^0-9A-Za-z]+", "_", component).strip("_")

    def match_activation_file(self, activation_dir: str, prefix: str, component_stub: str, kind: str) -> Optional[str]:
        if not activation_dir or not os.path.isdir(activation_dir):
            raise ValueError(f"Activation directory {activation_dir} does not exist")
        candidates = [
            # os.path.join(activation_dir, f"{prefix}_*_{component_stub}_{kind}_processed.pt"),
            # os.path.join(activation_dir, f"{prefix}_*_{component_stub}_{kind}.pt"),
            
            # do exact name match for now otherwise we may mess up rank ids
            os.path.join(activation_dir, f"{prefix}_{component_stub}_{kind}.pt"),
        ]
        for pattern in candidates:
            matches = sorted(glob(pattern))
            if matches:
                return matches[0]
        raise ValueError(f"No activation file found for {prefix}_{component_stub}_{kind}.pt in {activation_dir}")

    def resolve_activation_files(
        self,
        components: List[str],
        activation_dir: str,
        prefix: str,
    ) -> Dict[str, Dict[str, Optional[str]]]:
        resolved: Dict[str, Dict[str, Optional[str]]] = {}
        for component in components:
            # take out first model. in the component name
            component = component.split(".")[1:]
            if not component:
                continue
            component = ".".join(component)
            component_stub = component.replace(".", "_").replace("/", "_")
            resolved[component] = {
                # "input": self.match_activation_file(activation_dir, prefix, component_stub, "input"),
                "output": self.match_activation_file(activation_dir, prefix, component_stub, "output"),
            }
        return resolved

    # def tensor_preview(self, tensor: torch.Tensor, max_values: int = 5) -> str:
    #     try:
    #         if hasattr(tensor, "to_local"):
    #             tensor = tensor.to_local()
    #         flat = tensor.detach().cpu().reshape(-1)
    #     except Exception:
    #         try:
    #             flat = tensor.reshape(-1)
    #         except Exception:
    #             return "unavailable"

    #     if flat.numel() == 0:
    #         return "[]"
    #     values = flat[:max_values].tolist()
    #     try:
    #         formatted = ", ".join(f"{float(v):.4g}" for v in values)
    #     except Exception:
    #         formatted = ", ".join(str(v) for v in values)
    #     suffix = ", ..." if flat.numel() > max_values else ""
    #     return f"[{formatted}{suffix}]"
    
    def tensor_preview(self, t: torch.Tensor, n: int = 3):
        """
        Print a quick preview of a tensor:
        - shape, dtype
        - min, max, mean, std
        - first n elements per dimension
        """
        try:
            preview = ""
            # print(f"min: {t.min().item():.4f}, max: {t.max().item():.4f}, "
            #       f"mean: {t.mean().item():.4f}, std: {t.std().item():.4f}")
            
            # Build slice like [:n, :n, ...] automatically
            index = tuple(slice(0, min(n, s)) for s in t.shape)
            preview += str(t[index])
        except Exception as e:
            preview = f"Error previewing tensor: {e}"
        return preview

    def summarize_activation_payload(self, payload: Any, label: str, max_items: int = 10, max_depth: int = 3) -> str:
        lines: List[str] = []

        def _summarize(obj: Any, path: str, depth: int) -> None:
            if depth > max_depth:
                lines.append(f"{path}: ...")
                return

            if isinstance(obj, torch.Tensor):
                lines.append(
                    f"{path}: shape={tuple(obj.shape)}, dtype={obj.dtype}, sample={self.tensor_preview(obj)}"
                )
                return

            if isinstance(obj, dict):
                items = list(obj.items())
                for idx, (key, value) in enumerate(items[:max_items]):
                    _summarize(value, f"{path}['{key}']", depth + 1)
                if len(items) > max_items:
                    lines.append(f"{path}['...']: ({len(items) - max_items} more keys)")
                return

            if isinstance(obj, (list, tuple)):
                for idx, item in enumerate(obj[:max_items]):
                    _summarize(item, f"{path}[{idx}]", depth + 1)
                if len(obj) > max_items:
                    lines.append(f"{path}[...]: ({len(obj) - max_items} more)")
                return

            lines.append(f"{path}: {type(obj).__name__}")

        _summarize(payload, label, 0)
        return "\n".join(lines)

    def register_loaded_data(
        self,
        framework_prefix: str,
        component_stub: str,
        kind: str,
        payload: Any,
        counters: Dict[str, int],
    ) -> None:
        base_prefix = f"{framework_prefix}_{'inputs' if kind == 'input' else 'outputs'}"
        component_prefix = f"{framework_prefix}_{component_stub}_{kind}"
        index = counters.setdefault(kind, 0)

        def _register_tensor(key_suffix: str, tensor: torch.Tensor) -> None:
            register_tensor(f"{base_prefix}_{key_suffix}", tensor)
            register_tensor(f"{component_prefix}_{key_suffix}", tensor)

        if isinstance(payload, list):
            for item in payload:
                local_index = counters[kind]
                if isinstance(item, dict):
                    inputs = item.get("input", [])
                    for input_idx, tensor in enumerate(inputs):
                        if torch.is_tensor(tensor):
                            _register_tensor(f"{local_index}_input_{input_idx}", tensor)
                    kwargs = item.get("kwargs", {})
                    for key, value in kwargs.items():
                        if torch.is_tensor(value):
                            key_stub = self.sanitize_component_name(str(key))
                            _register_tensor(f"{local_index}_kwargs_{key_stub}", value)
                elif isinstance(item, (list, tuple)):
                    for output_idx, tensor in enumerate(item):
                        if torch.is_tensor(tensor):
                            _register_tensor(f"{local_index}_{output_idx}", tensor)
                elif torch.is_tensor(item):
                    _register_tensor(f"{local_index}", item)
                counters[kind] += 1
        elif isinstance(payload, dict):
            for key, value in payload.items():
                if torch.is_tensor(value):
                    key_stub = self.sanitize_component_name(str(key))
                    _register_tensor(f"{index}_kwargs_{key_stub}", value)
            counters[kind] += 1
        elif torch.is_tensor(payload):
            _register_tensor(f"{index}", payload)
            counters[kind] += 1

    def load_and_register_activation_set(
        self,
        files_map: Dict[str, Dict[str, Optional[str]]],
        framework_prefix: str,
    ) -> str:
        previews: List[str] = []
        counters: Dict[str, int] = {"input": 0, "output": 0}

        for component, paths in files_map.items():
            component_stub = self.sanitize_component_name(component) or framework_prefix
            for kind, path in paths.items():
                if not path:
                    previews.append(f"{framework_prefix} {component} {kind}: missing activation file")
                    continue
                try:
                    payload = torch.load(path, weights_only=False)
                except Exception as exc:
                    previews.append(f"{framework_prefix} {component} {kind}: failed to load ({exc})")
                    continue

                self.register_loaded_data(framework_prefix, component_stub, kind, payload, counters)
                if isinstance(payload, list):
                    # payload = payload[0:2] if len(payload) > 2 else payload
                    # payload = payload[0:1] if len(payload) > 2 else payload
                    if len(payload) > 2:
                        logging.warning(f"[WARNING][load_and_register_activation_set] {framework_prefix} {component} {kind}: Found {len(payload)} tensors in the activation file. Only the first 2 tensors will be used.")
                        payload = payload[0:2]
                summary = self.summarize_activation_payload(
                    payload,
                    label=f"{framework_prefix}.{component}.{kind}",
                )
                previews.append(f"{framework_prefix} {component} {kind} path: {path}\n{summary}")

        return "\n".join(previews).strip()

    def gather_forward_snippets(
        self,
        class_names: List[str],
        index_dir: Optional[str],
    ) -> Dict[str, str]:
        snippets: Dict[str, str] = {}
        if not class_names or not index_dir or not os.path.isdir(index_dir):
            return snippets

        try:
            extraction = extract_forward_from_index(index_dir, class_names)
        except Exception as exc:
            logging.warning(f"Failed to extract forward definitions from {index_dir}: {exc}")
            return snippets

        for cls, meta in extraction.items():
            source = meta.get("source")
            if source:
                snippets[cls] = source.strip()
        return snippets

    def format_forward_context(
        self,
        hf_forward: Dict[str, str],
        vllm_forward: Dict[str, str],
    ) -> str:
        blocks: List[str] = []
        for cls, source in hf_forward.items():
            blocks.append(f"HuggingFace {cls}:\n{source}")
        for cls, source in vllm_forward.items():
            blocks.append(f"vLLM {cls}:\n{source}")
        if not blocks:
            return "(forward definitions unavailable)"
        return "\n\n".join(blocks)

    def get_function_signature(self, hf_files: Dict[str, Dict[str, Optional[str]]], vllm_files: Dict[str, Dict[str, Optional[str]]]) -> str:
        hf_input_loading_code = ""
        for hf_component, hf_path_dict in hf_files.items():
            hf_component = self.sanitize_component_name(hf_component)
            for kind, hf_path in hf_path_dict.items():
                if hf_path:
                    hf_input_loading_code += f"    # hf_{hf_component}_{kind}: Tuple of tensors collected from the HF module's output\n    hf_{hf_component}_{kind} = torch.load({hf_path}, weights_only=False)\n"
        vllm_input_loading_code = ""
        for vllm_component, vllm_path_dict in vllm_files.items():
            vllm_component = self.sanitize_component_name(vllm_component)
            for kind, vllm_path in vllm_path_dict.items():
                if vllm_path:
                    vllm_input_loading_code += f"    # vllm_{vllm_component}_{kind}: Tuple of tensors collected from the vLLM module's output\n    vllm_{vllm_component}_{kind} = torch.load({vllm_path}, weights_only=False)\n"
        output_code = "result = {"
        for vllm_component, vllm_path_dict in vllm_files.items():
            output_code += f"\"{vllm_component}\": {{"
            output_code += f"\"HF\": None, \"vLLM\": None"
            output_code += "},"
        output_code += "}"
        return f"""
```python
import torch
from typing import Dict, List
# import any other necessary modules here
    
def postprocess_hf_activations() -> Dict[str, Dict[str, List[torch.Tensor]]]:
''' Post-process HuggingFace hidden states to match vLLM hidden states.
    Output:
    - result: A dictionary containing vLLM component names as keys and a dictionary as values with the following keys:
            - 'HF': a tuple of torch.Tensor that has the same length and shape as the output activations in vLLM. But each tensor is the postprocessed HuggingFace output activations to match the output activations in vLLM or copied from the original vLLM output activations if no matching HuggingFace output activations are found.
            - 'vLLM': a tuple of torch.Tensor that is the original vLLM output activations.
'''
    # === DO NOT CHANGE THE CODE BELOW ===
    # load the activations
{hf_input_loading_code}
{vllm_input_loading_code}
    # create the result dict
    {output_code}
    # === DO NOT CHANGE THE CODE ABOVE ===
    
    # TODO: implement the function
    ...
    
    # === DO NOT CHANGE THE CODE BELOW ===
    return result
    # === DO NOT CHANGE THE CODE ABOVE ===
```
"""
