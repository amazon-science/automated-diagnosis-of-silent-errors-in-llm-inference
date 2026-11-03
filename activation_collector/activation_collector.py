import torch
import os
import pickle
from typing import Any, Dict, List, Callable, Optional, Union, Literal

HookMode = Literal["forward", "forward_pre", "both", "replacement"]


# a hack for SGLang to skip collecting ForwardBatch
def should_skip_collect(obj) -> bool:
    cls = obj.__class__
    # cheapest check: class name
    if cls.__name__ == "ForwardBatch" or cls.__name__ == "ParallelLMHead" or cls.__name__ == "VocabParallelEmbedding" or cls.__name__ == "Gemma3TextScaledWordEmbedding":
        return True
    # extra safety: also check module string
    mod = getattr(cls, "__module__", "")
    if "sglang" in mod.lower() and "forward" in cls.__name__.lower():
        return True
    return False


class ActivationCollector:
    def __init__(self):
        self.save_path = None
        self.save_step = None
        self.collected_activation = {}
        self.rank = None
        self.replacement_activations: Dict[str, Any] = {}
        self.replacement_counter: Dict[str, int] = {}
        self.call_sequence: List[str] = []
        self.framework = None
    
    def create_default_hook(self, module_name: str, mode: HookMode = "forward_pre"):
        """Create a default hook function for a specific module"""
        # def default_forward_hook(module, input, kwargs, output):
        #     save_key = f"{module_name}.output"
        #     if save_key not in self.collected_activation:
        #         self.collected_activation[save_key] = []
                
            
        #     # Save output instead of input for forward hook
        #     if isinstance(output, tuple):
        #         tensor_to_save = output[0]
        #     else:
        #         tensor_to_save = output

        #     print(f"[default_forward_hook] Collecting output from {module_name}: shape {tensor_to_save.shape}")
        #     self.collected_activation[save_key].append(tensor_to_save.detach().clone().cpu())
        #     self.check_if_need_save(save_key)
        #     return output

        # def default_forward_pre_hook(module, input, kwargs):
        #     save_key = f"{module_name}.input"
        #     if save_key not in self.collected_activation:
        #         self.collected_activation[save_key] = []
            
        #     if isinstance(input, tuple):
        #         if len(input) == 0:
        #             print(f"Warning: No input provided to {module_name}. Skipping.")
        #             return
        #         tensor_to_save = input[0]
        #     else:
        #         tensor_to_save = input

        #     print(f"[default_forward_pre_hook] Collecting input from {module_name}: shape {tensor_to_save.shape}")
        #     self.collected_activation[save_key].append(tensor_to_save.detach().clone().cpu())
        #     self.check_if_need_save(save_key)

        def default_forward_hook(module, input, kwargs, output):
            self.call_sequence.append(f"{module_name}")
            save_key = f"{module_name}.output"
            if save_key not in self.collected_activation:
                self.collected_activation[save_key] = []
            
            # Save ALL output data
            if isinstance(output, tuple):
                # Save all elements in the tuple
                output_to_save = []
                for i, elem in enumerate(output):
                    if torch.is_tensor(elem):  # Better tensor check
                        output_to_save.append(elem.detach().clone().cpu())
                    elif should_skip_collect(elem):
                        print(f"[default_forward_hook] Collecting output from {module_name}: ForwardBatch is not pickable, skipping")
                        output_to_save.append(None)
                    else:
                        output_to_save.append(elem)  # Keep non-tensor elements as-is
                output_to_save = tuple(output_to_save)
                print(f"[default_forward_hook] Collecting output from {module_name}: tuple with {len(output)} elements")
            else:
                # Single output
                if torch.is_tensor(output):  # Better tensor check
                    output_to_save = output.detach().clone().cpu()
                    print(f"[default_forward_hook] Collecting output from {module_name}: shape {output_to_save.shape}")
                elif should_skip_collect(output):
                    print(f"[default_forward_hook] Collecting output from {module_name}: ForwardBatch is not pickable, skipping")
                    output_to_save = None
                else:
                    output_to_save = output
                    print(f"[default_forward_hook] Collecting output from {module_name}: non-tensor output")
                    
                output_to_save = (output_to_save,)
                
            self.collected_activation[save_key].append(output_to_save)
            self.check_if_need_save(save_key)
            return output

        def replacement_hook(module, input, kwargs, output):
            safe_name = module_name
            if self.rank is None:
                print(f"[replacement_hook] Rank not set, need to call set_rank")
                return output
            if not self.replacement_activations:
                print(f"[replacement_hook] No replacement activations loaded, need to call load_replacement_activations")
                return output

            if safe_name not in self.replacement_activations:
                print(f"[replacement_hook] No replacement activations found for {module_name}")
                return output

            replacement_data = self.replacement_activations[safe_name][self.replacement_counter[safe_name]]
            self.replacement_counter[safe_name] += 1
            prepared_output = self.prepare_replacement_output(replacement_data, output, module_name)
            print(f"[replacement_hook] Replacing output for {module_name} using key '{safe_name}'")
            print(f"[replacement_hook] Prepared output: {prepared_output}")
            return prepared_output

        def default_forward_pre_hook(module, input, kwargs):
            save_key = f"{module_name}.input"
            if save_key not in self.collected_activation:
                self.collected_activation[save_key] = []
            
            # Process ALL input data
            if isinstance(input, tuple):
                if len(input) == 0:
                    print(f"Warning: No input provided to {module_name}.")
                    input_to_save = ()
                else:
                    # Save all elements in the input tuple
                    input_to_save = []
                    for i, elem in enumerate(input):
                        if torch.is_tensor(elem):  # Better tensor check
                            input_to_save.append(elem.detach().clone().cpu())
                        elif should_skip_collect(elem):
                            print(f"[default_forward_pre_hook] Collecting input from {module_name}: ForwardBatch is not pickable, skipping")
                            input_to_save.append(None)
                        else:
                            input_to_save.append(elem)  # Keep non-tensor elements as-is
                    input_to_save = tuple(input_to_save)
                    print(f"[default_forward_pre_hook] Collecting input from {module_name}: tuple with {len(input)} elements")
            else:
                # Single input
                if torch.is_tensor(input):  # Better tensor check
                    input_to_save = input.detach().clone().cpu()
                    print(f"[default_forward_pre_hook] Collecting input from {module_name}: shape {input_to_save.shape}")
                elif should_skip_collect(input):
                    print(f"[default_forward_pre_hook] Collecting input from {module_name}: ForwardBatch is not pickable, skipping")
                    input_to_save = None
                else:
                    input_to_save = input
                    print(f"[default_forward_pre_hook] Collecting input from {module_name}: non-tensor input")
            
            # Process ALL kwargs data
            kwargs_to_save = {}
            for key, value in kwargs.items():
                if torch.is_tensor(value):  # Better tensor check
                    kwargs_to_save[key] = value.detach().clone().cpu()
                elif isinstance(value, (tuple, list)):
                    # Handle collections that might contain tensors
                    processed_collection = []
                    for item in value:
                        if torch.is_tensor(item):  # Better tensor check
                            processed_collection.append(item.detach().clone().cpu())
                        elif should_skip_collect(item):
                            print(f"[default_forward_pre_hook] Collecting kwargs from {module_name}: ForwardBatch is not pickable, skipping")
                            processed_collection.append(None)
                        else:
                            processed_collection.append(item)
                    kwargs_to_save[key] = type(value)(processed_collection)  # Preserve original type
                elif should_skip_collect(value):
                    print(f"[default_forward_pre_hook] Collecting kwargs from {module_name}: ForwardBatch is not pickable, skipping")
                    kwargs_to_save[key] = None
                else:
                    kwargs_to_save[key] = value  # Keep non-tensor values as-is
                    
            # Save as unified dictionary
            pre_hook_data = {
                "input": input_to_save,
                "kwargs": kwargs_to_save
            }
            
            print(f"[default_forward_pre_hook] Collecting from {module_name} - kwargs keys: {list(kwargs_to_save.keys())}")
            self.collected_activation[save_key].append(pre_hook_data)
            self.check_if_need_save(save_key)

        if mode == "forward":
            return default_forward_hook
        elif mode == "forward_pre":
            return default_forward_pre_hook
        elif mode == "replacement":
            return replacement_hook
        else:  # mode == "both"
            return default_forward_pre_hook, default_forward_hook

    def create_hook_wrapper(self, module_name: str, custom_hook_fn: Union[Callable, List[Optional[Callable]]], mode: HookMode = "forward_pre"):
        """Create a wrapper for custom hook functions that provides access to collector's state"""
        def forward_wrapper(module, input, kwargs, output):
            save_key = f"{module_name}.output"
            if save_key not in self.collected_activation:
                self.collected_activation[save_key] = []
            if isinstance(custom_hook_fn, list):
                # Use the forward hook (second element) from the list
                hook_fn = custom_hook_fn[1]
                if hook_fn is None:
                    # Use default forward hook if None
                    return self.create_default_hook(module_name, "forward")(module, input, kwargs, output)
                return hook_fn(self, module_name, module, input, kwargs, output)
            return custom_hook_fn(self, module_name, module, input, kwargs, output)

        def forward_pre_wrapper(module, input, kwargs):
            save_key = f"{module_name}.input"
            if save_key not in self.collected_activation:
                self.collected_activation[save_key] = []
            if isinstance(custom_hook_fn, list):
                # Use the pre-forward hook (first element) from the list
                hook_fn = custom_hook_fn[0]
                if hook_fn is None:
                    # Use default pre-forward hook if None
                    return self.create_default_hook(module_name, "forward_pre")(module, input, kwargs)
                return hook_fn(self, module_name, module, input, kwargs)
            return custom_hook_fn(self, module_name, module, input, kwargs)

        if mode == "both":
            # For "both" mode, custom_hook_fn should be a list of two elements
            if not isinstance(custom_hook_fn, list) or len(custom_hook_fn) != 2:
                raise ValueError("For 'both' mode, custom_hook_fn must be a list of two elements")
            
            # Create wrappers that handle both custom and default hooks
            pre_hook_fn = lambda m, i, k: forward_pre_wrapper(m, i, k)
            forward_hook_fn = lambda m, i, k, o: forward_wrapper(m, i, k, o)
            
            return pre_hook_fn, forward_hook_fn
        elif mode == "forward":
            return forward_wrapper
        else:  # mode == "forward_pre"
            return forward_pre_wrapper
    
    def register_hooks(self, 
                      model: torch.nn.Module,
                      target_modules: List[str],
                      hook_fns: Optional[List[Optional[Callable]]] = None,
                      hook_modes: Optional[List[Optional[HookMode]]] = None,
                      save_path: Optional[str] = None,
                      save_step: Optional[int] = None,
                      framework: Optional[str] = "vllm"
                      ) -> List[torch.utils.hooks.RemovableHandle]:
        """
        Register hooks for specified modules with custom hook functions.
        
        Args:
            model: The PyTorch model
            framework: The framework of the model, "vllm" or "hf"
            save_path: Path to save collected activation
            target_modules: List of module names to hook into
            hook_fns: List of custom hook functions corresponding to target_modules
            hook_modes: List of hook modes ("forward", "forward_pre", or "both") for each module.
                       If None, defaults to "forward_pre" for all modules
        """
        if not target_modules:
            return None

        if hook_fns is not None and len(hook_fns) != len(target_modules):
            raise ValueError("Length of hook_fns must match length of target_modules")

        if hook_modes is not None and len(hook_modes) != len(target_modules):
            raise ValueError("Length of hook_modes must match length of target_modules")

        if hook_modes is None:
            hook_modes = ["forward_pre"] * len(target_modules)

        hooks = []

        self.save_path = save_path
        self.save_step = save_step
        self.framework = framework
        
        def register_recursive(module, prefix=''):
            for name, child in module.named_children():
                full_name = f"{prefix}.{name}" if prefix else name
                print(f"checking module: {full_name} of type {type(child)}")
                
                if full_name in target_modules:
                    idx = target_modules.index(full_name)
                    mode = hook_modes[idx] or "forward_pre"
                    print(f"Registering {mode} hook for {full_name}")
                    
                    if hook_fns is not None and hook_fns[idx] is not None:
                        hook_fn = self.create_hook_wrapper(full_name, hook_fns[idx], mode)
                    else:
                        hook_fn = self.create_default_hook(full_name, mode)
                    
                    if mode == "both":
                        pre_hook_fn, forward_hook_fn = hook_fn
                        pre_hook = child.register_forward_pre_hook(pre_hook_fn, with_kwargs=True)
                        forward_hook = child.register_forward_hook(forward_hook_fn, with_kwargs=True)
                        hooks.extend([pre_hook, forward_hook])
                    elif mode == "forward" or mode == "replacement":
                        hook = child.register_forward_hook(hook_fn, with_kwargs=True)
                        hooks.append(hook)
                    else:  # forward_pre
                        hook = child.register_forward_pre_hook(hook_fn, with_kwargs=True)
                        hooks.append(hook)
                
                register_recursive(child, full_name)
        
        register_recursive(model)
        return hooks

    def save_all(self):
        """Save all collected activation"""
        assert self.save_path is not None, "save_path must be set to save activation"
        if not self.collected_activation:
            print("No activation collected to save.")
            return
        
        parent_path = os.path.dirname(self.save_path)
        call_seq_path = os.path.join(parent_path, f"call_sequence_{self.framework}")
        os.makedirs(call_seq_path, exist_ok=True)
        with open(os.path.join(call_seq_path, f"call_sequence_rank{self.rank}.log"), "w") as f:
            for call in self.call_sequence:
                f.write(f"{call}\n")

        for module_name, activation_list in self.collected_activation.items():
            self.save_module_activation(module_name, self.save_path)

    def save_module_activation(self, module_name: str, base_filename: str):
        """Save activation for a specific module"""
        if module_name not in self.collected_activation:
            print(f"No activation collected for module {module_name}.")
            return
        
        activation_list = self.collected_activation[module_name]
        # first two are warmups
        # activation_list = activation_list[2:]

        safe_name = module_name.replace('.', '_').replace('/', '_')
        filename = f"{base_filename}_rank{self.rank}_{safe_name}.pt"
        print(f"Saving activation from {module_name} to {filename}")
        if "logits_processor" in module_name:
            print(f"activation_list: {activation_list}")
        try:
            torch.save(activation_list, filename)
        except Exception as e:
            print(f"Error saving activation from {module_name} to {filename}: {e}")
            print(f"activation_list: {activation_list}")
            raise e
        print(f"Saved activation from {module_name} to {filename}")

    def check_if_need_save(self, module_name: str) -> bool:
        """Check if activation for a specific module need to be saved based on save_step"""
        # if module_name not in self.collected_activation or self.rank != 0:
        if module_name not in self.collected_activation:
            return
        
        if self.save_path is not None and self.save_step is not None:
            if len(self.collected_activation[module_name]) % self.save_step == 0:
                print(f"Saving activation for {module_name} at step {len(self.collected_activation[module_name])}")
                self.save_module_activation(module_name, self.save_path)
                
    
    def clear(self):
        """Clear all collected activation"""
        self.collected_activation = {}
        self.idx = 0

    def set_rank(self, rank: int):
        """Set the rank for distributed serving"""
        self.rank = rank
        
    def get_rank(self) -> int:
        """Get the rank for distributed serving"""
        return self.rank

    def load_replacement_activations(self, replacement_path: Optional[str]):
        """Load activation replacements from disk."""
        self.replacement_activations = {}
        self.replacement_path = replacement_path
        if replacement_path is None:
            return

        self.replacement_activations = torch.load(replacement_path, map_location="cpu")
        self.replacement_counter = {name: 0 for name in self.replacement_activations.keys()}
        print(f"[ActivationCollector] Loaded replacement activations from {replacement_path}")
        print(f"[ActivationCollector] Replacement activations keys: {list(self.replacement_activations.keys())}")

    def prepare_replacement_output(self, replacement_data: Any, reference_output: Any, module_name: str):
        """Move replacement tensors to the appropriate device/dtype and mirror the reference structure."""
        return self.match_structure_to_reference(replacement_data, reference_output, module_name)

    def match_structure_to_reference(self, replacement: Any, reference: Any, module_name: str):
        if torch.is_tensor(replacement):
            if torch.is_tensor(reference):
                result = replacement.to(device=reference.device, dtype=reference.dtype)
                return result
            return replacement

        if isinstance(replacement, tuple):
            if isinstance(reference, tuple):
                if len(replacement) != len(reference):
                    raise ValueError(
                        f"Replacement tuple length mismatch for module '{module_name}': "
                        f"{len(replacement)} != {len(reference)}"
                    )
                return tuple(
                    self.match_structure_to_reference(r_item, ref_item, module_name)
                    for r_item, ref_item in zip(replacement, reference)
                )
            return tuple(
                self.match_structure_to_reference(r_item, None, module_name)
                for r_item in replacement
            )

        if isinstance(replacement, list):
            if isinstance(reference, list):
                if len(replacement) != len(reference):
                    raise ValueError(
                        f"Replacement list length mismatch for module '{module_name}': "
                        f"{len(replacement)} != {len(reference)}"
                    )
                return [
                    self.match_structure_to_reference(r_item, ref_item, module_name)
                    for r_item, ref_item in zip(replacement, reference)
                ]
            return [
                self.match_structure_to_reference(r_item, None, module_name)
                for r_item in replacement
            ]

        if isinstance(replacement, dict):
            result = {}
            if isinstance(reference, dict):
                for key, value in replacement.items():
                    ref_value = reference.get(key) if reference is not None else None
                    result[key] = self.match_structure_to_reference(value, ref_value, module_name)
            else:
                for key, value in replacement.items():
                    result[key] = self.match_structure_to_reference(value, None, module_name)
            return result

        return replacement

activation_collector = ActivationCollector()
