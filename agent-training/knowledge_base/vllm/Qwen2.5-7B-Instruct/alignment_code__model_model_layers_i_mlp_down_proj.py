import torch
from typing import Dict, List
# import any other necessary modules here
    
def postprocess_hf_activations() -> Dict[str, Dict[str, List[torch.Tensor]]]:
    ''' Post-process HuggingFace hidden states to match vLLM hidden states.
    
    Reasoning:
    1. Both HuggingFace and vLLM use a standard Linear layer for down_proj, which returns F.linear(input, weight, bias)
    2. The loaded activations are tuples/lists:
       - HF: contains a tensor with shape [1, 1051, 3584] (with batch dimension)
       - vLLM: contains a tensor with shape [1051, 3584] (without batch dimension) and None as second element
    3. HF tensor has an extra batch dimension at position 0 that needs to be squeezed
    4. vLLM output structure is (tensor, None) based on the snapshots
    5. To align: extract tensor from HF, squeeze batch dimension, return as (tensor, None) tuple
    
    Output:
    - result: A dictionary containing vLLM component names as keys and a dictionary as values with the following keys:
            - 'HF': a tuple that matches the structure and shapes of vLLM output activations
            - 'vLLM': the original vLLM output activations as a tuple
    '''
    # === DO NOT CHANGE THE CODE BELOW ===
    # load the activations
    # hf_model_layers_0_mlp_down_proj_output: Tuple of tensors collected from the HF module's output
    hf_model_layers_0_mlp_down_proj_output = torch.load('/path/to/activations/agent-training/Qwen2.5-7B-Instruct/dec2/hf_activation_rank0_model_layers_0_mlp_down_proj_output.pt', weights_only=False)

    # vllm_model_layers_0_mlp_down_proj_output: Tuple of tensors collected from the vLLM module's output
    vllm_model_layers_0_mlp_down_proj_output = torch.load('/path/to/activations/agent-training/Qwen2.5-7B-Instruct/dec2/vllm_activation_rank0_model_layers_0_mlp_down_proj_output.pt', weights_only=False)

    # create the result dict
    result = {"model.layers.0.mlp.down_proj": {"HF": None, "vLLM": None},}
    # === DO NOT CHANGE THE CODE ABOVE ===
    
    # The loaded data is a list/tuple structure
    # Extract the first element which should be the tensor
    # Handle nested structures by iterating until we get to the actual tensor
    hf_data = hf_model_layers_0_mlp_down_proj_output
    
    # Navigate to the actual tensor - it's in a list/tuple structure
    # Based on the key naming (output_0_0), the tensor is at [0][0] or similar
    if isinstance(hf_data, (list, tuple)) and len(hf_data) > 0:
        hf_element = hf_data[0]
        # Check if this is another nested structure
        if isinstance(hf_element, (list, tuple)) and len(hf_element) > 0:
            hf_tensor = hf_element[0]
        else:
            hf_tensor = hf_element
    else:
        hf_tensor = hf_data
    
    # Squeeze the batch dimension: [1, 1051, 3584] -> [1051, 3584]
    if isinstance(hf_tensor, torch.Tensor) and hf_tensor.ndim == 3 and hf_tensor.shape[0] == 1:
        hf_tensor_squeezed = hf_tensor.squeeze(0)
    else:
        hf_tensor_squeezed = hf_tensor
    
    # vLLM data structure - extract similarly
    vllm_data = vllm_model_layers_0_mlp_down_proj_output
    
    # Navigate to get the tensor and potentially the None
    if isinstance(vllm_data, (list, tuple)) and len(vllm_data) > 0:
        vllm_first = vllm_data[0]
        # Check if this is another nested structure
        if isinstance(vllm_first, (list, tuple)) and len(vllm_first) > 0:
            vllm_tensor = vllm_first[0]
            vllm_second = vllm_first[1] if len(vllm_first) > 1 else None
        else:
            vllm_tensor = vllm_first
            vllm_second = vllm_data[1] if len(vllm_data) > 1 else None
    else:
        vllm_tensor = vllm_data
        vllm_second = None
    
    # Create tuples matching vLLM's output structure: (tensor, None)
    result["model.layers.0.mlp.down_proj"]["HF"] = (hf_tensor_squeezed, None)
    result["model.layers.0.mlp.down_proj"]["vLLM"] = (vllm_tensor, vllm_second)
    
    # === DO NOT CHANGE THE CODE BELOW ===
    return result
    # === DO NOT CHANGE THE CODE ABOVE ===