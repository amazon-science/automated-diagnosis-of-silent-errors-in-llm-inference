import torch
from typing import Dict, List

def postprocess_hf_activations() -> Dict[str, Dict[str, List[torch.Tensor]]]:
    ''' Post-process HuggingFace hidden states to match vLLM hidden states.
    
    The main transformation is to remove the batch dimension (first dimension) from 
    HuggingFace MLP output to match vLLM's format which doesn't include the batch dimension.
    
    Reasoning:
    - HuggingFace MLP outputs contain a tensor of shape (1, 1051, 3584) with batch dimension
    - vLLM MLP outputs contain a tensor of shape (1051, 3584) without batch dimension
    - Both represent the same MLP output activations
    - We need to squeeze dimension 0 from the HF tensor to align with vLLM
    - The loaded activations are saved outputs from module hooks which return tuples
    
    Output:
    - result: A dictionary containing vLLM component names as keys and a dictionary as values with the following keys:
            - 'HF': a tuple of torch.Tensor that has the same length and shape as the output activations in vLLM. But each tensor is the postprocessed HuggingFace output activations to match the output activations in vLLM or copied from the original vLLM output activations if no matching HuggingFace output activations are found.
            - 'vLLM': a tuple of torch.Tensor that is the original vLLM output activations.
    '''
    # === DO NOT CHANGE THE CODE BELOW ===
    # load the activations
    # hf_model_layers_0_mlp_output: Tuple of tensors collected from the HF module's output
    hf_model_layers_0_mlp_output = torch.load('/path/to/activations/agent-training/Qwen2.5-7B-Instruct/dec2/hf_activation_rank0_model_layers_0_mlp_output.pt', weights_only=False)

    # vllm_model_layers_0_mlp_output: Tuple of tensors collected from the vLLM module's output
    vllm_model_layers_0_mlp_output = torch.load('/path/to/activations/agent-training/Qwen2.5-7B-Instruct/dec2/vllm_activation_rank0_model_layers_0_mlp_output.pt', weights_only=False)

    # create the result dict
    result = {"model.model.layers.0.mlp": {"HF": None, "vLLM": None},}
    # === DO NOT CHANGE THE CODE ABOVE ===
    
    # Helper function to find the first tensor in a nested structure
    def extract_tensor(obj):
        """Recursively extract the first tensor from nested tuples/lists"""
        if isinstance(obj, torch.Tensor):
            return obj
        elif isinstance(obj, (tuple, list)) and len(obj) > 0:
            return extract_tensor(obj[0])
        else:
            return obj
    
    # Extract the actual tensors from the loaded data
    hf_tensor = extract_tensor(hf_model_layers_0_mlp_output)
    vllm_tensor = extract_tensor(vllm_model_layers_0_mlp_output)
    
    # Process HF tensor: squeeze the batch dimension
    if isinstance(hf_tensor, torch.Tensor) and hf_tensor.dim() >= 3:
        # HF shape is (1, 1051, 3584), squeeze first dimension to get (1051, 3584)
        if hf_tensor.shape[0] == 1:
            hf_tensor_processed = hf_tensor.squeeze(0)
        else:
            hf_tensor_processed = hf_tensor
    else:
        hf_tensor_processed = hf_tensor
    
    # Wrap both tensors in tuples to match the expected output format
    hf_processed = (hf_tensor_processed,)
    
    # For vLLM, check if it's already a tuple, otherwise wrap it
    if isinstance(vllm_model_layers_0_mlp_output, tuple):
        vllm_processed = vllm_model_layers_0_mlp_output
    else:
        vllm_processed = (vllm_tensor,)
    
    # Store the results
    result["model.model.layers.0.mlp"]["HF"] = hf_processed
    result["model.model.layers.0.mlp"]["vLLM"] = vllm_processed
    
    # === DO NOT CHANGE THE CODE BELOW ===
    return result
    # === DO NOT CHANGE THE CODE ABOVE ===