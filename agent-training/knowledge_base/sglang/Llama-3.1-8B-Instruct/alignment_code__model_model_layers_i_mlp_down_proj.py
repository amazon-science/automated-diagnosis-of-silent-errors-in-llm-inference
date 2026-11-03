import torch
from typing import Dict, List

def postprocess_hf_activations() -> Dict[str, Dict[str, List[torch.Tensor]]]:
    ''' Post-process HuggingFace hidden states to match vLLM hidden states.
    
    Analysis:
    - HuggingFace Linear.forward returns F.linear(input, weight, bias), which returns a Tensor
    - This tensor is collected into a tuple by the hook system
    - vLLM output appears to be a nested tuple structure: ((hidden_states, None), (residual, None))
    
    Based on the tensor keys available:
    - hf_model_layers_0_mlp_down_proj_output_0_0: shape (1, 926, 4096)
    - vllm_model_layers_0_mlp_down_proj_output_0_0: shape (926, 4096) - hidden states
    - vllm_model_layers_0_mlp_down_proj_output_1_0: shape (1, 4096) - residual
    
    The alignment checker expects flat tuples of tensors only (no None values).
    We need to:
    1. Squeeze HF output from (1, 926, 4096) to (926, 4096) to match vLLM hidden states
    2. Return flat tuples: (hidden_states_tensor, residual_tensor)
    
    Output:
    - result: A dictionary containing vLLM component names as keys and a dictionary as values with the following keys:
            - 'HF': a tuple of torch.Tensor that has the same length and shape as the output activations in vLLM. But each tensor is the postprocessed HuggingFace output activations to match the output activations in vLLM or copied from the original vLLM output activations if no matching HuggingFace output activations are found.
            - 'vLLM': a tuple of torch.Tensor that is the original vLLM output activations.
    '''
    # === DO NOT CHANGE THE CODE BELOW ===
    # load the activations
    # hf_model_layers_0_mlp_down_proj_output: Tuple of tensors collected from the HF module's output
    hf_model_layers_0_mlp_down_proj_output = torch.load('/path/to/activations/agent-training/Llama-3.1-8B-Instruct/dec2/hf_activation_rank0_model_layers_0_mlp_down_proj_output.pt', weights_only=False)

    # vllm_model_layers_0_mlp_down_proj_output: Tuple of tensors collected from the vLLM module's output
    vllm_model_layers_0_mlp_down_proj_output = torch.load('/path/to/activations/agent-training/Llama-3.1-8B-Instruct/dec2/sglang_activation_rank0_model_layers_0_mlp_down_proj_output.pt', weights_only=False)

    # create the result dict
    result = {"model.layers.0.mlp.down_proj": {"HF": None, "vLLM": None},}
    # === DO NOT CHANGE THE CODE ABOVE ===
    
    # Extract HF hidden states (should be a tensor with shape [1, 926, 4096])
    if isinstance(hf_model_layers_0_mlp_down_proj_output[0], torch.Tensor):
        hf_hidden_states = hf_model_layers_0_mlp_down_proj_output[0]
    else:
        # If nested, extract the actual tensor
        hf_hidden_states = hf_model_layers_0_mlp_down_proj_output[0][0]
    
    # Squeeze batch dimension: (1, 926, 4096) -> (926, 4096)
    hf_hidden_states_squeezed = hf_hidden_states.squeeze(0)
    
    # Extract vLLM outputs - flatten the nested structure to get only tensors
    # vLLM structure: ((hidden_states, None), (residual, None))
    vllm_hidden_states = vllm_model_layers_0_mlp_down_proj_output[0][0]  # shape: (926, 4096)
    vllm_residual = vllm_model_layers_0_mlp_down_proj_output[1][0]  # shape: (1, 4096)
    
    # Create flat tuple of tensors (no None values) for alignment checking
    # Since HF doesn't compute residual, we copy vLLM's residual
    hf_output_flat = (hf_hidden_states_squeezed, vllm_residual)
    vllm_output_flat = (vllm_hidden_states, vllm_residual)
    
    # Store results
    result["model.layers.0.mlp.down_proj"]["HF"] = hf_output_flat
    result["model.layers.0.mlp.down_proj"]["vLLM"] = vllm_output_flat
    
    # === DO NOT CHANGE THE CODE BELOW ===
    return result
    # === DO NOT CHANGE THE CODE ABOVE ===