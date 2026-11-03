import torch
from typing import Dict, List
# import any other necessary modules here
    
def postprocess_hf_activations() -> Dict[str, Dict[str, List[torch.Tensor]]]:
    ''' Post-process HuggingFace hidden states to match vLLM hidden states.
    
    The vLLM gate_up_proj module fuses the gate_proj and up_proj Linear layers
    into a single module. The output concatenates gate_proj and up_proj outputs
    along the last dimension.
    
    vLLM output structure is a tuple of tuples where each inner tuple contains
    (tensor, None). We need to match only the tensor parts.
    
    Output:
    - result: A dictionary containing vLLM component names as keys and a dictionary as values with the following keys:
            - 'HF': a tuple of torch.Tensor that has the same length and shape as the output activations in vLLM. But each tensor is the postprocessed HuggingFace output activations to match the output activations in vLLM or copied from the original vLLM output activations if no matching HuggingFace output activations are found.
            - 'vLLM': a tuple of torch.Tensor that is the original vLLM output activations.
    '''
    # === DO NOT CHANGE THE CODE BELOW ===
    # load the activations
    # hf_model_layers_0_mlp_gate_proj_output: Tuple of tensors collected from the HF module's output
    hf_model_layers_0_mlp_gate_proj_output = torch.load('/path/to/activations/agent-training/Llama-3.1-8B-Instruct/dec2/hf_activation_rank0_model_layers_0_mlp_gate_proj_output.pt', weights_only=False)
    # hf_model_layers_0_mlp_up_proj_output: Tuple of tensors collected from the HF module's output
    hf_model_layers_0_mlp_up_proj_output = torch.load('/path/to/activations/agent-training/Llama-3.1-8B-Instruct/dec2/hf_activation_rank0_model_layers_0_mlp_up_proj_output.pt', weights_only=False)

    # vllm_model_layers_0_mlp_gate_up_proj_output: Tuple of tensors collected from the vLLM module's output
    vllm_model_layers_0_mlp_gate_up_proj_output = torch.load('/path/to/activations/agent-training/Llama-3.1-8B-Instruct/dec2/sglang_activation_rank0_model_layers_0_mlp_gate_up_proj_output.pt', weights_only=False)

    # create the result dict
    result = {"model.model.layers.0.mlp.gate_up_proj": {"HF": None, "vLLM": None},}
    # === DO NOT CHANGE THE CODE ABOVE ===
    
    # Extract the actual tensors from the nested tuple structure
    # HuggingFace Linear layer output format: ((tensor,),) based on snapshot notation [0][0]
    hf_gate_tensor = hf_model_layers_0_mlp_gate_proj_output[0][0]  # shape: (1, 926, 3584)
    hf_up_tensor = hf_model_layers_0_mlp_up_proj_output[0][0]      # shape: (1, 926, 3584)
    
    # vLLM output - keep original structure
    vllm_output = vllm_model_layers_0_mlp_gate_up_proj_output
    
    # Process HuggingFace activations to match vLLM:
    # 1. Remove the batch dimension (squeeze from (1, 926, 3584) to (926, 3584))
    # 2. Concatenate gate_proj and up_proj along the last dimension to get (926, 7168)
    hf_gate_squeezed = hf_gate_tensor.squeeze(0)  # (926, 3584)
    hf_up_squeezed = hf_up_tensor.squeeze(0)      # (926, 3584)
    
    # Concatenate to form gate_up_proj output matching vLLM's fused output
    hf_gate_up = torch.cat([hf_gate_squeezed, hf_up_squeezed], dim=-1)  # (926, 7168)
    
    # Extract vLLM tensors from the structure (ignoring None values)
    # vLLM structure: ((tensor_0, None), (tensor_1, None))
    vllm_tensor_0 = vllm_output[0][0]  # shape: (926, 7168)
    vllm_tensor_1 = vllm_output[1][0]  # shape: (1, 7168)
    
    # Create tuples with only tensors (no None values)
    # The HF processed output should match the first vLLM tensor
    # For the second tensor, we use vLLM's original since HF doesn't compute a residual
    hf_processed = (hf_gate_up, vllm_tensor_1)
    vllm_processed = (vllm_tensor_0, vllm_tensor_1)
    
    # Populate the result
    result["model.model.layers.0.mlp.gate_up_proj"]["HF"] = hf_processed
    result["model.model.layers.0.mlp.gate_up_proj"]["vLLM"] = vllm_processed
    
    # === DO NOT CHANGE THE CODE BELOW ===
    return result
    # === DO NOT CHANGE THE CODE ABOVE ===