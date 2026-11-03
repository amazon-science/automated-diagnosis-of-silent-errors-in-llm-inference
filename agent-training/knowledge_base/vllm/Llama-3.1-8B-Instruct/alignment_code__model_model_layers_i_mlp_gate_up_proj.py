import torch
from typing import Dict, List
# import any other necessary modules here
    
def postprocess_hf_activations() -> Dict[str, Dict[str, List[torch.Tensor]]]:
    ''' Post-process HuggingFace hidden states to match vLLM hidden states.
    
    The vLLM gate_up_proj is a fused version of HuggingFace's gate_proj and up_proj.
    We need to concatenate the gate_proj and up_proj outputs along the last dimension
    and remove the batch dimension to match vLLM's output shape.
    
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
    vllm_model_layers_0_mlp_gate_up_proj_output = torch.load('/path/to/activations/agent-training/Llama-3.1-8B-Instruct/dec2/vllm_activation_rank0_model_layers_0_mlp_gate_up_proj_output.pt', weights_only=False)

    # create the result dict
    result = {"model.model.layers.0.mlp.gate_up_proj": {"HF": None, "vLLM": None},}
    # === DO NOT CHANGE THE CODE ABOVE ===
    
    # Extract tensors from the tuple structure
    # Linear layer forward returns a tensor, which gets wrapped in a tuple: (tensor,)
    # Navigate through the tuple to extract the actual tensor
    
    # Unwrap gate_proj output
    gate_proj_unwrapped = hf_model_layers_0_mlp_gate_proj_output
    while isinstance(gate_proj_unwrapped, (tuple, list)) and len(gate_proj_unwrapped) > 0:
        if isinstance(gate_proj_unwrapped[0], torch.Tensor):
            gate_proj_out = gate_proj_unwrapped[0]
            break
        gate_proj_unwrapped = gate_proj_unwrapped[0]
    
    # Unwrap up_proj output  
    up_proj_unwrapped = hf_model_layers_0_mlp_up_proj_output
    while isinstance(up_proj_unwrapped, (tuple, list)) and len(up_proj_unwrapped) > 0:
        if isinstance(up_proj_unwrapped[0], torch.Tensor):
            up_proj_out = up_proj_unwrapped[0]
            break
        up_proj_unwrapped = up_proj_unwrapped[0]
    
    # Concatenate gate_proj and up_proj along the last dimension
    # This matches vLLM's fused gate_up_proj output
    fused_output = torch.cat([gate_proj_out, up_proj_out], dim=-1)  # shape: (1, 926, 7168)
    
    # Remove the batch dimension to match vLLM's shape
    fused_output = fused_output.squeeze(0)  # shape: (926, 7168)
    
    # Unwrap vLLM output
    vllm_unwrapped = vllm_model_layers_0_mlp_gate_up_proj_output
    while isinstance(vllm_unwrapped, (tuple, list)) and len(vllm_unwrapped) > 0:
        if isinstance(vllm_unwrapped[0], torch.Tensor):
            vllm_tensor = vllm_unwrapped[0]
            break
        vllm_unwrapped = vllm_unwrapped[0]
    
    # Package as tuple to match vLLM output format
    hf_processed = (fused_output,)
    vllm_output = (vllm_tensor,)
    
    # Populate result
    result["model.model.layers.0.mlp.gate_up_proj"]["HF"] = hf_processed
    result["model.model.layers.0.mlp.gate_up_proj"]["vLLM"] = vllm_output
    
    # === DO NOT CHANGE THE CODE BELOW ===
    return result
    # === DO NOT CHANGE THE CODE ABOVE ===