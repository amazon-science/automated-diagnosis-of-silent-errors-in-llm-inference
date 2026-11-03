import torch
from typing import Dict, List, Tuple

def postprocess_hf_activations() -> Dict[str, Dict[str, Tuple[torch.Tensor, ...]]]:
    ''' Post-process HuggingFace hidden states to match vLLM hidden states.
    
    vLLM fuses gate_proj and up_proj into a single gate_up_proj module.
    The output of gate_up_proj is the concatenation of gate_proj and up_proj
    along the last dimension, with the batch dimension removed.
    
    Output:
    - result: A dictionary containing vLLM component names as keys and a dictionary as values with the following keys:
            - 'HF': a tuple of torch.Tensor that has the same length and shape as the output activations in vLLM. 
              Each tensor is the postprocessed HuggingFace output activations to match the output activations in vLLM 
              or copied from the original vLLM output activations if no matching HuggingFace output activations are found.
            - 'vLLM': a tuple of torch.Tensor that is the original vLLM output activations.
    '''
    # === DO NOT CHANGE THE CODE BELOW ===
    # load the activations
    # hf_model_layers_0_mlp_gate_proj_output: Tuple of tensors collected from the HF module's output
    hf_model_layers_0_mlp_gate_proj_output = torch.load('/path/to/activations/agent-training/Qwen2.5-7B-Instruct/dec2/hf_activation_rank0_model_layers_0_mlp_gate_proj_output.pt', weights_only=False)
    # hf_model_layers_0_mlp_up_proj_output: Tuple of tensors collected from the HF module's output
    hf_model_layers_0_mlp_up_proj_output = torch.load('/path/to/activations/agent-training/Qwen2.5-7B-Instruct/dec2/hf_activation_rank0_model_layers_0_mlp_up_proj_output.pt', weights_only=False)

    # vllm_model_layers_0_mlp_gate_up_proj_output: Tuple of tensors collected from the vLLM module's output
    vllm_model_layers_0_mlp_gate_up_proj_output = torch.load('/path/to/activations/agent-training/Qwen2.5-7B-Instruct/dec2/vllm_activation_rank0_model_layers_0_mlp_gate_up_proj_output.pt', weights_only=False)

    # create the result dict
    result = {"model.model.layers.0.mlp.gate_up_proj": {"HF": None, "vLLM": None},}
    # === DO NOT CHANGE THE CODE ABOVE ===
    
    # Extract the actual tensors from HuggingFace outputs
    hf_gate_proj_tensor = hf_model_layers_0_mlp_gate_proj_output[0][0]
    hf_up_proj_tensor = hf_model_layers_0_mlp_up_proj_output[0][0]
    
    # Remove batch dimension (dimension 0) to match vLLM's shape
    hf_gate_proj_tensor = hf_gate_proj_tensor.squeeze(0)
    hf_up_proj_tensor = hf_up_proj_tensor.squeeze(0)
    
    # Concatenate gate_proj and up_proj along the last dimension to match vLLM's fused projection
    hf_gate_up_proj = torch.cat([hf_gate_proj_tensor, hf_up_proj_tensor], dim=-1)
    
    # Extract vLLM tensor from its structure
    vllm_tensor = vllm_model_layers_0_mlp_gate_up_proj_output[0][0]
    
    # Create matching structures
    result["model.model.layers.0.mlp.gate_up_proj"]["HF"] = (hf_gate_up_proj,)
    result["model.model.layers.0.mlp.gate_up_proj"]["vLLM"] = (vllm_tensor,)
    
    # === DO NOT CHANGE THE CODE BELOW ===
    return result
    # === DO NOT CHANGE THE CODE ABOVE ===