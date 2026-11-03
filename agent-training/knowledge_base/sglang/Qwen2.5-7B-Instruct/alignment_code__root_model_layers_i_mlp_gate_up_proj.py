import torch
from typing import Dict, List

def postprocess_hf_activations() -> Dict[str, Dict[str, List[torch.Tensor]]]:
    ''' Post-process HuggingFace hidden states to match vLLM hidden states.
    
    Reasoning:
    - vLLM fuses gate_proj and up_proj into a single gate_up_proj module for efficiency
    - The vLLM output structure contains tensors at [0][0] with shape (1051, 9472) and [1][0] with shape (1, 9472)
    - HF gate_proj output[0][0] has shape (1, 1051, 4736)
    - HF up_proj output[0][0] has shape (1, 1051, 4736)
    - We need to: 1) extract tensors from HF tuples, 2) squeeze batch dim, 3) concatenate along last dim
    - The result should be a flat tuple of tensors matching vLLM's structure
    
    Output:
    - result: A dictionary containing vLLM component names as keys and a dictionary as values with the following keys:
            - 'HF': a tuple of torch.Tensor that has the same length and shape as the output activations in vLLM. 
            - 'vLLM': a tuple of torch.Tensor that is the original vLLM output activations.
    '''
    # === DO NOT CHANGE THE CODE BELOW ===
    # load the activations
    # hf_model_layers_0_mlp_gate_proj_output: Tuple of tensors collected from the HF module's output
    hf_model_layers_0_mlp_gate_proj_output = torch.load('/path/to/activations/agent-training/Qwen2.5-7B-Instruct/dec10/hf_activation_rank0_model_layers_0_mlp_gate_proj_output.pt', weights_only=False)
    # hf_model_layers_0_mlp_up_proj_output: Tuple of tensors collected from the HF module's output
    hf_model_layers_0_mlp_up_proj_output = torch.load('/path/to/activations/agent-training/Qwen2.5-7B-Instruct/dec10/hf_activation_rank0_model_layers_0_mlp_up_proj_output.pt', weights_only=False)

    # vllm_model_layers_0_mlp_gate_up_proj_output: Tuple of tensors collected from the vLLM module's output
    vllm_model_layers_0_mlp_gate_up_proj_output = torch.load('/path/to/activations/agent-training/Qwen2.5-7B-Instruct/dec10/sglang_activation_rank0_model_layers_0_mlp_gate_up_proj_output.pt', weights_only=False)

    # create the result dict
    result = {"model.layers.0.mlp.gate_up_proj": {"HF": None, "vLLM": None},}
    # === DO NOT CHANGE THE CODE ABOVE ===
    
    # Extract the actual tensor from tuples
    hf_gate_output = hf_model_layers_0_mlp_gate_proj_output[0][0]
    hf_up_output = hf_model_layers_0_mlp_up_proj_output[0][0]
    
    # Squeeze the batch dimension from HF outputs
    hf_gate_output_squeezed = hf_gate_output.squeeze(0)
    hf_up_output_squeezed = hf_up_output.squeeze(0)
    
    # Concatenate gate and up projections along the last dimension
    hf_fused_output = torch.cat([hf_gate_output_squeezed, hf_up_output_squeezed], dim=-1)
    
    # For the second tensor in vLLM output, we don't have a corresponding HF activation
    # so we use the vLLM tensor directly
    vllm_second_tensor = vllm_model_layers_0_mlp_gate_up_proj_output[1][0]
    
    # Create flat tuples of tensors (not nested tuples)
    hf_aligned_output = (hf_fused_output, vllm_second_tensor)
    vllm_flat_output = (vllm_model_layers_0_mlp_gate_up_proj_output[0][0], 
                        vllm_model_layers_0_mlp_gate_up_proj_output[1][0])
    
    # Set the result
    result["model.layers.0.mlp.gate_up_proj"]["HF"] = hf_aligned_output
    result["model.layers.0.mlp.gate_up_proj"]["vLLM"] = vllm_flat_output
    
    # === DO NOT CHANGE THE CODE BELOW ===
    return result
    # === DO NOT CHANGE THE CODE ABOVE ===