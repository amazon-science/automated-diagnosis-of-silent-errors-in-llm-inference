import torch
from typing import Dict, List
    
def postprocess_hf_activations() -> Dict[str, Dict[str, List[torch.Tensor]]]:
    ''' Post-process HuggingFace hidden states to match vLLM hidden states.
    
    Analysis:
    - HF Linear layer outputs a single tensor of shape (1, 926, 4096), captured in nested structure
    - vLLM has nested tuple outputs, but only the first element corresponds to o_proj output
    - Based on tensor sum analysis (HF: 163.0 vs vLLM[0][0]: 164.0 vs vLLM[1][0]: 0.24),
      only vLLM's output[0][0] matches the o_proj computation
    - The vLLM output[1][0] appears to be from a different source/computation
    
    Strategy: Extract and compare only the main sequence outputs (full 926 tokens)
    
    Output:
    - result: Dictionary mapping vLLM component names to HF and vLLM activation tuples
    '''
    # === DO NOT CHANGE THE CODE BELOW ===
    hf_model_layers_0_self_attn_o_proj_output = torch.load('/path/to/activations/agent-training/Llama-3.1-8B-Instruct/dec2/hf_activation_rank0_model_layers_0_self_attn_o_proj_output.pt', weights_only=False)
    vllm_model_layers_0_self_attn_o_proj_output = torch.load('/path/to/activations/agent-training/Llama-3.1-8B-Instruct/dec2/sglang_activation_rank0_model_layers_0_self_attn_o_proj_output.pt', weights_only=False)
    result = {"model.layers.0.self_attn.o_proj": {"HF": None, "vLLM": None},}
    # === DO NOT CHANGE THE CODE ABOVE ===
    
    # Extract HF output tensor from nested structure and remove batch dimension
    # Structure: hf_output[0][0] has shape (1, 926, 4096)
    # After squeeze(0): (926, 4096)
    hf_output_squeezed = hf_model_layers_0_self_attn_o_proj_output[0][0].squeeze(0)
    
    # Extract vLLM main output (the actual o_proj output)
    # Structure: vllm_output[0][0] has shape (926, 4096)
    vllm_main_output = vllm_model_layers_0_self_attn_o_proj_output[0][0]
    
    # Return as single-element tuples containing only the comparable outputs
    result["model.layers.0.self_attn.o_proj"]["HF"] = (hf_output_squeezed,)
    result["model.layers.0.self_attn.o_proj"]["vLLM"] = (vllm_main_output,)
    
    # === DO NOT CHANGE THE CODE BELOW ===
    return result
    # === DO NOT CHANGE THE CODE ABOVE ===