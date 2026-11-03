import torch
from typing import Dict, List
# import any other necessary modules here
    
def postprocess_hf_activations() -> Dict[str, Dict[str, List[torch.Tensor]]]:
    ''' Post-process HuggingFace hidden states to match vLLM hidden states.
    
    Reasoning:
    - vLLM fuses the separate Q, K, V projections in HuggingFace into a single QKV projection
    - The vLLM qkv_proj output is a concatenation of [Q, K, V] along the last dimension
    - vLLM also squeezes the batch dimension (changes from (B, seq_len, hidden) to (seq_len, hidden))
    - Based on the snapshot, vLLM output[0] is a tuple containing (tensor, None), but we only need the tensor
    - We extract just the tensor from vLLM and create matching tuple for HF
    
    Output:
    - result: A dictionary containing vLLM component names as keys and a dictionary as values with the following keys:
            - 'HF': a tuple of torch.Tensor that has the same length and shape as the output activations in vLLM
            - 'vLLM': a tuple of torch.Tensor that is the original vLLM output activations
    '''
    # === DO NOT CHANGE THE CODE BELOW ===
    # load the activations
    # hf_model_layers_0_self_attn_q_proj_output: Tuple of tensors collected from the HF module's output
    hf_model_layers_0_self_attn_q_proj_output = torch.load('/path/to/activations/agent-training/Qwen2.5-7B-Instruct/dec2/hf_activation_rank0_model_layers_0_self_attn_q_proj_output.pt', weights_only=False)
    # hf_model_layers_0_self_attn_k_proj_output: Tuple of tensors collected from the HF module's output
    hf_model_layers_0_self_attn_k_proj_output = torch.load('/path/to/activations/agent-training/Qwen2.5-7B-Instruct/dec2/hf_activation_rank0_model_layers_0_self_attn_k_proj_output.pt', weights_only=False)
    # hf_model_layers_0_self_attn_v_proj_output: Tuple of tensors collected from the HF module's output
    hf_model_layers_0_self_attn_v_proj_output = torch.load('/path/to/activations/agent-training/Qwen2.5-7B-Instruct/dec2/hf_activation_rank0_model_layers_0_self_attn_v_proj_output.pt', weights_only=False)

    # vllm_model_layers_0_self_attn_qkv_proj_output: Tuple of tensors collected from the vLLM module's output
    vllm_model_layers_0_self_attn_qkv_proj_output = torch.load('/path/to/activations/agent-training/Qwen2.5-7B-Instruct/dec2/vllm_activation_rank0_model_layers_0_self_attn_qkv_proj_output.pt', weights_only=False)

    # create the result dict
    result = {"model.layers.0.self_attn.qkv_proj": {"HF": None, "vLLM": None},}
    # === DO NOT CHANGE THE CODE ABOVE ===
    
    # Extract the actual tensors - HF Linear layers wrap outputs in tuples
    # The structure is ((tensor,),) for each projection
    hf_q = hf_model_layers_0_self_attn_q_proj_output[0][0]  # (1, 1051, 896)
    hf_k = hf_model_layers_0_self_attn_k_proj_output[0][0]  # (1, 1051, 128)
    hf_v = hf_model_layers_0_self_attn_v_proj_output[0][0]  # (1, 1051, 128)
    
    # Concatenate Q, K, V along the last dimension to form QKV
    hf_qkv = torch.cat([hf_q, hf_k, hf_v], dim=-1)  # (1, 1051, 1152)
    
    # Squeeze the batch dimension to match vLLM's output shape
    hf_qkv = hf_qkv.squeeze(0)  # (1051, 1152)
    
    # Match vLLM's exact output structure
    # vLLM output is ((tensor, None),), so we extract just the tensor and create matching tuples
    result["model.layers.0.self_attn.qkv_proj"]["HF"] = (hf_qkv,)
    result["model.layers.0.self_attn.qkv_proj"]["vLLM"] = (vllm_model_layers_0_self_attn_qkv_proj_output[0][0],)
    
    # === DO NOT CHANGE THE CODE BELOW ===
    return result
    # === DO NOT CHANGE THE CODE ABOVE ===