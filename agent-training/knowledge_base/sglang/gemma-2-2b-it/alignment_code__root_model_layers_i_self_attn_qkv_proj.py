import torch
from typing import Dict, List
# import any other necessary modules here
    
def postprocess_hf_activations() -> Dict[str, Dict[str, List[torch.Tensor]]]:
    ''' Post-process HuggingFace hidden states to match vLLM hidden states.
    
    This function aligns HuggingFace's separate q_proj, k_proj, v_proj outputs
    with vLLM's combined qkv_proj output by concatenating them along the last dimension
    and adjusting shape to match vLLM's format.
    
    Output:
    - result: A dictionary containing vLLM component names as keys and a dictionary as values with the following keys:
            - 'HF': a tuple of torch.Tensor that has the same length and shape as the output activations in vLLM. 
            - 'vLLM': a tuple of torch.Tensor that is the original vLLM output activations.
    '''
    # === DO NOT CHANGE THE CODE BELOW ===
    # load the activations
    # hf_model_layers_0_self_attn_q_proj_output: Tuple of tensors collected from the HF module's output
    hf_model_layers_0_self_attn_q_proj_output = torch.load('/path/to/activations/agent-training/gemma-2-2b-it/dec10/hf_activation_rank0_model_layers_0_self_attn_q_proj_output.pt', weights_only=False)
    # hf_model_layers_0_self_attn_k_proj_output: Tuple of tensors collected from the HF module's output
    hf_model_layers_0_self_attn_k_proj_output = torch.load('/path/to/activations/agent-training/gemma-2-2b-it/dec10/hf_activation_rank0_model_layers_0_self_attn_k_proj_output.pt', weights_only=False)
    # hf_model_layers_0_self_attn_v_proj_output: Tuple of tensors collected from the HF module's output
    hf_model_layers_0_self_attn_v_proj_output = torch.load('/path/to/activations/agent-training/gemma-2-2b-it/dec10/hf_activation_rank0_model_layers_0_self_attn_v_proj_output.pt', weights_only=False)

    # vllm_model_layers_0_self_attn_qkv_proj_output: Tuple of tensors collected from the vLLM module's output
    vllm_model_layers_0_self_attn_qkv_proj_output = torch.load('/path/to/activations/agent-training/gemma-2-2b-it/dec10/sglang_activation_rank0_model_layers_0_self_attn_qkv_proj_output.pt', weights_only=False)

    # create the result dict
    result = {"model.layers.0.self_attn.qkv_proj": {"HF": None, "vLLM": None},}
    # === DO NOT CHANGE THE CODE ABOVE ===
    
    # Extract Q, K, V tensors from HF outputs
    hf_q = hf_model_layers_0_self_attn_q_proj_output[0][0] if isinstance(hf_model_layers_0_self_attn_q_proj_output[0], tuple) else hf_model_layers_0_self_attn_q_proj_output[0]
    hf_k = hf_model_layers_0_self_attn_k_proj_output[0][0] if isinstance(hf_model_layers_0_self_attn_k_proj_output[0], tuple) else hf_model_layers_0_self_attn_k_proj_output[0]
    hf_v = hf_model_layers_0_self_attn_v_proj_output[0][0] if isinstance(hf_model_layers_0_self_attn_v_proj_output[0], tuple) else hf_model_layers_0_self_attn_v_proj_output[0]
    
    # Concatenate Q, K, V along the last dimension
    hf_qkv = torch.cat([hf_q, hf_k, hf_v], dim=-1)
    
    # Reshape to match vLLM's shape
    hf_qkv_reshaped = hf_qkv.squeeze(0)
    
    # Extract vLLM tensors from the output structure
    vllm_tensor_0 = vllm_model_layers_0_self_attn_qkv_proj_output[0][0] if isinstance(vllm_model_layers_0_self_attn_qkv_proj_output[0], tuple) else vllm_model_layers_0_self_attn_qkv_proj_output[0]
    vllm_tensor_1 = vllm_model_layers_0_self_attn_qkv_proj_output[1][0] if isinstance(vllm_model_layers_0_self_attn_qkv_proj_output[1], tuple) else vllm_model_layers_0_self_attn_qkv_proj_output[1]
    
    # Create output tuples
    hf_output = (hf_qkv_reshaped, vllm_tensor_1)
    vllm_output = (vllm_tensor_0, vllm_tensor_1)
    
    # Assign to result
    result["model.layers.0.self_attn.qkv_proj"]["HF"] = hf_output
    result["model.layers.0.self_attn.qkv_proj"]["vLLM"] = vllm_output
    
    # === DO NOT CHANGE THE CODE BELOW ===
    return result
    # === DO NOT CHANGE THE CODE ABOVE ===