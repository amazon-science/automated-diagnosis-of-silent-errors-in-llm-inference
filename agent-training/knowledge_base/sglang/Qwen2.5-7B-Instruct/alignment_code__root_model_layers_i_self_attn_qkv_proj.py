import torch
from typing import Dict, List
# import any other necessary modules here
    
def postprocess_hf_activations() -> Dict[str, Dict[str, List[torch.Tensor]]]:
    ''' Post-process HuggingFace hidden states to match vLLM hidden states.
    
    Reasoning:
    - HuggingFace uses separate Q, K, V projection layers with outputs:
      * Q: shape (1, 1051, 896)
      * K: shape (1, 1051, 128)
      * V: shape (1, 1051, 128)
    - vLLM uses a fused QKV projection with output tuple containing:
      * output[0][0]: (1051, 1152) - fused QKV activations where 1152 = 896 + 128 + 128
      * output[1][0]: (1, 1152) - aggregated/summary tensor
    
    To align HF with vLLM:
    1. Extract tensors from HF output tuples (outputs are ((tensor,),) format)
    2. Squeeze batch dimension from HF tensors: (1, 1051, D) -> (1051, D)
    3. Concatenate Q, K, V along last dimension: (1051, 896+128+128) = (1051, 1152)
    4. Match the vLLM output tuple structure with two tensors
    
    Output:
    - result: A dictionary containing vLLM component names as keys and a dictionary as values with the following keys:
            - 'HF': a tuple of torch.Tensor that has the same length and shape as the output activations in vLLM.
            - 'vLLM': a tuple of torch.Tensor that is the original vLLM output activations.
    '''
    # === DO NOT CHANGE THE CODE BELOW ===
    # load the activations
    # hf_model_layers_0_self_attn_q_proj_output: Tuple of tensors collected from the HF module's output
    hf_model_layers_0_self_attn_q_proj_output = torch.load('/path/to/activations/agent-training/Qwen2.5-7B-Instruct/dec10/hf_activation_rank0_model_layers_0_self_attn_q_proj_output.pt', weights_only=False)
    # hf_model_layers_0_self_attn_k_proj_output: Tuple of tensors collected from the HF module's output
    hf_model_layers_0_self_attn_k_proj_output = torch.load('/path/to/activations/agent-training/Qwen2.5-7B-Instruct/dec10/hf_activation_rank0_model_layers_0_self_attn_k_proj_output.pt', weights_only=False)
    # hf_model_layers_0_self_attn_v_proj_output: Tuple of tensors collected from the HF module's output
    hf_model_layers_0_self_attn_v_proj_output = torch.load('/path/to/activations/agent-training/Qwen2.5-7B-Instruct/dec10/hf_activation_rank0_model_layers_0_self_attn_v_proj_output.pt', weights_only=False)

    # vllm_model_layers_0_self_attn_qkv_proj_output: Tuple of tensors collected from the vLLM module's output
    vllm_model_layers_0_self_attn_qkv_proj_output = torch.load('/path/to/activations/agent-training/Qwen2.5-7B-Instruct/dec10/sglang_activation_rank0_model_layers_0_self_attn_qkv_proj_output.pt', weights_only=False)

    # create the result dict
    result = {"model.layers.0.self_attn.qkv_proj": {"HF": None, "vLLM": None},}
    # === DO NOT CHANGE THE CODE ABOVE ===
    
    # Extract the actual tensors from nested tuples
    # The output structure is: ((tensor,),) based on how outputs are collected
    hf_q = hf_model_layers_0_self_attn_q_proj_output[0][0] if isinstance(hf_model_layers_0_self_attn_q_proj_output[0], tuple) else hf_model_layers_0_self_attn_q_proj_output[0]
    hf_k = hf_model_layers_0_self_attn_k_proj_output[0][0] if isinstance(hf_model_layers_0_self_attn_k_proj_output[0], tuple) else hf_model_layers_0_self_attn_k_proj_output[0]
    hf_v = hf_model_layers_0_self_attn_v_proj_output[0][0] if isinstance(hf_model_layers_0_self_attn_v_proj_output[0], tuple) else hf_model_layers_0_self_attn_v_proj_output[0]
    
    # Squeeze batch dimension to match vLLM shape
    hf_q = hf_q.squeeze(0)  # (1051, 896)
    hf_k = hf_k.squeeze(0)  # (1051, 128)
    hf_v = hf_v.squeeze(0)  # (1051, 128)
    
    # Concatenate Q, K, V along last dimension to create fused QKV
    hf_qkv_fused = torch.cat([hf_q, hf_k, hf_v], dim=-1)  # (1051, 1152)
    
    # Extract the vLLM output tensors
    vllm_output_0 = vllm_model_layers_0_self_attn_qkv_proj_output[0][0] if isinstance(vllm_model_layers_0_self_attn_qkv_proj_output[0], tuple) else vllm_model_layers_0_self_attn_qkv_proj_output[0]
    vllm_output_1 = vllm_model_layers_0_self_attn_qkv_proj_output[1][0] if isinstance(vllm_model_layers_0_self_attn_qkv_proj_output[1], tuple) else vllm_model_layers_0_self_attn_qkv_proj_output[1]
    
    # Create HF output tuple matching vLLM structure
    # vLLM output is a tuple of 2 tensors: (tensor_shape_1051x1152, tensor_shape_1x1152)
    hf_output = (hf_qkv_fused, vllm_output_1)
    vllm_output = (vllm_output_0, vllm_output_1)
    
    # Store results
    result["model.layers.0.self_attn.qkv_proj"]["HF"] = hf_output
    result["model.layers.0.self_attn.qkv_proj"]["vLLM"] = vllm_output
    
    # === DO NOT CHANGE THE CODE BELOW ===
    return result
    # === DO NOT CHANGE THE CODE ABOVE ===