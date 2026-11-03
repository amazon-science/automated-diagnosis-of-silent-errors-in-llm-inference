import torch
from typing import Dict, List
# import any other necessary modules here
    
def postprocess_hf_activations() -> Dict[str, Dict[str, List[torch.Tensor]]]:
    ''' Post-process HuggingFace hidden states to match vLLM hidden states.
    
    vLLM fuses the Q, K, V projections into a single qkv_proj layer that outputs
    concatenated [Q, K, V] tensors with shape (seq_len, q_dim + k_dim + v_dim).
    
    HuggingFace has separate q_proj, k_proj, v_proj layers with shapes:
    - q_proj output: (1, seq_len, 512)
    - k_proj output: (1, seq_len, 256)
    - v_proj output: (1, seq_len, 256)
    
    The loaded files contain lists/tuples where the actual tensors can be accessed
    by indexing. Based on the tensor key structure, the tensors are at [0][0].
    
    We need to:
    1. Extract the tensors from the loaded structures
    2. Squeeze the batch dimension from HF outputs
    3. Concatenate Q, K, V along the last dimension
    4. Return as tuple structure matching vLLM
    
    Output:
    - result: A dictionary containing vLLM component names as keys and a dictionary as values with the following keys:
            - 'HF': a tuple of torch.Tensor that has the same length and shape as the output activations in vLLM. But each tensor is the postprocessed HuggingFace output activations to match the output activations in vLLM or copied from the original vLLM output activations if no matching HuggingFace output activations are found.
            - 'vLLM': a tuple of torch.Tensor that is the original vLLM output activations.
    '''
    # === DO NOT CHANGE THE CODE BELOW ===
    # load the activations
    # hf_model_layers_0_self_attn_q_proj_output: Tuple of tensors collected from the HF module's output
    hf_model_layers_0_self_attn_q_proj_output = torch.load('/path/to/activations/agent-training/gemma-2-2b-it/dec3/hf_activation_rank0_model_layers_0_self_attn_q_proj_output.pt', weights_only=False)
    # hf_model_layers_0_self_attn_k_proj_output: Tuple of tensors collected from the HF module's output
    hf_model_layers_0_self_attn_k_proj_output = torch.load('/path/to/activations/agent-training/gemma-2-2b-it/dec3/hf_activation_rank0_model_layers_0_self_attn_k_proj_output.pt', weights_only=False)
    # hf_model_layers_0_self_attn_v_proj_output: Tuple of tensors collected from the HF module's output
    hf_model_layers_0_self_attn_v_proj_output = torch.load('/path/to/activations/agent-training/gemma-2-2b-it/dec3/hf_activation_rank0_model_layers_0_self_attn_v_proj_output.pt', weights_only=False)

    # vllm_model_layers_0_self_attn_qkv_proj_output: Tuple of tensors collected from the vLLM module's output
    vllm_model_layers_0_self_attn_qkv_proj_output = torch.load('/path/to/activations/agent-training/gemma-2-2b-it/dec3/vllm_activation_rank0_model_layers_0_self_attn_qkv_proj_output.pt', weights_only=False)

    # create the result dict
    result = {"model.layers.0.self_attn.qkv_proj": {"HF": None, "vLLM": None},}
    # === DO NOT CHANGE THE CODE ABOVE ===
    
    # Extract tensors from loaded structures
    hf_q = hf_model_layers_0_self_attn_q_proj_output[0][0]
    hf_k = hf_model_layers_0_self_attn_k_proj_output[0][0]
    hf_v = hf_model_layers_0_self_attn_v_proj_output[0][0]
    
    # Squeeze batch dimension (from [1, seq_len, dim] to [seq_len, dim])
    hf_q_squeezed = hf_q.squeeze(0)
    hf_k_squeezed = hf_k.squeeze(0)
    hf_v_squeezed = hf_v.squeeze(0)
    
    # Concatenate to match vLLM's qkv_proj output: [Q, K, V]
    hf_qkv = torch.cat([hf_q_squeezed, hf_k_squeezed, hf_v_squeezed], dim=-1)
    
    # Extract vLLM tensor
    vllm_qkv = vllm_model_layers_0_self_attn_qkv_proj_output[0][0]
    
    # Match vLLM's output structure
    hf_processed = (hf_qkv,)
    vllm_processed = (vllm_qkv,)
    
    # Store the results
    result["model.layers.0.self_attn.qkv_proj"]["HF"] = hf_processed
    result["model.layers.0.self_attn.qkv_proj"]["vLLM"] = vllm_processed
    
    # === DO NOT CHANGE THE CODE BELOW ===
    return result
    # === DO NOT CHANGE THE CODE ABOVE ===