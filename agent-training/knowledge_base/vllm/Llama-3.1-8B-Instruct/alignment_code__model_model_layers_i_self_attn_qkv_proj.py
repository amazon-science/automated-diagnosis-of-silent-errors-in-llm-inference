import torch
from typing import Dict, List
# import any other necessary modules here
    
def postprocess_hf_activations() -> Dict[str, Dict[str, List[torch.Tensor]]]:
    ''' Post-process HuggingFace hidden states to match vLLM hidden states.
    
    vLLM combines the separate Q, K, V projections from HuggingFace into a single
    QKV projection by concatenating them along the last dimension and removing
    the batch dimension.
    
    The loaded activations are tuples. Each HF module (q_proj, k_proj, v_proj) outputs
    a tuple structure where the actual tensor is accessed via [0][0].
    
    Output:
    - result: A dictionary containing vLLM component names as keys and a dictionary as values with the following keys:
            - 'HF': a tuple of torch.Tensor that has the same length and shape as the output activations in vLLM. But each tensor is the postprocessed HuggingFace output activations to match the output activations in vLLM or copied from the original vLLM output activations if no matching HuggingFace output activations are found.
            - 'vLLM': a tuple of torch.Tensor that is the original vLLM output activations.
    '''
    # === DO NOT CHANGE THE CODE BELOW ===
    # load the activations
    # hf_model_layers_0_self_attn_q_proj_output: Tuple of tensors collected from the HF module's output
    hf_model_layers_0_self_attn_q_proj_output = torch.load('/path/to/activations/agent-training/Llama-3.1-8B-Instruct/dec2/hf_activation_rank0_model_layers_0_self_attn_q_proj_output.pt', weights_only=False)
    # hf_model_layers_0_self_attn_k_proj_output: Tuple of tensors collected from the HF module's output
    hf_model_layers_0_self_attn_k_proj_output = torch.load('/path/to/activations/agent-training/Llama-3.1-8B-Instruct/dec2/hf_activation_rank0_model_layers_0_self_attn_k_proj_output.pt', weights_only=False)
    # hf_model_layers_0_self_attn_v_proj_output: Tuple of tensors collected from the HF module's output
    hf_model_layers_0_self_attn_v_proj_output = torch.load('/path/to/activations/agent-training/Llama-3.1-8B-Instruct/dec2/hf_activation_rank0_model_layers_0_self_attn_v_proj_output.pt', weights_only=False)

    # vllm_model_layers_0_self_attn_qkv_proj_output: Tuple of tensors collected from the vLLM module's output
    vllm_model_layers_0_self_attn_qkv_proj_output = torch.load('/path/to/activations/agent-training/Llama-3.1-8B-Instruct/dec2/vllm_activation_rank0_model_layers_0_self_attn_qkv_proj_output.pt', weights_only=False)

    # create the result dict
    result = {"model.layers.0.self_attn.qkv_proj": {"HF": None, "vLLM": None},}
    # === DO NOT CHANGE THE CODE ABOVE ===
    
    # Extract the actual tensor from each loaded tuple structure
    hf_q = hf_model_layers_0_self_attn_q_proj_output[0][0]
    hf_k = hf_model_layers_0_self_attn_k_proj_output[0][0]
    hf_v = hf_model_layers_0_self_attn_v_proj_output[0][0]
    
    # Concatenate Q, K, V along the last dimension
    hf_qkv = torch.cat([hf_q, hf_k, hf_v], dim=-1)
    
    # Remove the batch dimension to match vLLM format
    hf_qkv = hf_qkv.squeeze(0)
    
    # vLLM output is a tuple structure, extract the actual tensor
    vllm_qkv = vllm_model_layers_0_self_attn_qkv_proj_output[0][0]
    
    # Store the postprocessed HF activations and original vLLM activations as tuples
    result["model.layers.0.self_attn.qkv_proj"]["HF"] = (hf_qkv,)
    result["model.layers.0.self_attn.qkv_proj"]["vLLM"] = (vllm_qkv,)
    
    # === DO NOT CHANGE THE CODE BELOW ===
    return result
    # === DO NOT CHANGE THE CODE ABOVE ===