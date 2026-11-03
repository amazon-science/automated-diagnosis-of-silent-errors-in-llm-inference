import torch
from typing import Dict, List

def postprocess_hf_activations() -> Dict[str, Dict[str, List[torch.Tensor]]]:
    '''Post-process HuggingFace hidden states to match vLLM hidden states.
    
    The post_attention_layernorm in HuggingFace returns a tuple containing a single tensor 
    with shape [1, seq_len, hidden_size].
    
    The post_attention_layernorm in vLLM returns a nested tuple structure:
    ((norm_output_full_seq, residual_full_seq), (norm_output_single, residual_single))
    
    However, for alignment purposes, we need to flatten this to a simple tuple of 4 tensors.
    We can only match HF's normalized output (after squeezing) to vLLM's first output.
    The rest (residuals and single token outputs) come from vLLM since HF doesn't compute those.
    
    Output:
    - result: A dictionary containing vLLM component names as keys and a dictionary as values with the following keys:
            - 'HF': a tuple of torch.Tensor that has the same length and shape as the output activations in vLLM
            - 'vLLM': a tuple of torch.Tensor that is the original vLLM output activations
    '''
    # === DO NOT CHANGE THE CODE BELOW ===
    # load the activations
    # hf_model_layers_0_post_attention_layernorm_output: Tuple of tensors collected from the HF module's output
    hf_model_layers_0_post_attention_layernorm_output = torch.load('/path/to/activations/agent-training/Llama-3.1-8B-Instruct/dec2/hf_activation_rank0_model_layers_0_post_attention_layernorm_output.pt', weights_only=False)

    # vllm_model_layers_0_post_attention_layernorm_output: Tuple of tensors collected from the vLLM module's output
    vllm_model_layers_0_post_attention_layernorm_output = torch.load('/path/to/activations/agent-training/Llama-3.1-8B-Instruct/dec2/sglang_activation_rank0_model_layers_0_post_attention_layernorm_output.pt', weights_only=False)

    # create the result dict
    result = {"model.layers.0.post_attention_layernorm": {"HF": None, "vLLM": None}}
    # === DO NOT CHANGE THE CODE ABOVE ===
    
    # HF output is a nested tuple structure: ((tensor,),)
    # Extract the actual tensor from nested tuples
    hf_norm_output = hf_model_layers_0_post_attention_layernorm_output[0][0]
    
    # Squeeze the batch dimension from HF output to match vLLM format
    # HF: [1, 926, 4096] -> [926, 4096]
    hf_norm_output_squeezed = hf_norm_output.squeeze(0)
    
    # vLLM output structure: ((norm_output, residual), (single_token_norm, single_token_residual))
    vllm_output = vllm_model_layers_0_post_attention_layernorm_output
    
    # Flatten both structures to simple tuples of 4 tensors for comparison
    # HF: (matched_norm, vllm_residual, vllm_single_norm, vllm_single_residual)
    # vLLM: (norm, residual, single_norm, single_residual)
    hf_output_flattened = (
        hf_norm_output_squeezed,  # Matched normalized output
        vllm_output[0][1],        # Residual from vLLM (HF doesn't compute this)
        vllm_output[1][0],        # Single token norm from vLLM (HF doesn't compute this)
        vllm_output[1][1]         # Single token residual from vLLM (HF doesn't compute this)
    )
    
    vllm_output_flattened = (
        vllm_output[0][0],  # Normalized output
        vllm_output[0][1],  # Residual
        vllm_output[1][0],  # Single token norm
        vllm_output[1][1]   # Single token residual
    )
    
    result["model.layers.0.post_attention_layernorm"]["HF"] = hf_output_flattened
    result["model.layers.0.post_attention_layernorm"]["vLLM"] = vllm_output_flattened
    
    # === DO NOT CHANGE THE CODE BELOW ===
    return result
    # === DO NOT CHANGE THE CODE ABOVE ===