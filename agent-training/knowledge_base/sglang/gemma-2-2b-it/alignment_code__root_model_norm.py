import torch
from typing import Dict, List, Tuple

def postprocess_hf_activations() -> Dict[str, Dict[str, List[torch.Tensor]]]:
    '''Post-process HuggingFace hidden states to match vLLM hidden states.
    
    vLLM returns: ((normalized_all, residual_all), (normalized_last, residual_last))
    HF returns: ((normalized_all_with_batch_dim,),)
    
    We can only align the normalized outputs since HF doesn't return residuals.
    For residuals, we use zeros as placeholders to maintain structure but they won't be compared.
    
    Output:
    - result: A dictionary containing vLLM component names as keys and a dictionary as values with the following keys:
            - 'HF': a tuple of torch.Tensor (flat, not nested) that has the same length and shape as the output activations in vLLM. 
            - 'vLLM': a tuple of torch.Tensor (flat, not nested) that is the original vLLM output activations.
    '''
    # === DO NOT CHANGE THE CODE BELOW ===
    # load the activations
    # hf_model_norm_output: Tuple of tensors collected from the HF module's output
    hf_model_norm_output = torch.load('/path/to/activations/agent-training/gemma-2-2b-it/dec10/hf_activation_rank0_model_norm_output.pt', weights_only=False)

    # vllm_model_norm_output: Tuple of tensors collected from the vLLM module's output
    vllm_model_norm_output = torch.load('/path/to/activations/agent-training/gemma-2-2b-it/dec10/sglang_activation_rank0_model_norm_output.pt', weights_only=False)

    # create the result dict
    result = {"model.norm": {"HF": None, "vLLM": None},}
    # === DO NOT CHANGE THE CODE ABOVE ===
    
    # Extract HF normalized output tensor from nested tuple structure
    # HF structure: ((tensor,),) -> access [0][0] to get the actual tensor
    # HF tensor shape: [1, 1067, 2304] -> squeeze to [1067, 2304]
    hf_normalized_all = hf_model_norm_output[0][0].squeeze(0)
    
    # Extract last token from HF output for matching vLLM's last token output
    # Shape: [1067, 2304] -> [1, 2304] (take last along seq dimension)
    hf_normalized_last = hf_normalized_all[-1:, :]
    
    # vLLM structure: ((normalized_all, residual_all), (normalized_last, residual_last))
    # Extract individual tensors
    vllm_normalized_all = vllm_model_norm_output[0][0]  # [1067, 2304]
    vllm_residual_all = vllm_model_norm_output[0][1]    # [1067, 2304]
    vllm_normalized_last = vllm_model_norm_output[1][0] # [1, 2304]
    vllm_residual_last = vllm_model_norm_output[1][1]   # [1, 2304]
    
    # Create zero tensors for residuals (HF doesn't compute them in output)
    # These act as placeholders and should ideally not be used in comparison
    hf_residual_all = torch.zeros_like(hf_normalized_all)
    hf_residual_last = torch.zeros_like(hf_normalized_last)
    
    # Flatten vLLM output to a tuple of 4 tensors
    vllm_flat = (vllm_normalized_all, vllm_residual_all, vllm_normalized_last, vllm_residual_last)
    
    # Construct HF output to match vLLM structure (flat tuple of 4 tensors)
    hf_flat = (hf_normalized_all, hf_residual_all, hf_normalized_last, hf_residual_last)
    
    # Store results
    result["model.norm"]["HF"] = hf_flat
    result["model.norm"]["vLLM"] = vllm_flat
    
    # === DO NOT CHANGE THE CODE BELOW ===
    return result
    # === DO NOT CHANGE THE CODE ABOVE ===