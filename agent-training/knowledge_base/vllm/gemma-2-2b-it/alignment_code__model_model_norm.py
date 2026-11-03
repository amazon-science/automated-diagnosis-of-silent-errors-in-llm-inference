import torch
from typing import Dict, List
    
def postprocess_hf_activations() -> Dict[str, Dict[str, List[torch.Tensor]]]:
    ''' Post-process HuggingFace hidden states to match vLLM hidden states.
    
    Reasoning:
    - vLLM's RMSNorm module returns a tuple containing a tuple of 2 tensors: ((normalized_output, residual_input),)
      - normalized_output shape: (1067, 2304) - without batch dimension
      - residual_input shape: (1067, 2304) - the input before normalization
    - HuggingFace's RMSNorm module returns a tuple containing 1 tensor: (normalized_output,)
      - normalized_output shape: (1, 1067, 2304) - with batch dimension
    - We need to:
      1. Extract the HF tensor and squeeze the batch dimension (dim 0) to match vLLM's shape (1067, 2304)
      2. For the second tensor (residual), use vLLM's original since HF doesn't output it
      3. Match vLLM's output structure
    
    Output:
    - result: A dictionary containing vLLM component names as keys and a dictionary as values with the following keys:
            - 'HF': a tuple of torch.Tensor that has the same length and shape as the output activations in vLLM. 
            - 'vLLM': a tuple of torch.Tensor that is the original vLLM output activations.
    '''
    # === DO NOT CHANGE THE CODE BELOW ===
    # load the activations
    # hf_model_norm_output: Tuple of tensors collected from the HF module's output
    hf_model_norm_output = torch.load('/path/to/activations/agent-training/gemma-2-2b-it/dec3/hf_activation_rank0_model_norm_output.pt', weights_only=False)

    # vllm_model_norm_output: Tuple of tensors collected from the vLLM module's output
    vllm_model_norm_output = torch.load('/path/to/activations/agent-training/gemma-2-2b-it/dec3/vllm_activation_rank0_model_norm_output.pt', weights_only=False)

    # create the result dict
    result = {"model.norm": {"HF": None, "vLLM": None},}
    # === DO NOT CHANGE THE CODE ABOVE ===
    
    # Check if vLLM output is nested tuple
    if isinstance(vllm_model_norm_output[0], tuple):
        vllm_tuple = vllm_model_norm_output[0]
    else:
        vllm_tuple = vllm_model_norm_output
    
    # Check HF structure
    if isinstance(hf_model_norm_output[0], tuple):
        hf_tensor = hf_model_norm_output[0][0]
    else:
        hf_tensor = hf_model_norm_output[0]
    
    # Squeeze the batch dimension from HF normalized output
    hf_normalized_squeezed = hf_tensor.squeeze(0)  # shape: (1067, 2304)
    
    # Create result matching vLLM's structure
    if len(vllm_tuple) == 2:
        # vLLM has 2 tensors: (normalized, residual)
        result["model.norm"]["HF"] = (hf_normalized_squeezed, vllm_tuple[1])
    else:
        # vLLM has only 1 tensor
        result["model.norm"]["HF"] = (hf_normalized_squeezed,)
    
    result["model.norm"]["vLLM"] = vllm_tuple
    
    # === DO NOT CHANGE THE CODE BELOW ===
    return result
    # === DO NOT CHANGE THE CODE ABOVE ===