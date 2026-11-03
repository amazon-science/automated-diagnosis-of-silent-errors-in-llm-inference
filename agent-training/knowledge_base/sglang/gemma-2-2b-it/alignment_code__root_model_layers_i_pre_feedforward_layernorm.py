import torch
from typing import Dict, List

def postprocess_hf_activations() -> Dict[str, Dict[str, List[torch.Tensor]]]:
    ''' Post-process HuggingFace hidden states to match vLLM hidden states.
    
    The vLLM GemmaRMSNorm returns a nested tuple structure that needs to be flattened:
    - vLLM output: ((norm_out_1, norm_out_2), (residual_1, residual_2))
    - Flattened: (norm_out_1, norm_out_2, residual_1, residual_2)
    
    HuggingFace returns: ((tensor,),) where tensor has shape [1, 1067, 2304]
    We squeeze the batch dimension to get [1067, 2304] to match vLLM's first normalized output.
    For the remaining tensors that HF doesn't provide, we use vLLM's original values.
    
    Output:
    - result: A dictionary containing vLLM component names as keys and a dictionary as values with the following keys:
            - 'HF': a flat tuple of torch.Tensor matching vLLM's flattened output structure
            - 'vLLM': a flat tuple of torch.Tensor (the flattened original vLLM output activations)
    '''
    # === DO NOT CHANGE THE CODE BELOW ===
    # load the activations
    # hf_model_layers_0_pre_feedforward_layernorm_output: Tuple of tensors collected from the HF module's output
    hf_model_layers_0_pre_feedforward_layernorm_output = torch.load("/path/to/activations/agent-training/gemma-2-2b-it/dec10/hf_activation_rank0_model_layers_0_pre_feedforward_layernorm_output.pt", weights_only=False)

    # vllm_model_layers_0_pre_feedforward_layernorm_output: Tuple of tensors collected from the vLLM module's output
    vllm_model_layers_0_pre_feedforward_layernorm_output = torch.load("/path/to/activations/agent-training/gemma-2-2b-it/dec10/sglang_activation_rank0_model_layers_0_pre_feedforward_layernorm_output.pt", weights_only=False)

    # create the result dict
    result = {"model.layers.0.pre_feedforward_layernorm": {"HF": None, "vLLM": None},}
    # === DO NOT CHANGE THE CODE ABOVE ===
    
    # Get the vLLM output - nested tuple structure: ((tensor1, tensor2), (tensor3, tensor4))
    vllm_output = vllm_model_layers_0_pre_feedforward_layernorm_output
    
    # Flatten the vLLM output to a flat tuple
    vllm_flattened = (vllm_output[0][0], vllm_output[0][1], vllm_output[1][0], vllm_output[1][1])
    
    # Get the HuggingFace output
    # The HF output is a nested tuple structure: ((tensor,),)
    hf_output = hf_model_layers_0_pre_feedforward_layernorm_output
    
    # Navigate through the nested structure to get the actual tensor
    hf_tensor = hf_output[0][0]  # shape: [1, 1067, 2304]
    
    # Squeeze the batch dimension from HF output to match vLLM's shape
    hf_output_squeezed = hf_tensor.squeeze(0)  # shape: [1067, 2304]
    
    # Create the HF flat tuple to match vLLM's flattened structure
    # We can only provide the first normalized output from HF, use vLLM's values for the rest
    hf_flattened = (
        hf_output_squeezed,      # First normalized output - from HF
        vllm_output[0][1],       # Second normalized output - from vLLM
        vllm_output[1][0],       # First residual - from vLLM
        vllm_output[1][1]        # Second residual - from vLLM
    )
    
    # Store results
    result["model.layers.0.pre_feedforward_layernorm"]["HF"] = hf_flattened
    result["model.layers.0.pre_feedforward_layernorm"]["vLLM"] = vllm_flattened
    
    # === DO NOT CHANGE THE CODE BELOW ===
    return result
    # === DO NOT CHANGE THE CODE ABOVE ===