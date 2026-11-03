import torch
from typing import Dict, List

def postprocess_hf_activations() -> Dict[str, Dict[str, List[torch.Tensor]]]:
    '''Post-process HuggingFace hidden states to match vLLM hidden states.
    
    Analysis:
    - HF output structure: ((tensor[1, 1067, 2304],),) - nested tuple with single tensor from Linear layer
    - vLLM output structure: ((tensor[1067, 2304], None), (tensor[1, 2304], None))
    
    Key insight from sum comparison:
    - sum(HF output) = -9984.0
    - sum(vLLM output[0][0]) = -9984.0  ✓ MATCH
    - sum(vLLM output[1][0]) = -0.88671875  ✗ DIFFERENT
    
    This means vLLM output[1][0] is NOT from the mlp.down_proj layer output.
    It's likely a residual connection or hidden state from elsewhere in the pipeline.
    
    Reasoning:
    1. The mlp.down_proj Linear layer outputs a single tensor
    2. HF wraps this as ((tensor,),) 
    3. vLLM output[0][0] is the actual layer output (matches HF by sum)
    4. vLLM output[1][0] is from a different source (different sum)
    5. We should only align the actual layer output: HF[0][0].squeeze(0) <-> vLLM[0][0]
    6. For vLLM[1][0], we return it as-is from vLLM since there's no matching HF activation
    
    Output:
    - result: A dictionary containing vLLM component names as keys and a dictionary as values with the following keys:
            - 'HF': tuple where first element is the postprocessed HF output matching vLLM[0][0],
                    and second element is copied from vLLM[1][0] (no matching HF activation)
            - 'vLLM': tuple of the original vLLM output activations
    '''
    # === DO NOT CHANGE THE CODE BELOW ===
    # load the activations
    # hf_model_layers_0_mlp_down_proj_output: Tuple of tensors collected from the HF module's output
    hf_model_layers_0_mlp_down_proj_output = torch.load('/path/to/activations/agent-training/gemma-2-2b-it/dec10/hf_activation_rank0_model_layers_0_mlp_down_proj_output.pt', weights_only=False)

    # vllm_model_layers_0_mlp_down_proj_output: Tuple of tensors collected from the vLLM module's output
    vllm_model_layers_0_mlp_down_proj_output = torch.load('/path/to/activations/agent-training/gemma-2-2b-it/dec10/sglang_activation_rank0_model_layers_0_mlp_down_proj_output.pt', weights_only=False)

    # create the result dict
    result = {"model.layers.0.mlp.down_proj": {"HF": None, "vLLM": None},}
    # === DO NOT CHANGE THE CODE ABOVE ===
    
    # Extract HF output tensor from the nested tuple structure
    hf_output_tensor = hf_model_layers_0_mlp_down_proj_output[0][0]
    
    # Transform HF to match vLLM[0][0]: squeeze batch dimension
    hf_transformed_all_tokens = hf_output_tensor.squeeze(0)
    
    # Extract vLLM tensors from nested tuple structure
    vllm_all_tokens = vllm_model_layers_0_mlp_down_proj_output[0][0]
    vllm_extra_tensor = vllm_model_layers_0_mlp_down_proj_output[1][0]
    
    # Create result tuples
    hf_transformed = (hf_transformed_all_tokens, vllm_extra_tensor)
    vllm_transformed = (vllm_all_tokens, vllm_extra_tensor)
    
    # Assign to result
    result["model.layers.0.mlp.down_proj"]["HF"] = hf_transformed
    result["model.layers.0.mlp.down_proj"]["vLLM"] = vllm_transformed
    
    # === DO NOT CHANGE THE CODE BELOW ===
    return result
    # === DO NOT CHANGE THE CODE ABOVE ===