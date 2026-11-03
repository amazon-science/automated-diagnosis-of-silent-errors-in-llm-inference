import torch
from typing import Dict, List, Tuple

def postprocess_hf_activations() -> Dict[str, Dict[str, List[torch.Tensor]]]:
    ''' Post-process HuggingFace hidden states to match vLLM hidden states.
    
    The HuggingFace pre_feedforward_layernorm outputs a nested tuple structure: ((tensor[1, seq_len, hidden_size],),)
    while vLLM outputs a nested tuple structure: ((tensor1[seq_len, hidden_size], tensor2[seq_len, hidden_size]),)
    
    The first vLLM tensor corresponds to the normalized output (matching HF after squeezing batch dim).
    The second vLLM tensor is an additional output used internally by vLLM.
    
    Since HuggingFace only outputs the normalized result, we:
    1. Squeeze the batch dimension from HF output to match vLLM's first output
    2. Use vLLM's second output as-is (no matching HF activation exists)
    
    Output:
    - result: A dictionary containing vLLM component names as keys and a dictionary as values with the following keys:
            - 'HF': a tuple of torch.Tensor that has the same length and shape as the output activations in vLLM. 
                    First tensor is postprocessed HF output (squeezed), second is copied from vLLM.
            - 'vLLM': a tuple of torch.Tensor that is the original vLLM output activations.
    '''
    # === DO NOT CHANGE THE CODE BELOW ===
    # load the activations
    # hf_model_layers_0_pre_feedforward_layernorm_output: Tuple of tensors collected from the HF module's output
    hf_model_layers_0_pre_feedforward_layernorm_output = torch.load('/path/to/activations/agent-training/gemma-2-2b-it/dec3/hf_activation_rank0_model_layers_0_pre_feedforward_layernorm_output.pt', weights_only=False)

    # vllm_model_layers_0_pre_feedforward_layernorm_output: Tuple of tensors collected from the vLLM module's output
    vllm_model_layers_0_pre_feedforward_layernorm_output = torch.load('/path/to/activations/agent-training/gemma-2-2b-it/dec3/vllm_activation_rank0_model_layers_0_pre_feedforward_layernorm_output.pt', weights_only=False)

    # create the result dict
    result = {"model.layers.0.pre_feedforward_layernorm": {"HF": None, "vLLM": None},}
    # === DO NOT CHANGE THE CODE ABOVE ===
    
    # Extract HF output - nested tuple structure: ((tensor[1, 1067, 2304],),)
    hf_nested = hf_model_layers_0_pre_feedforward_layernorm_output[0]
    hf_tensor = hf_nested[0]
    
    # Squeeze the batch dimension to match vLLM's shape [1067, 2304]
    hf_output_squeezed = hf_tensor.squeeze(0)
    
    # Extract vLLM outputs - nested tuple structure: ((tensor1[1067, 2304], tensor2[1067, 2304]),)
    vllm_nested = vllm_model_layers_0_pre_feedforward_layernorm_output[0]
    vllm_output_0 = vllm_nested[0]
    vllm_output_1 = vllm_nested[1]
    
    # Create HF tuple: first element is postprocessed HF output, second is copied from vLLM
    hf_tuple = (hf_output_squeezed, vllm_output_1)
    
    # vLLM tuple stays as-is
    vllm_tuple = (vllm_output_0, vllm_output_1)
    
    # Populate result
    result["model.layers.0.pre_feedforward_layernorm"]["HF"] = hf_tuple
    result["model.layers.0.pre_feedforward_layernorm"]["vLLM"] = vllm_tuple
    
    # === DO NOT CHANGE THE CODE BELOW ===
    return result
    # === DO NOT CHANGE THE CODE ABOVE ===