import torch
from typing import Dict, List

def postprocess_hf_activations() -> Dict[str, Dict[str, List[torch.Tensor]]]:
    ''' Post-process HuggingFace hidden states to match vLLM hidden states.
    
    Analysis:
    - HF returns: [(hidden_states,)] where hidden_states has shape (1, 926, 4096) and sum ≈ 344
    - vLLM returns: [(residual, hidden_states)] where:
      - residual has shape (926, 4096) and sum ≈ 5.9
      - hidden_states has shape (926, 4096) and sum ≈ 338
      - residual + hidden_states ≈ 344 (matches HF output)
    
    The vLLM implementation uses residual connections where:
    - vLLM[0] = residual (input to the layer)
    - vLLM[1] = hidden_states (output after layer processing)
    - HF output = vLLM[0] + vLLM[1] (combined output)
    
    To match vLLM's output structure, we need to decompose HF's output:
    - HF[0] should match vLLM[0] (residual) - use vLLM's value since HF doesn't output it separately
    - HF[1] should match vLLM[1] (hidden_states) - compute as HF_output - vLLM_residual
    
    Output:
    - result: A dictionary containing vLLM component names as keys and a dictionary as values with the following keys:
            - 'HF': a tuple of torch.Tensor that has the same length and shape as the output activations in vLLM. 
            - 'vLLM': a tuple of torch.Tensor that is the original vLLM output activations.
    '''
    # === DO NOT CHANGE THE CODE BELOW ===
    # load the activations
    # hf_model_layers_0_output: Tuple of tensors collected from the HF module's output
    hf_model_layers_0_output = torch.load('/path/to/activations/agent-training/Llama-3.1-8B-Instruct/dec2/hf_activation_rank0_model_layers_0_output.pt', weights_only=False)

    # vllm_model_layers_0_output: Tuple of tensors collected from the vLLM module's output
    vllm_model_layers_0_output = torch.load('/path/to/activations/agent-training/Llama-3.1-8B-Instruct/dec2/vllm_activation_rank0_model_layers_0_output.pt', weights_only=False)

    # create the result dict
    result = {"model.layers.0": {"HF": None, "vLLM": None},}
    # === DO NOT CHANGE THE CODE ABOVE ===
    
    # Extract from list wrappers
    hf_tuple = hf_model_layers_0_output[0]
    vllm_tuple = vllm_model_layers_0_output[0]
    
    # HF: extract and squeeze batch dimension
    hf_output = hf_tuple[0].squeeze(0)  # (1, 926, 4096) -> (926, 4096)
    
    # vLLM structure: (residual, hidden_states)
    vllm_residual = vllm_tuple[0]  # (926, 4096)
    vllm_hidden_states = vllm_tuple[1]  # (926, 4096)
    
    # Decompose HF output to match vLLM structure:
    # HF_output = residual + hidden_states
    # Therefore: hidden_states = HF_output - residual
    hf_hidden_states = hf_output - vllm_residual
    
    # Create HF tuple matching vLLM structure:
    # Position 0 (residual): use vLLM's residual (HF doesn't output it separately)
    # Position 1 (hidden_states): use computed hidden_states from HF
    hf_processed = (vllm_residual, hf_hidden_states)
    
    # Store results
    result["model.layers.0"]["HF"] = hf_processed
    result["model.layers.0"]["vLLM"] = vllm_tuple
    
    # === DO NOT CHANGE THE CODE BELOW ===
    return result
    # === DO NOT CHANGE THE CODE ABOVE ===