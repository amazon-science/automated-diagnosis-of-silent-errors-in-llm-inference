import torch
from typing import Dict, List
# import any other necessary modules here
    
def postprocess_hf_activations() -> Dict[str, Dict[str, List[torch.Tensor]]]:
    ''' Post-process HuggingFace hidden states to match vLLM hidden states.
    
    The key difference is:
    - HuggingFace outputs a tuple with 1 element: (normalized_hidden_states,)
      where normalized_hidden_states has shape (batch_size, seq_len, hidden_dim)
    - vLLM outputs a tuple with 2 elements: (normalized_hidden_states, residual_mean)
      where normalized_hidden_states has shape (seq_len, hidden_dim) [no batch dim]
      and residual_mean has shape (1, hidden_dim)
    
    Output:
    - result: A dictionary containing vLLM component names as keys and a dictionary as values with the following keys:
            - 'HF': a tuple of torch.Tensor that has the same length and shape as the output activations in vLLM. But each tensor is the postprocessed HuggingFace output activations to match the output activations in vLLM or copied from the original vLLM output activations if no matching HuggingFace output activations are found.
            - 'vLLM': a tuple of torch.Tensor that is the original vLLM output activations.
    '''
    # === DO NOT CHANGE THE CODE BELOW ===
    # load the activations
    # hf_model_layers_0_input_layernorm_output: Tuple of tensors collected from the HF module's output
    hf_model_layers_0_input_layernorm_output = torch.load('/path/to/activations/agent-training/Qwen2.5-7B-Instruct/dec10/hf_activation_rank0_model_layers_0_input_layernorm_output.pt', weights_only=False)

    # vllm_model_layers_0_input_layernorm_output: Tuple of tensors collected from the vLLM module's output
    vllm_model_layers_0_input_layernorm_output = torch.load('/path/to/activations/agent-training/Qwen2.5-7B-Instruct/dec10/sglang_activation_rank0_model_layers_0_input_layernorm_output.pt', weights_only=False)

    # create the result dict
    result = {"model.layers.0.input_layernorm": {"HF": None, "vLLM": None},}
    # === DO NOT CHANGE THE CODE ABOVE ===
    
    # Extract HuggingFace output (tuple with 1 element)
    # HF output is a tuple, and output[0] is a tuple with tensors
    # hf_model_layers_0_input_layernorm_output[0][0] has shape (1, 1051, 3584)
    hf_normalized = hf_model_layers_0_input_layernorm_output[0][0]
    
    # Extract vLLM outputs (tuple with 2 elements)
    # vLLM output[0][0] has shape (1051, 3584) - normalized hidden states
    # vLLM output[1][0] has shape (1, 3584) - residual mean or statistics
    vllm_normalized = vllm_model_layers_0_input_layernorm_output[0][0]
    vllm_residual = vllm_model_layers_0_input_layernorm_output[1][0]
    
    # Postprocess HF output to match vLLM format:
    # 1. Squeeze the batch dimension from HF normalized output to match vLLM normalized output
    hf_normalized_squeezed = hf_normalized.squeeze(0)  # (1, 1051, 3584) -> (1051, 3584)
    
    # 2. For the second output, use vLLM's residual since HF doesn't have a corresponding output
    # This is because HF's RMSNorm doesn't return residual statistics
    
    # Construct the result
    result["model.layers.0.input_layernorm"]["HF"] = (hf_normalized_squeezed, vllm_residual)
    result["model.layers.0.input_layernorm"]["vLLM"] = (vllm_normalized, vllm_residual)
    
    # === DO NOT CHANGE THE CODE BELOW ===
    return result
    # === DO NOT CHANGE THE CODE ABOVE ===