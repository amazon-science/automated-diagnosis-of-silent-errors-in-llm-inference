import torch
from typing import Dict, List
# import any other necessary modules here
    
def postprocess_hf_activations() -> Dict[str, Dict[str, List[torch.Tensor]]]:
    ''' Post-process HuggingFace hidden states to match vLLM hidden states.
    
    Based on the file structure analysis:
    - HuggingFace: outputs a list containing a tuple with one tensor of shape (1, seq_len, hidden_dim)
    - vLLM: outputs a list containing a tuple with two tensors of shape (seq_len, hidden_dim)
    
    The vLLM's first output is the normalized output (matches HF after squeezing batch dim).
    The vLLM's second output is a residual tensor that HF doesn't compute.
    
    Output:
    - result: A dictionary containing vLLM component names as keys and a dictionary as values with the following keys:
            - 'HF': a tuple of torch.Tensor that has the same length and shape as the output activations in vLLM.
            - 'vLLM': a tuple of torch.Tensor that is the original vLLM output activations.
    '''
    # === DO NOT CHANGE THE CODE BELOW ===
    # load the activations
    # hf_model_layers_0_post_attention_layernorm_output: Tuple of tensors collected from the HF module's output
    hf_model_layers_0_post_attention_layernorm_output = torch.load('/path/to/activations/agent-training/Llama-3.1-8B-Instruct/dec2/hf_activation_rank0_model_layers_0_post_attention_layernorm_output.pt', weights_only=False)

    # vllm_model_layers_0_post_attention_layernorm_output: Tuple of tensors collected from the vLLM module's output
    vllm_model_layers_0_post_attention_layernorm_output = torch.load('/path/to/activations/agent-training/Llama-3.1-8B-Instruct/dec2/vllm_activation_rank0_model_layers_0_post_attention_layernorm_output.pt', weights_only=False)

    # create the result dict
    result = {"model.layers.0.post_attention_layernorm": {"HF": None, "vLLM": None},}
    # === DO NOT CHANGE THE CODE ABOVE ===
    
    # Extract from the list wrapper
    # Both outputs are lists with a single element
    hf_output_list_item = hf_model_layers_0_post_attention_layernorm_output[0]
    vllm_output_list_item = vllm_model_layers_0_post_attention_layernorm_output[0]
    
    # Now handle the tuple structure
    # HF: the list item is a tuple with one tensor
    # vLLM: the list item is a tuple with two tensors
    if isinstance(hf_output_list_item, (tuple, list)):
        hf_tensor = hf_output_list_item[0]  # Extract the tensor from the tuple
    else:
        hf_tensor = hf_output_list_item
    
    # vLLM output is a tuple of tensors
    if isinstance(vllm_output_list_item, (tuple, list)):
        vllm_output_tuple = vllm_output_list_item
    else:
        # If it's a single tensor, wrap it in a tuple
        vllm_output_tuple = (vllm_output_list_item,)
    
    # Squeeze the batch dimension from HF output to match vLLM shape
    # HF shape: (1, seq_len, hidden_dim) -> (seq_len, hidden_dim)
    hf_normalized_squeezed = hf_tensor.squeeze(0)
    
    # Create the HF tuple to match vLLM structure
    if len(vllm_output_tuple) == 1:
        # vLLM has only one output
        hf_tuple = (hf_normalized_squeezed,)
    else:
        # vLLM has multiple outputs (e.g., normalized + residual)
        # For the first output: use the squeezed HF normalized output
        # For additional outputs: use vLLM's tensors (since HF doesn't compute them)
        hf_tuple = (hf_normalized_squeezed,) + vllm_output_tuple[1:]
    
    # Store the results
    result["model.layers.0.post_attention_layernorm"]["HF"] = hf_tuple
    result["model.layers.0.post_attention_layernorm"]["vLLM"] = vllm_output_tuple
    
    # === DO NOT CHANGE THE CODE BELOW ===
    return result
    # === DO NOT CHANGE THE CODE ABOVE ===