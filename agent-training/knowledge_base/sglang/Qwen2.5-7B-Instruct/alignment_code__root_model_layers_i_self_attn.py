import torch
from typing import Dict, List
    
def postprocess_hf_activations() -> Dict[str, Dict[str, List[torch.Tensor]]]:
    ''' Post-process HuggingFace hidden states to match vLLM hidden states.
    
    The HuggingFace self_attn output is a tuple where the first element is the attention output
    with shape (batch, seq_len, hidden_dim). The vLLM output has two elements:
    1. First element: attention output with shape (seq_len, hidden_dim) - squeeze batch dim from HF
    2. Second element: appears to be an internal state (residual) that HF doesn't expose - copy from vLLM
    
    Output:
    - result: A dictionary containing vLLM component names as keys and a dictionary as values with the following keys:
            - 'HF': a tuple of torch.Tensor that has the same length and shape as the output activations in vLLM.
            - 'vLLM': a tuple of torch.Tensor that is the original vLLM output activations.
    '''
    # === DO NOT CHANGE THE CODE BELOW ===
    # load the activations
    # hf_model_layers_0_self_attn_output: Tuple of tensors collected from the HF module's output
    hf_model_layers_0_self_attn_output = torch.load('/path/to/activations/agent-training/Qwen2.5-7B-Instruct/dec10/hf_activation_rank0_model_layers_0_self_attn_output.pt', weights_only=False)

    # vllm_model_layers_0_self_attn_output: Tuple of tensors collected from the vLLM module's output
    vllm_model_layers_0_self_attn_output = torch.load('/path/to/activations/agent-training/Qwen2.5-7B-Instruct/dec10/sglang_activation_rank0_model_layers_0_self_attn_output.pt', weights_only=False)

    # create the result dict
    result = {"model.layers.0.self_attn": {"HF": None, "vLLM": None},}
    # === DO NOT CHANGE THE CODE ABOVE ===
    
    # Extract HuggingFace activations
    hf_output_0 = hf_model_layers_0_self_attn_output[0][0]  # shape: (1, 1051, 3584)
    
    # Process HF output to match vLLM format
    # vLLM first output: shape (1051, 3584) - squeeze batch dimension from HF
    hf_processed_0 = hf_output_0.squeeze(0)  # shape: (1051, 3584)
    
    # vLLM second output: shape (1, 3584) - this appears to be an internal vLLM state
    # that HF doesn't expose in self_attn output. Use vLLM's tensor directly.
    vllm_output_1 = vllm_model_layers_0_self_attn_output[1][0]
    
    # Flatten vLLM structure to tuple of tensors
    vllm_flattened = (vllm_model_layers_0_self_attn_output[0][0], 
                      vllm_output_1)
    
    # Create HF tuple: first element from processed HF, second element from vLLM
    hf_processed = (hf_processed_0, vllm_output_1)
    
    # Store results
    result["model.layers.0.self_attn"]["HF"] = hf_processed
    result["model.layers.0.self_attn"]["vLLM"] = vllm_flattened
    
    # === DO NOT CHANGE THE CODE BELOW ===
    return result
    # === DO NOT CHANGE THE CODE ABOVE ===