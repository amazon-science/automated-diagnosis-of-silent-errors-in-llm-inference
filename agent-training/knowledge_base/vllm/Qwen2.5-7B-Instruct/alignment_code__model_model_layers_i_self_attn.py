import torch
from typing import Dict, List
    
def postprocess_hf_activations() -> Dict[str, Dict[str, List[torch.Tensor]]]:
    ''' Post-process HuggingFace hidden states to match vLLM hidden states.
    
    The HuggingFace self_attn module returns a nested tuple where:
    - output[0][0]: attention output tensor with shape [batch_size, seq_len, hidden_size]
    - output[0][1]: None (or attention weights if output_attentions=True)
    
    The vLLM self_attn module returns a nested tuple where:
    - output[0][0]: attention output tensor with shape [seq_len, hidden_size] (no batch dim)
    
    Post-processing: Remove the batch dimension from HF output by squeezing dimension 0.
    
    Output:
    - result: A dictionary containing vLLM component names as keys and a dictionary as values with the following keys:
            - 'HF': a tuple of torch.Tensor that has the same length and shape as the output activations in vLLM.
            - 'vLLM': a tuple of torch.Tensor that is the original vLLM output activations.
    '''
    # === DO NOT CHANGE THE CODE BELOW ===
    # load the activations
    # hf_model_layers_0_self_attn_output: Tuple of tensors collected from the HF module's output
    hf_model_layers_0_self_attn_output = torch.load('/path/to/activations/agent-training/Qwen2.5-7B-Instruct/dec2/hf_activation_rank0_model_layers_0_self_attn_output.pt', weights_only=False)

    # vllm_model_layers_0_self_attn_output: Tuple of tensors collected from the vLLM module's output
    vllm_model_layers_0_self_attn_output = torch.load('/path/to/activations/agent-training/Qwen2.5-7B-Instruct/dec2/vllm_activation_rank0_model_layers_0_self_attn_output.pt', weights_only=False)

    # create the result dict
    result = {"model.layers.0.self_attn": {"HF": None, "vLLM": None},}
    # === DO NOT CHANGE THE CODE ABOVE ===
    
    # Extract the attention output tensor from HF (nested indexing: [0][0])
    # HF shape: [batch_size, seq_len, hidden_size] = [1, 1051, 3584]
    hf_attn_output = hf_model_layers_0_self_attn_output[0][0]
    
    # Remove batch dimension to match vLLM format
    # vLLM shape: [seq_len, hidden_size] = [1051, 3584]
    hf_attn_output_squeezed = hf_attn_output.squeeze(0)
    
    # vLLM output: extract the tensor using nested indexing [0][0]
    vllm_attn_output = vllm_model_layers_0_self_attn_output[0][0]
    
    # Create tuples matching vLLM's output structure
    result["model.layers.0.self_attn"]["HF"] = (hf_attn_output_squeezed,)
    result["model.layers.0.self_attn"]["vLLM"] = (vllm_attn_output,)
    
    # === DO NOT CHANGE THE CODE BELOW ===
    return result
    # === DO NOT CHANGE THE CODE ABOVE ===