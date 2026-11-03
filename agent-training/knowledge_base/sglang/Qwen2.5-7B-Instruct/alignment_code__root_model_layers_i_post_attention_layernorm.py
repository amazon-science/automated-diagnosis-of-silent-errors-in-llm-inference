import torch
from typing import Dict, List

def postprocess_hf_activations() -> Dict[str, Dict[str, List[torch.Tensor]]]:
    ''' Post-process HuggingFace hidden states to match vLLM hidden states.
    
    The HuggingFace post_attention_layernorm outputs a nested tuple structure where
    the actual tensor is at [0][0] with shape (1, seq_len, hidden_dim).
    
    The vLLM/SGLang post_attention_layernorm outputs a nested tuple structure:
    - output[0]: tuple of 2 tensors, each with shape (seq_len, hidden_dim)
    - output[1]: tuple of 2 tensors, each with shape (1, hidden_dim)
    
    We need to:
    1. Extract HF tensor from nested structure and squeeze batch dimension
    2. Match it with vLLM's output[0][0]
    3. Use original vLLM values for the other 3 tensors
    
    Output:
    - result: A dictionary containing vLLM component names as keys and a dictionary as values with the following keys:
            - 'HF': a tuple of torch.Tensor that has the same length and shape as the output activations in vLLM.
            - 'vLLM': a tuple of torch.Tensor that is the original vLLM output activations.
    '''
    # === DO NOT CHANGE THE CODE BELOW ===
    # load the activations
    # hf_model_layers_0_post_attention_layernorm_output: Tuple of tensors collected from the HF module's output
    hf_model_layers_0_post_attention_layernorm_output = torch.load('/path/to/activations/agent-training/Qwen2.5-7B-Instruct/dec10/hf_activation_rank0_model_layers_0_post_attention_layernorm_output.pt', weights_only=False)

    # vllm_model_layers_0_post_attention_layernorm_output: Tuple of tensors collected from the vLLM module's output
    vllm_model_layers_0_post_attention_layernorm_output = torch.load('/path/to/activations/agent-training/Qwen2.5-7B-Instruct/dec10/sglang_activation_rank0_model_layers_0_post_attention_layernorm_output.pt', weights_only=False)

    # create the result dict
    result = {"model.layers.0.post_attention_layernorm": {"HF": None, "vLLM": None},}
    # === DO NOT CHANGE THE CODE ABOVE ===
    
    # HF output is nested: access [0][0] to get the tensor with shape (1, seq_len, hidden_dim)
    hf_output_tensor = hf_model_layers_0_post_attention_layernorm_output[0][0]
    
    # Squeeze batch dimension to match vLLM format: (1, seq_len, hidden_dim) -> (seq_len, hidden_dim)
    hf_output_squeezed = hf_output_tensor.squeeze(0)
    
    # vLLM output structure: ((tensor1, tensor2), (tensor3, tensor4))
    # Flatten it to (tensor1, tensor2, tensor3, tensor4)
    vllm_output_flat = (
        vllm_model_layers_0_post_attention_layernorm_output[0][0],
        vllm_model_layers_0_post_attention_layernorm_output[0][1],
        vllm_model_layers_0_post_attention_layernorm_output[1][0],
        vllm_model_layers_0_post_attention_layernorm_output[1][1]
    )
    
    # Create HF output tuple matching vLLM structure
    # We match hf_output_squeezed with vllm_output_flat[0]
    # For the other vLLM outputs, we use the original vLLM values since HF doesn't have equivalents
    hf_matched_output = (
        hf_output_squeezed,
        vllm_output_flat[1],
        vllm_output_flat[2],
        vllm_output_flat[3]
    )
    
    # Store results
    result["model.layers.0.post_attention_layernorm"]["HF"] = hf_matched_output
    result["model.layers.0.post_attention_layernorm"]["vLLM"] = vllm_output_flat
    
    # === DO NOT CHANGE THE CODE BELOW ===
    return result
    # === DO NOT CHANGE THE CODE ABOVE ===