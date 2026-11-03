import torch
from typing import Dict, List

def postprocess_hf_activations() -> Dict[str, Dict[str, List[torch.Tensor]]]:
    ''' Post-process HuggingFace hidden states to match vLLM hidden states.
    
    Output:
    - result: A dictionary containing vLLM component names as keys and a dictionary as values with the following keys:
            - 'HF': a tuple of torch.Tensor that has the same length and shape as the output activations in vLLM. 
            - 'vLLM': a tuple of torch.Tensor that is the original vLLM output activations.
    '''
    # === DO NOT CHANGE THE CODE BELOW ===
    # load the activations
    # hf_model_layers_0_post_attention_layernorm_output: Tuple of tensors collected from the HF module's output
    hf_model_layers_0_post_attention_layernorm_output = torch.load('/path/to/activations/agent-training/Qwen2.5-7B-Instruct/dec2/hf_activation_rank0_model_layers_0_post_attention_layernorm_output.pt', weights_only=False)

    # vllm_model_layers_0_post_attention_layernorm_output: Tuple of tensors collected from the vLLM module's output
    vllm_model_layers_0_post_attention_layernorm_output = torch.load('/path/to/activations/agent-training/Qwen2.5-7B-Instruct/dec2/vllm_activation_rank0_model_layers_0_post_attention_layernorm_output.pt', weights_only=False)

    # create the result dict
    result = {"model.layers.0.post_attention_layernorm": {"HF": None, "vLLM": None},}
    # === DO NOT CHANGE THE CODE ABOVE ===
    
    # Extract HF output and squeeze batch dimension
    # HF shape: (1, seq_len, hidden_dim) -> (seq_len, hidden_dim)
    hf_output_tensor = hf_model_layers_0_post_attention_layernorm_output[0][0]
    hf_output_squeezed = hf_output_tensor.squeeze(0)
    
    # vLLM returns a tuple of two tensors
    vllm_output_0 = vllm_model_layers_0_post_attention_layernorm_output[0][0]
    vllm_output_1 = vllm_model_layers_0_post_attention_layernorm_output[0][1]
    
    # Match HF first output with vLLM first output, copy vLLM second output
    result["model.layers.0.post_attention_layernorm"]["HF"] = (hf_output_squeezed, vllm_output_1)
    result["model.layers.0.post_attention_layernorm"]["vLLM"] = (vllm_output_0, vllm_output_1)
    
    # === DO NOT CHANGE THE CODE BELOW ===
    return result
    # === DO NOT CHANGE THE CODE ABOVE ===