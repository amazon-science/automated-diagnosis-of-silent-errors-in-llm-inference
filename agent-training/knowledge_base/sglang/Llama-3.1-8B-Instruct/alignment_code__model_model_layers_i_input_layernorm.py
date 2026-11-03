import torch
from typing import Dict, List

def postprocess_hf_activations() -> Dict[str, Dict[str, List[torch.Tensor]]]:
    ''' Post-process HuggingFace hidden states to match vLLM hidden states.
    
    For the input_layernorm layer:
    - vLLM returns a tuple of 2 tensors:
      1. All normalized tokens with shape (seq_len, hidden_dim) - batch dim removed
      2. A tensor with shape (1, hidden_dim) - purpose unclear from HF perspective
    
    - HuggingFace returns a single tensor with shape (batch_size, seq_len, hidden_dim)
    
    Reasoning:
    1. vLLM's first output (926, 4096) aligns with HF output after squeezing the batch dimension
    2. vLLM's second output (1, 4096) does not have a clear corresponding activation in HF output
       that we can derive. According to guidelines, we should use the original vLLM activation.
    
    Output:
    - result: A dictionary containing vLLM component names as keys and a dictionary as values with the following keys:
            - 'HF': a tuple of torch.Tensor - first from processed HF, second from vLLM
            - 'vLLM': a tuple of torch.Tensor that is the original vLLM output activations.
    '''
    # === DO NOT CHANGE THE CODE BELOW ===
    # load the activations
    # hf_model_layers_0_input_layernorm_output: Tuple of tensors collected from the HF module's output
    hf_model_layers_0_input_layernorm_output = torch.load('/path/to/activations/agent-training/Llama-3.1-8B-Instruct/dec2/hf_activation_rank0_model_layers_0_input_layernorm_output.pt', weights_only=False)

    # vllm_model_layers_0_input_layernorm_output: Tuple of tensors collected from the vLLM module's output
    vllm_model_layers_0_input_layernorm_output = torch.load('/path/to/activations/agent-training/Llama-3.1-8B-Instruct/dec2/sglang_activation_rank0_model_layers_0_input_layernorm_output.pt', weights_only=False)

    # create the result dict
    result = {"model.layers.0.input_layernorm": {"HF": None, "vLLM": None}}
    # === DO NOT CHANGE THE CODE ABOVE ===
    
    # Extract tensors from nested tuple structure
    hf_tensor = hf_model_layers_0_input_layernorm_output[0][0]
    vllm_tensor_0 = vllm_model_layers_0_input_layernorm_output[0][0]
    vllm_tensor_1 = vllm_model_layers_0_input_layernorm_output[1][0]
    
    # Process HF first output to match vLLM's first output
    hf_all_tokens = hf_tensor.squeeze(0)
    
    # For vLLM's second output, no corresponding HF activation found, use vLLM's value
    hf_processed = (hf_all_tokens, vllm_tensor_1)
    vllm_processed = (vllm_tensor_0, vllm_tensor_1)
    
    # Store results
    result["model.layers.0.input_layernorm"]["HF"] = hf_processed
    result["model.layers.0.input_layernorm"]["vLLM"] = vllm_processed
    
    # === DO NOT CHANGE THE CODE BELOW ===
    return result
    # === DO NOT CHANGE THE CODE ABOVE ===