import torch
from typing import Dict, List
# import any other necessary modules here
    
def postprocess_hf_activations() -> Dict[str, Dict[str, List[torch.Tensor]]]:
    ''' Post-process HuggingFace hidden states to match vLLM hidden states.
    
    The HuggingFace Linear layer outputs a tensor with shape (batch, seq_len, hidden_dim).
    The vLLM output is a nested tuple structure that we need to flatten into a list of tensors.
    
    vLLM has two output tensors:
    - output[0][0]: shape (seq_len, hidden_dim) - the main output
    - output[1][0]: shape (batch, hidden_dim) - additional output (possibly for decode/caching)
    
    We need to:
    1. Extract and squeeze HF's tensor to match vLLM's first output shape
    2. For vLLM's second output, we use the original vLLM tensor since HF doesn't produce this
    
    Output:
    - result: A dictionary containing vLLM component names as keys and a dictionary as values with the following keys:
            - 'HF': a tuple of torch.Tensor (flattened, not nested)
            - 'vLLM': a tuple of torch.Tensor (flattened, not nested)
    '''
    # === DO NOT CHANGE THE CODE BELOW ===
    # load the activations
    # hf_model_layers_0_self_attn_o_proj_output: Tuple of tensors collected from the HF module's output
    hf_model_layers_0_self_attn_o_proj_output = torch.load('/path/to/activations/agent-training/Qwen2.5-7B-Instruct/dec10/hf_activation_rank0_model_layers_0_self_attn_o_proj_output.pt', weights_only=False)

    # vllm_model_layers_0_self_attn_o_proj_output: Tuple of tensors collected from the vLLM module's output
    vllm_model_layers_0_self_attn_o_proj_output = torch.load('/path/to/activations/agent-training/Qwen2.5-7B-Instruct/dec10/sglang_activation_rank0_model_layers_0_self_attn_o_proj_output.pt', weights_only=False)

    # create the result dict
    result = {"model.layers.0.self_attn.o_proj": {"HF": None, "vLLM": None},}
    # === DO NOT CHANGE THE CODE ABOVE ===
    
    # HF output: extract the tensor from nested tuple structure
    # hf_output[0][0] has shape (batch, seq_len, hidden_dim) = (1, 1051, 3584)
    hf_output_tensor = hf_model_layers_0_self_attn_o_proj_output[0][0]
    
    # vLLM output: extract tensors from nested tuple structure
    # vllm_output[0][0] has shape (seq_len, hidden_dim) = (1051, 3584)
    # vllm_output[1][0] has shape (batch, hidden_dim) = (1, 3584)
    vllm_output_0 = vllm_model_layers_0_self_attn_o_proj_output[0][0]
    vllm_output_1 = vllm_model_layers_0_self_attn_o_proj_output[1][0]
    
    # Process HF tensor to match vLLM's first output shape
    # Squeeze batch dimension: (1, 1051, 3584) -> (1051, 3584)
    hf_output_0_squeezed = hf_output_tensor.squeeze(0)
    
    # For vLLM's second output, we don't have a corresponding HF activation,
    # so we use the original vLLM tensor
    hf_output_1 = vllm_output_1
    
    # Create flat tuples of tensors (not nested)
    hf_processed = (hf_output_0_squeezed, hf_output_1)
    vllm_processed = (vllm_output_0, vllm_output_1)
    
    # Store results
    result["model.layers.0.self_attn.o_proj"]["HF"] = hf_processed
    result["model.layers.0.self_attn.o_proj"]["vLLM"] = vllm_processed
    
    # === DO NOT CHANGE THE CODE BELOW ===
    return result
    # === DO NOT CHANGE THE CODE ABOVE ===