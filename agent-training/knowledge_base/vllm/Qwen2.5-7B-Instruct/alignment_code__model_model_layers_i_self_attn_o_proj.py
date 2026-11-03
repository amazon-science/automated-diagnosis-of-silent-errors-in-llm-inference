import torch
from typing import Dict, List
# import any other necessary modules here
    
def postprocess_hf_activations() -> Dict[str, Dict[str, List[torch.Tensor]]]:
    ''' Post-process HuggingFace hidden states to match vLLM hidden states.
    
    Based on the tensor matching results:
    - HF outputs a single tensor with shape (1, 1051, 3584) stored as the first element ([0][0])
    - vLLM outputs a single tensor with shape (1051, 3584) stored as the first element ([0][0])
    
    The activations are stored with keys like "hf_model_layers_0_self_attn_o_proj_output_0_0"
    which means the actual tensor is already extracted from nested structures during saving.
    
    We need to:
    1. The tensors are already extracted, so we access them directly via the loaded keys
    2. Squeeze the batch dimension from HF: (1, 1051, 3584) -> (1051, 3584)
    3. Return the tensors in a list format matching vLLM's output structure
    
    Output:
    - result: A dictionary containing vLLM component names as keys and a dictionary as values with the following keys:
            - 'HF': a list containing the postprocessed HuggingFace tensor (batch dimension squeezed)
            - 'vLLM': a list containing the original vLLM tensor
    '''
    # === DO NOT CHANGE THE CODE BELOW ===
    # load the activations
    # hf_model_layers_0_self_attn_o_proj_output: Tuple of tensors collected from the HF module's output
    hf_model_layers_0_self_attn_o_proj_output = torch.load('/path/to/activations/agent-training/Qwen2.5-7B-Instruct/dec2/hf_activation_rank0_model_layers_0_self_attn_o_proj_output.pt', weights_only=False)

    # vllm_model_layers_0_self_attn_o_proj_output: Tuple of tensors collected from the vLLM module's output
    vllm_model_layers_0_self_attn_o_proj_output = torch.load('/path/to/activations/agent-training/Qwen2.5-7B-Instruct/dec2/vllm_activation_rank0_model_layers_0_self_attn_o_proj_output.pt', weights_only=False)

    # create the result dict
    result = {"model.layers.0.self_attn.o_proj": {"HF": None, "vLLM": None},}
    # === DO NOT CHANGE THE CODE ABOVE ===
    
    # Extract tensors from the loaded structures
    # Based on the key names and snapshots, the actual tensor is at [0][0]
    hf_tensor = hf_model_layers_0_self_attn_o_proj_output[0][0]
    vllm_tensor = vllm_model_layers_0_self_attn_o_proj_output[0][0]
    
    # Squeeze the batch dimension from HF tensor: (1, 1051, 3584) -> (1051, 3584)
    hf_tensor_squeezed = hf_tensor.squeeze(0)
    
    # Create lists matching the structure expected by the alignment checker
    # Both should be lists containing the tensors
    hf_output_postprocessed = [hf_tensor_squeezed]
    vllm_output_list = [vllm_tensor]
    
    # Assign to result
    result["model.layers.0.self_attn.o_proj"]["HF"] = hf_output_postprocessed
    result["model.layers.0.self_attn.o_proj"]["vLLM"] = vllm_output_list
    
    # === DO NOT CHANGE THE CODE BELOW ===
    return result
    # === DO NOT CHANGE THE CODE ABOVE ===