import torch
from typing import Dict, List
# import any other necessary modules here
    
def postprocess_hf_activations() -> Dict[str, Dict[str, List[torch.Tensor]]]:
    ''' Post-process HuggingFace hidden states to match vLLM hidden states.
    Output:
    - result: A dictionary containing vLLM component names as keys and a dictionary as values with the following keys:
            - 'HF': a tuple of torch.Tensor that has the same length and shape as the output activations in vLLM. But each tensor is the postprocessed HuggingFace output activations to match the output activations in vLLM or copied from the original vLLM output activations if no matching HuggingFace output activations are found.
            - 'vLLM': a tuple of torch.Tensor that is the original vLLM output activations.
    '''
    # === DO NOT CHANGE THE CODE BELOW ===
    # load the activations
    # hf_model_layers_0_self_attn_output: Tuple of tensors collected from the HF module's output
    hf_model_layers_0_self_attn_output = torch.load('/path/to/activations/agent-training/Llama-3.1-8B-Instruct/dec2/hf_activation_rank0_model_layers_0_self_attn_output.pt', weights_only=False)

    # vllm_model_layers_0_self_attn_output: Tuple of tensors collected from the vLLM module's output
    vllm_model_layers_0_self_attn_output = torch.load('/path/to/activations/agent-training/Llama-3.1-8B-Instruct/dec2/vllm_activation_rank0_model_layers_0_self_attn_output.pt', weights_only=False)

    # create the result dict
    result = {"model.model.layers.0.self_attn": {"HF": None, "vLLM": None},}
    # === DO NOT CHANGE THE CODE ABOVE ===
    
    # Post-process HuggingFace activations to match vLLM format
    # HF output structure: tuple containing (tensor of shape [1, 926, 4096], None)
    # vLLM output structure: tuple containing (tensor of shape [926, 4096],)
    
    # Extract the first tensor from HF output (accessing [0][0] to get the actual tensor)
    hf_tensor = hf_model_layers_0_self_attn_output[0][0]
    
    # Squeeze the batch dimension to match vLLM shape
    hf_tensor_squeezed = hf_tensor.squeeze(0)  # Remove batch dimension: [1, 926, 4096] -> [926, 4096]
    
    # Create the processed HF tuple to match vLLM's output structure
    processed_hf_output = (hf_tensor_squeezed,)
    
    # Extract vLLM output (it's already in the correct format)
    vllm_output = vllm_model_layers_0_self_attn_output[0]
    
    # Populate the result dictionary
    result["model.model.layers.0.self_attn"]["HF"] = processed_hf_output
    result["model.model.layers.0.self_attn"]["vLLM"] = vllm_output
    
    # === DO NOT CHANGE THE CODE BELOW ===
    return result
    # === DO NOT CHANGE THE CODE ABOVE ===