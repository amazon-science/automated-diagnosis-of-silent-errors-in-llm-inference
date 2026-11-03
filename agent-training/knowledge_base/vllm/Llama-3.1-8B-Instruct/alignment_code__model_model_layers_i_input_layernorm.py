import torch
from typing import Dict, List
# import any other necessary modules here
    
def postprocess_hf_activations() -> Dict[str, Dict[str, List[torch.Tensor]]]:
    ''' Post-process HuggingFace hidden states to match vLLM hidden states.
    
    The input_layernorm module outputs a nested structure.
    HuggingFace keeps the batch dimension (shape: [1, seq_len, hidden_size])
    while vLLM removes it (shape: [seq_len, hidden_size]).
    
    This function extracts the actual tensor and squeezes the batch dimension 
    from HuggingFace outputs to match vLLM.
    
    Output:
    - result: A dictionary containing vLLM component names as keys and a dictionary as values with the following keys:
            - 'HF': a tuple of torch.Tensor that has the same length and shape as the output activations in vLLM. But each tensor is the postprocessed HuggingFace output activations to match the output activations in vLLM or copied from the original vLLM output activations if no matching HuggingFace output activations are found.
            - 'vLLM': a tuple of torch.Tensor that is the original vLLM output activations.
    '''
    # === DO NOT CHANGE THE CODE BELOW ===
    # load the activations
    # hf_model_layers_0_input_layernorm_output: Tuple of tensors collected from the HF module's output
    hf_model_layers_0_input_layernorm_output = torch.load('/path/to/activations/agent-training/Llama-3.1-8B-Instruct/dec2/hf_activation_rank0_model_layers_0_input_layernorm_output.pt', weights_only=False)

    # vllm_model_layers_0_input_layernorm_output: Tuple of tensors collected from the vLLM module's output
    vllm_model_layers_0_input_layernorm_output = torch.load('/path/to/activations/agent-training/Llama-3.1-8B-Instruct/dec2/vllm_activation_rank0_model_layers_0_input_layernorm_output.pt', weights_only=False)

    # create the result dict
    result = {"model.layers.0.input_layernorm": {"HF": None, "vLLM": None},}
    # === DO NOT CHANGE THE CODE ABOVE ===
    
    # Process HuggingFace activations to match vLLM format
    # The loaded data appears to be nested: access with [0][0] to get the actual tensor
    # HF: tensor[1, 926, 4096] -> vLLM: tensor[926, 4096]
    # We need to squeeze the batch dimension (dim=0) from HF output
    
    # Extract the actual tensor from HF output and squeeze batch dimension
    hf_tensor = hf_model_layers_0_input_layernorm_output[0][0].squeeze(0)
    hf_processed = (hf_tensor,)
    
    # Extract the actual tensor from vLLM output (already without batch dimension)
    vllm_tensor = vllm_model_layers_0_input_layernorm_output[0][0]
    vllm_processed = (vllm_tensor,)
    
    # Store the results
    result["model.layers.0.input_layernorm"]["HF"] = hf_processed
    result["model.layers.0.input_layernorm"]["vLLM"] = vllm_processed
    
    # === DO NOT CHANGE THE CODE BELOW ===
    return result
    # === DO NOT CHANGE THE CODE ABOVE ===