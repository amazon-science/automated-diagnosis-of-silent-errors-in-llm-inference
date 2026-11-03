import torch
from typing import Dict, List
    
def postprocess_hf_activations() -> Dict[str, Dict[str, List[torch.Tensor]]]:
    ''' Post-process HuggingFace hidden states to match vLLM hidden states.
    
    The vLLM output contains two tensors:
    - First tensor: The normalized hidden states with batch dimension removed (shape: [seq_len, hidden_size])
    - Second tensor: Unknown vLLM-specific tensor (shape: [1, hidden_size]) - copied from vLLM as no HF match found
    
    The HuggingFace output contains one tensor:
    - Single tensor: The normalized hidden states with batch dimension (shape: [batch_size, seq_len, hidden_size])
    
    We transform the first HF output to match vLLM's first output, and copy vLLM's second output as-is.
    
    Output:
    - result: A dictionary containing vLLM component names as keys and a dictionary as values with the following keys:
            - 'HF': a tuple of torch.Tensor that has the same length and shape as the output activations in vLLM. But each tensor is the postprocessed HuggingFace output activations to match the output activations in vLLM or copied from the original vLLM output activations if no matching HuggingFace output activations are found.
            - 'vLLM': a tuple of torch.Tensor that is the original vLLM output activations.
    '''
    # === DO NOT CHANGE THE CODE BELOW ===
    # load the activations
    # hf_model_layers_0_input_layernorm_output: Tuple of tensors collected from the HF module's output
    hf_model_layers_0_input_layernorm_output = torch.load("/path/to/activations/agent-training/gemma-2-2b-it/dec10/hf_activation_rank0_model_layers_0_input_layernorm_output.pt", weights_only=False)

    # vllm_model_layers_0_input_layernorm_output: Tuple of tensors collected from the vLLM module's output
    vllm_model_layers_0_input_layernorm_output = torch.load("/path/to/activations/agent-training/gemma-2-2b-it/dec10/sglang_activation_rank0_model_layers_0_input_layernorm_output.pt", weights_only=False)

    # create the result dict
    result = {"model.layers.0.input_layernorm": {"HF": None, "vLLM": None},}
    # === DO NOT CHANGE THE CODE ABOVE ===
    
    # Extract HF output
    hf_output = hf_model_layers_0_input_layernorm_output[0][0]
    
    # Extract vLLM outputs
    vllm_output_0 = vllm_model_layers_0_input_layernorm_output[0][0]
    vllm_output_1 = vllm_model_layers_0_input_layernorm_output[1][0]
    
    # Transform HF output to match vLLM's first output
    # Remove batch dimension to match vLLM format
    hf_transformed_0 = hf_output.squeeze(0)
    
    # For the second tensor, no matching HF activation found, so copy from vLLM
    hf_transformed_1 = vllm_output_1
    
    # Package the results as tuples
    result["model.layers.0.input_layernorm"]["HF"] = (hf_transformed_0, hf_transformed_1)
    result["model.layers.0.input_layernorm"]["vLLM"] = (vllm_output_0, vllm_output_1)
    
    # === DO NOT CHANGE THE CODE BELOW ===
    return result
    # === DO NOT CHANGE THE CODE ABOVE ===