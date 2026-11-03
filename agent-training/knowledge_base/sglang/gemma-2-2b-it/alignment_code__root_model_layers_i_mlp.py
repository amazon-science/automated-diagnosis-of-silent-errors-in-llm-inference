import torch
from typing import Dict, List

def postprocess_hf_activations() -> Dict[str, Dict[str, List[torch.Tensor]]]:
    ''' Post-process HuggingFace hidden states to match vLLM hidden states.
    
    Analysis:
    - HF MLP output: tuple containing tensor of shape [1, 1067, 2304]
    - vLLM MLP output: tuple containing two tensors:
        - [0][0]: shape [1067, 2304] - all tokens
        - [1][0]: shape [1, 2304] - specific representation
    
    The first vLLM tensor matches HF when we squeeze the batch dimension.
    The second vLLM tensor doesn't have a clear match in HF outputs, so we use vLLM's value directly.
    
    Output:
    - result: A dictionary containing vLLM component names as keys and a dictionary as values with the following keys:
            - 'HF': a tuple of torch.Tensor that has the same length and shape as the output activations in vLLM. But each tensor is the postprocessed HuggingFace output activations to match the output activations in vLLM or copied from the original vLLM output activations if no matching HuggingFace output activations are found.
            - 'vLLM': a tuple of torch.Tensor that is the original vLLM output activations.
    '''
    # === DO NOT CHANGE THE CODE BELOW ===
    # load the activations
    # hf_model_layers_0_mlp_output: Tuple of tensors collected from the HF module's output
    hf_model_layers_0_mlp_output = torch.load('/path/to/activations/agent-training/gemma-2-2b-it/dec10/hf_activation_rank0_model_layers_0_mlp_output.pt', weights_only=False)

    # vllm_model_layers_0_mlp_output: Tuple of tensors collected from the vLLM module's output
    vllm_model_layers_0_mlp_output = torch.load('/path/to/activations/agent-training/gemma-2-2b-it/dec10/sglang_activation_rank0_model_layers_0_mlp_output.pt', weights_only=False)

    # create the result dict
    result = {"model.layers.0.mlp": {"HF": None, "vLLM": None},}
    # === DO NOT CHANGE THE CODE ABOVE ===
    
    # Extract HF tensor: shape [1, 1067, 2304]
    hf_output_tensor = hf_model_layers_0_mlp_output[0][0]
    
    # Extract vLLM tensors
    vllm_output_0 = vllm_model_layers_0_mlp_output[0][0]  # Shape: [1067, 2304]
    vllm_output_1 = vllm_model_layers_0_mlp_output[1][0]  # Shape: [1, 2304]
    
    # Transform HF to match vLLM structure:
    # 1. First tensor: squeeze batch dimension from [1, 1067, 2304] to [1067, 2304]
    hf_output_0 = hf_output_tensor.squeeze(0)
    
    # 2. Second tensor: No matching HF activation found for vLLM's second output
    #    Use vLLM's value directly as per instructions
    hf_output_1 = vllm_output_1
    
    # Package results as tuples
    result["model.layers.0.mlp"]["HF"] = (hf_output_0, hf_output_1)
    result["model.layers.0.mlp"]["vLLM"] = (vllm_output_0, vllm_output_1)
    
    # === DO NOT CHANGE THE CODE BELOW ===
    return result
    # === DO NOT CHANGE THE CODE ABOVE ===