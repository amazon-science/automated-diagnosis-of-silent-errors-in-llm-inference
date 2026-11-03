import torch
from typing import Dict, List
# import any other necessary modules here
    
def postprocess_hf_activations() -> Dict[str, Dict[str, List[torch.Tensor]]]:
    ''' Post-process HuggingFace hidden states to match vLLM hidden states.
    Output:
    - result: A dictionary containing vLLM component names as keys and a dictionary as values with the following keys:
            - 'HF': a tuple of torch.Tensor that has the same length and shape as the output activations in vLLM. 
                    Each tensor is the postprocessed HuggingFace output activations to match the output activations in vLLM 
                    or copied from the original vLLM output activations if no matching HuggingFace output activations are found.
            - 'vLLM': a tuple of torch.Tensor that is the original vLLM output activations.
    '''
    # === DO NOT CHANGE THE CODE BELOW ===
    # load the activations
    # hf_model_layers_0_mlp_output: Tuple of tensors collected from the HF module's output
    hf_model_layers_0_mlp_output = torch.load('/path/to/activations/agent-training/Qwen2.5-7B-Instruct/dec10/hf_activation_rank0_model_layers_0_mlp_output.pt', weights_only=False)

    # vllm_model_layers_0_mlp_output: Tuple of tensors collected from the vLLM module's output
    vllm_model_layers_0_mlp_output = torch.load('/path/to/activations/agent-training/Qwen2.5-7B-Instruct/dec10/sglang_activation_rank0_model_layers_0_mlp_output.pt', weights_only=False)

    # create the result dict
    result = {"model.layers.0.mlp": {"HF": None, "vLLM": None},}
    # === DO NOT CHANGE THE CODE ABOVE ===
    
    # Extract the HF tensor (navigating the nested structure)
    hf_tensor = hf_model_layers_0_mlp_output[0][0]  # shape: [1, 1051, 3584]
    
    # Squeeze the batch dimension to match vLLM's first output shape
    hf_tensor_squeezed = hf_tensor.squeeze(0)  # shape: [1051, 3584]
    
    # Create the aligned HF tuple matching vLLM structure
    hf_aligned = (hf_tensor_squeezed, vllm_model_layers_0_mlp_output[1][0])
    
    # Store results
    result["model.layers.0.mlp"]["HF"] = hf_aligned
    result["model.layers.0.mlp"]["vLLM"] = (vllm_model_layers_0_mlp_output[0][0], vllm_model_layers_0_mlp_output[1][0])
    
    # === DO NOT CHANGE THE CODE BELOW ===
    return result
    # === DO NOT CHANGE THE CODE ABOVE ===