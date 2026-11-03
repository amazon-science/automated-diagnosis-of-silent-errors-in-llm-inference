import torch
from typing import Dict, Tuple
# import any other necessary modules here
    
def postprocess_hf_activations() -> Dict[str, Dict[str, Tuple[torch.Tensor, ...]]]:
    ''' Post-process HuggingFace hidden states to match vLLM hidden states.
    
    The HuggingFace MLP output has shape (1, 1067, 2304) with a batch dimension,
    while vLLM output has shape (1067, 2304) without the batch dimension.
    We need to squeeze the batch dimension from HF output to match vLLM.
    
    Output:
    - result: A dictionary containing vLLM component names as keys and a dictionary as values with the following keys:
            - 'HF': a tuple of torch.Tensor that has the same length and shape as the output activations in vLLM. But each tensor is the postprocessed HuggingFace output activations to match the output activations in vLLM or copied from the original vLLM output activations if no matching HuggingFace output activations are found.
            - 'vLLM': a tuple of torch.Tensor that is the original vLLM output activations.
    '''
    # === DO NOT CHANGE THE CODE BELOW ===
    # load the activations
    # hf_model_layers_0_mlp_output: Tuple of tensors collected from the HF module's output
    hf_model_layers_0_mlp_output = torch.load('/path/to/activations/agent-training/gemma-2-2b-it/dec3/hf_activation_rank0_model_layers_0_mlp_output.pt', weights_only=False)

    # vllm_model_layers_0_mlp_output: Tuple of tensors collected from the vLLM module's output
    vllm_model_layers_0_mlp_output = torch.load('/path/to/activations/agent-training/gemma-2-2b-it/dec3/vllm_activation_rank0_model_layers_0_mlp_output.pt', weights_only=False)

    # create the result dict
    result = {"model.layers.0.mlp": {"HF": None, "vLLM": None},}
    # === DO NOT CHANGE THE CODE ABOVE ===
    
    # Extract the tensor from the nested tuple structure
    hf_tensor = hf_model_layers_0_mlp_output[0][0]
    
    # Process HF tensor: squeeze the batch dimension
    # HF shape: (1, 1067, 2304) -> (1067, 2304)
    if isinstance(hf_tensor, torch.Tensor) and hf_tensor.dim() == 3 and hf_tensor.shape[0] == 1:
        hf_processed_tensor = hf_tensor.squeeze(0)
    else:
        hf_processed_tensor = hf_tensor
    
    # Create the output tuple in the same format as the input
    hf_processed = (hf_processed_tensor,)
    vllm_output = (vllm_model_layers_0_mlp_output[0][0],)
    
    # Store the results
    result["model.layers.0.mlp"]["HF"] = hf_processed
    result["model.layers.0.mlp"]["vLLM"] = vllm_output
    
    # === DO NOT CHANGE THE CODE BELOW ===
    return result
    # === DO NOT CHANGE THE CODE ABOVE ===