import torch
from typing import Dict, List

def postprocess_hf_activations() -> Dict[str, Dict[str, List[torch.Tensor]]]:
    ''' Post-process HuggingFace hidden states to match vLLM hidden states.
    
    The HuggingFace output has shape (1, 1051, 3584) with a batch dimension,
    while vLLM output has shape (1051, 3584) without the batch dimension.
    We need to squeeze/remove the batch dimension from HuggingFace output.
    
    The outputs are structured as nested tuples/lists where:
    - output[0] gives us a tuple/list
    - output[0][0] gives us the actual tensor
    
    Output:
    - result: A dictionary containing vLLM component names as keys and a dictionary as values with the following keys:
            - 'HF': a tuple of torch.Tensor that has the same length and shape as the output activations in vLLM. But each tensor is the postprocessed HuggingFace output activations to match the output activations in vLLM or copied from the original vLLM output activations if no matching HuggingFace output activations are found.
            - 'vLLM': a tuple of torch.Tensor that is the original vLLM output activations.
    '''
    # === DO NOT CHANGE THE CODE BELOW ===
    # load the activations
    # hf_model_layers_0_input_layernorm_output: Tuple of tensors collected from the HF module's output
    hf_model_layers_0_input_layernorm_output = torch.load('/path/to/activations/agent-training/Qwen2.5-7B-Instruct/dec2/hf_activation_rank0_model_layers_0_input_layernorm_output.pt', weights_only=False)

    # vllm_model_layers_0_input_layernorm_output: Tuple of tensors collected from the vLLM module's output
    vllm_model_layers_0_input_layernorm_output = torch.load('/path/to/activations/agent-training/Qwen2.5-7B-Instruct/dec2/vllm_activation_rank0_model_layers_0_input_layernorm_output.pt', weights_only=False)

    # create the result dict
    result = {"model.layers.0.input_layernorm": {"HF": None, "vLLM": None},}
    # === DO NOT CHANGE THE CODE ABOVE ===
    
    # Extract the actual tensors from the nested structure
    hf_tensor = None
    if isinstance(hf_model_layers_0_input_layernorm_output, (list, tuple)):
        if len(hf_model_layers_0_input_layernorm_output) > 0:
            first_elem = hf_model_layers_0_input_layernorm_output[0]
            if isinstance(first_elem, (list, tuple)) and len(first_elem) > 0:
                hf_tensor = first_elem[0]
            elif isinstance(first_elem, torch.Tensor):
                hf_tensor = first_elem
    elif isinstance(hf_model_layers_0_input_layernorm_output, torch.Tensor):
        hf_tensor = hf_model_layers_0_input_layernorm_output
    
    vllm_tensor = None
    if isinstance(vllm_model_layers_0_input_layernorm_output, (list, tuple)):
        if len(vllm_model_layers_0_input_layernorm_output) > 0:
            first_elem = vllm_model_layers_0_input_layernorm_output[0]
            if isinstance(first_elem, (list, tuple)) and len(first_elem) > 0:
                vllm_tensor = first_elem[0]
            elif isinstance(first_elem, torch.Tensor):
                vllm_tensor = first_elem
    elif isinstance(vllm_model_layers_0_input_layernorm_output, torch.Tensor):
        vllm_tensor = vllm_model_layers_0_input_layernorm_output
    
    if hf_tensor is not None and isinstance(hf_tensor, torch.Tensor) and hf_tensor.numel() > 0:
        if hf_tensor.dim() == 3 and hf_tensor.shape[0] == 1:
            hf_processed_tensor = hf_tensor.squeeze(0)
        else:
            hf_processed_tensor = hf_tensor
    else:
        hf_processed_tensor = vllm_tensor
    
    result["model.layers.0.input_layernorm"]["HF"] = (hf_processed_tensor,)
    result["model.layers.0.input_layernorm"]["vLLM"] = (vllm_tensor,) if vllm_tensor is not None else tuple()
    
    # === DO NOT CHANGE THE CODE BELOW ===
    return result
    # === DO NOT CHANGE THE CODE ABOVE ===