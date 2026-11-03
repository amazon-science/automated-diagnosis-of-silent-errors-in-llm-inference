import torch
from typing import Dict, List
    
def postprocess_hf_activations() -> Dict[str, Dict[str, List[torch.Tensor]]]:
    ''' Post-process HuggingFace hidden states to match vLLM hidden states.
    
    The HuggingFace output is a single tensor with shape [1, seq_len, hidden_dim].
    The vLLM output is a single tensor with shape [seq_len, hidden_dim].
    We need to squeeze the batch dimension (dimension 0) from HuggingFace output to match vLLM's format.
    
    Output:
    - result: A dictionary containing vLLM component names as keys and a dictionary as values with the following keys:
            - 'HF': a tuple of torch.Tensor that has the same length and shape as the output activations in vLLM. But each tensor is the postprocessed HuggingFace output activations to match the output activations in vLLM or copied from the original vLLM output activations if no matching HuggingFace output activations are found.
            - 'vLLM': a tuple of torch.Tensor that is the original vLLM output activations.
    '''
    # === DO NOT CHANGE THE CODE BELOW ===
    # load the activations
    # hf_model_layers_0_post_feedforward_layernorm_output: Tuple of tensors collected from the HF module's output
    hf_model_layers_0_post_feedforward_layernorm_output = torch.load('/path/to/activations/agent-training/gemma-2-2b-it/dec3/hf_activation_rank0_model_layers_0_post_feedforward_layernorm_output.pt', weights_only=False)

    # vllm_model_layers_0_post_feedforward_layernorm_output: Tuple of tensors collected from the vLLM module's output
    vllm_model_layers_0_post_feedforward_layernorm_output = torch.load('/path/to/activations/agent-training/gemma-2-2b-it/dec3/vllm_activation_rank0_model_layers_0_post_feedforward_layernorm_output.pt', weights_only=False)

    # create the result dict
    result = {"model.layers.0.post_feedforward_layernorm": {"HF": None, "vLLM": None},}
    # === DO NOT CHANGE THE CODE ABOVE ===
    
    if isinstance(hf_model_layers_0_post_feedforward_layernorm_output, (tuple, list)):
        if len(hf_model_layers_0_post_feedforward_layernorm_output) > 0:
            hf_tensor = hf_model_layers_0_post_feedforward_layernorm_output[0]
            while isinstance(hf_tensor, (tuple, list)) and len(hf_tensor) > 0:
                hf_tensor = hf_tensor[0]
        else:
            hf_tensor = None
    else:
        hf_tensor = hf_model_layers_0_post_feedforward_layernorm_output
    
    if isinstance(vllm_model_layers_0_post_feedforward_layernorm_output, (tuple, list)):
        if len(vllm_model_layers_0_post_feedforward_layernorm_output) > 0:
            vllm_tensor = vllm_model_layers_0_post_feedforward_layernorm_output[0]
            while isinstance(vllm_tensor, (tuple, list)) and len(vllm_tensor) > 0:
                vllm_tensor = vllm_tensor[0]
        else:
            vllm_tensor = None
    else:
        vllm_tensor = vllm_model_layers_0_post_feedforward_layernorm_output
    
    if hf_tensor is not None and isinstance(hf_tensor, torch.Tensor):
        if hf_tensor.ndim == 3 and hf_tensor.shape[0] == 1:
            hf_processed = hf_tensor.squeeze(0)
        else:
            hf_processed = hf_tensor
    else:
        hf_processed = vllm_tensor
    
    result["model.layers.0.post_feedforward_layernorm"]["HF"] = (hf_processed,)
    result["model.layers.0.post_feedforward_layernorm"]["vLLM"] = (vllm_tensor,)
    
    # === DO NOT CHANGE THE CODE BELOW ===
    return result
    # === DO NOT CHANGE THE CODE ABOVE ===