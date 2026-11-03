import torch
from typing import Dict, List

def postprocess_hf_activations() -> Dict[str, Dict[str, List[torch.Tensor]]]:
    ''' Post-process HuggingFace hidden states to match vLLM hidden states.
    
    The HuggingFace embedding layer outputs tensors with a batch dimension [batch, seq_len, hidden_dim],
    while vLLM outputs tensors without the batch dimension [seq_len, hidden_dim].
    This function squeezes the batch dimension from HF outputs to match vLLM format.
    
    Output:
    - result: A dictionary containing vLLM component names as keys and a dictionary as values with the following keys:
            - 'HF': a tuple of torch.Tensor that has the same length and shape as the output activations in vLLM. But each tensor is the postprocessed HuggingFace output activations to match the output activations in vLLM or copied from the original vLLM output activations if no matching HuggingFace output activations are found.
            - 'vLLM': a tuple of torch.Tensor that is the original vLLM output activations.
    '''
    # === DO NOT CHANGE THE CODE BELOW ===
    # load the activations
    # hf_model_embed_tokens_output: Tuple of tensors collected from the HF module's output
    hf_model_embed_tokens_output = torch.load('/path/to/activations/agent-training/Llama-3.1-8B-Instruct/dec2/hf_activation_rank0_model_embed_tokens_output.pt', weights_only=False)

    # vllm_model_embed_tokens_output: Tuple of tensors collected from the vLLM module's output
    vllm_model_embed_tokens_output = torch.load('/path/to/activations/agent-training/Llama-3.1-8B-Instruct/dec2/vllm_activation_rank0_model_embed_tokens_output.pt', weights_only=False)

    # create the result dict
    result = {"model.embed_tokens": {"HF": None, "vLLM": None},}
    # === DO NOT CHANGE THE CODE ABOVE ===
    
    # Post-process HF activations to match vLLM shape
    # Navigate the structure to find the actual tensor
    if isinstance(hf_model_embed_tokens_output, list):
        hf_data = hf_model_embed_tokens_output[0]
        if isinstance(hf_data, tuple):
            hf_tensor = hf_data[0]
        else:
            hf_tensor = hf_data
    elif isinstance(hf_model_embed_tokens_output, tuple):
        hf_tensor = hf_model_embed_tokens_output[0]
    else:
        hf_tensor = hf_model_embed_tokens_output
    
    # Squeeze the batch dimension (dim 0) to match vLLM shape
    if isinstance(hf_tensor, torch.Tensor):
        hf_processed_tensor = hf_tensor.squeeze(0)
    else:
        raise TypeError(f"Expected torch.Tensor but got {type(hf_tensor)}")
    
    # Extract the vLLM tensor
    if isinstance(vllm_model_embed_tokens_output, list):
        vllm_data = vllm_model_embed_tokens_output[0]
        if isinstance(vllm_data, tuple):
            vllm_tensor = vllm_data[0]
        else:
            vllm_tensor = vllm_data
    elif isinstance(vllm_model_embed_tokens_output, tuple):
        vllm_tensor = vllm_model_embed_tokens_output[0]
    else:
        vllm_tensor = vllm_model_embed_tokens_output
    
    # Wrap tensors back into tuples to match the expected output format
    hf_processed = (hf_processed_tensor,)
    vllm_processed = (vllm_tensor,) if isinstance(vllm_tensor, torch.Tensor) else vllm_model_embed_tokens_output
    
    # Assign the processed activations
    result["model.embed_tokens"]["HF"] = hf_processed
    result["model.embed_tokens"]["vLLM"] = vllm_processed
    
    # === DO NOT CHANGE THE CODE BELOW ===
    return result
    # === DO NOT CHANGE THE CODE ABOVE ===