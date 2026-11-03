import torch
from typing import Dict, List
    
def postprocess_hf_activations() -> Dict[str, Dict[str, List[torch.Tensor]]]:
    ''' Post-process HuggingFace hidden states to match vLLM hidden states.
    
    For the embedding layer (model.embed_tokens):
    - HuggingFace outputs shape: (batch_size, seq_len, hidden_dim) = (1, 1067, 2304)
    - vLLM outputs shape: (seq_len, hidden_dim) = (1067, 2304)
    
    The transformation required is to squeeze the batch dimension from HuggingFace output
    since vLLM removes the batch dimension in its embedding output.
    
    Output:
    - result: A dictionary containing vLLM component names as keys and a dictionary as values with the following keys:
            - 'HF': a tuple of torch.Tensor that has the same length and shape as the output activations in vLLM. But each tensor is the postprocessed HuggingFace output activations to match the output activations in vLLM or copied from the original vLLM output activations if no matching HuggingFace output activations are found.
            - 'vLLM': a tuple of torch.Tensor that is the original vLLM output activations.
    '''
    # === DO NOT CHANGE THE CODE BELOW ===
    # load the activations
    # hf_model_embed_tokens_output: Tuple of tensors collected from the HF module's output
    hf_model_embed_tokens_output = torch.load('/path/to/activations/agent-training/gemma-2-2b-it/dec3/hf_activation_rank0_model_embed_tokens_output.pt', weights_only=False)

    # vllm_model_embed_tokens_output: Tuple of tensors collected from the vLLM module's output
    vllm_model_embed_tokens_output = torch.load('/path/to/activations/agent-training/gemma-2-2b-it/dec3/vllm_activation_rank0_model_embed_tokens_output.pt', weights_only=False)

    # create the result dict
    result = {"model.embed_tokens": {"HF": None, "vLLM": None},}
    # === DO NOT CHANGE THE CODE ABOVE ===
    
    # Process the embedding outputs
    # The loaded data is a list of lists. Access the actual tensor at [0][0]
    # HF shape: (1, 1067, 2304) -> need to squeeze to (1067, 2304) to match vLLM
    
    # Extract the actual tensor from the nested structure
    if isinstance(hf_model_embed_tokens_output, (list, tuple)):
        hf_embed_tensor = hf_model_embed_tokens_output[0][0]
    else:
        hf_embed_tensor = hf_model_embed_tokens_output
    
    # Squeeze the batch dimension (dimension 0) to match vLLM shape
    # HF shape: (1, 1067, 2304) -> vLLM shape: (1067, 2304)
    hf_embed_tensor_squeezed = hf_embed_tensor.squeeze(0)
    
    # Extract vLLM tensor from nested structure
    if isinstance(vllm_model_embed_tokens_output, (list, tuple)):
        vllm_embed_tensor = vllm_model_embed_tokens_output[0][0]
    else:
        vllm_embed_tensor = vllm_model_embed_tokens_output
    
    # Wrap in tuples to match the expected format
    result["model.embed_tokens"]["HF"] = (hf_embed_tensor_squeezed,)
    result["model.embed_tokens"]["vLLM"] = (vllm_embed_tensor,)
    
    # === DO NOT CHANGE THE CODE BELOW ===
    return result
    # === DO NOT CHANGE THE CODE ABOVE ===