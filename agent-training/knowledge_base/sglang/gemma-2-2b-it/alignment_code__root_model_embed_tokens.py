import torch
from typing import Dict, List

def postprocess_hf_activations() -> Dict[str, Dict[str, List[torch.Tensor]]]:
    ''' Post-process HuggingFace hidden states to match vLLM hidden states.
    
    vLLM outputs a tuple of 2 tensors:
    - output[0]: shape (seq_len, hidden_dim) - all token embeddings without batch dimension
    - output[1]: shape (1, hidden_dim) - last token embedding with batch dimension
    
    HuggingFace outputs a tuple of 1 tensor:
    - output[0]: shape (1, seq_len, hidden_dim) - all token embeddings with batch dimension
    
    We need to:
    1. Extract the tensor from HF tuple and remove batch dimension to match vLLM output[0]
    2. Extract last position from HF output to match vLLM output[1]
    
    Output:
    - result: A dictionary containing vLLM component names as keys and a dictionary as values with the following keys:
            - 'HF': a tuple of torch.Tensor that has the same length and shape as the output activations in vLLM.
            - 'vLLM': a tuple of torch.Tensor that is the original vLLM output activations.
    '''
    # === DO NOT CHANGE THE CODE BELOW ===
    # load the activations
    # hf_model_embed_tokens_output: Tuple of tensors collected from the HF module's output
    hf_model_embed_tokens_output = torch.load('/path/to/activations/agent-training/gemma-2-2b-it/dec10/hf_activation_rank0_model_embed_tokens_output.pt', weights_only=False)

    # vllm_model_embed_tokens_output: Tuple of tensors collected from the vLLM module's output
    vllm_model_embed_tokens_output = torch.load('/path/to/activations/agent-training/gemma-2-2b-it/dec10/sglang_activation_rank0_model_embed_tokens_output.pt', weights_only=False)

    # create the result dict
    result = {"model.embed_tokens": {"HF": None, "vLLM": None},}
    # === DO NOT CHANGE THE CODE ABOVE ===
    
    # HF output is a tuple with 1 tensor
    # Extract the tensor from the tuple
    hf_embeddings = hf_model_embed_tokens_output[0][0]  # shape: (1, seq_len, hidden_dim)
    
    # Process to match vLLM format:
    # 1. Remove batch dimension for first output
    hf_output_0 = hf_embeddings.squeeze(0)  # shape: (seq_len, hidden_dim)
    
    # 2. Extract last position for second output (keep batch dimension)
    hf_output_1 = hf_embeddings[:, -1, :]  # shape: (1, hidden_dim)
    
    # Create tuple matching vLLM output structure
    hf_processed = (hf_output_0, hf_output_1)
    
    # Assign to result
    result["model.embed_tokens"]["HF"] = hf_processed
    result["model.embed_tokens"]["vLLM"] = vllm_model_embed_tokens_output[0]
    
    # === DO NOT CHANGE THE CODE BELOW ===
    return result
    # === DO NOT CHANGE THE CODE ABOVE ===