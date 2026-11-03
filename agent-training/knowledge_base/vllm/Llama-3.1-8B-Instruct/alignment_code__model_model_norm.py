import torch
from typing import Dict, List
    
def postprocess_hf_activations() -> Dict[str, Dict[str, List[torch.Tensor]]]:
    ''' Post-process HuggingFace hidden states to match vLLM hidden states.
    
    The HuggingFace model.norm outputs a tuple with one tensor of shape [batch, seq_len, hidden_dim].
    The vLLM model.norm outputs a tuple with two tensors of shape [seq_len, hidden_dim]:
        - First tensor: normalized output (corresponds to HF output after squeezing batch dim)
        - Second tensor: residual connection that vLLM explicitly returns
    
    Since HuggingFace doesn't return the residual explicitly, we copy it from vLLM.
    
    Output:
    - result: A dictionary containing vLLM component names as keys and a dictionary as values with the following keys:
            - 'HF': a tuple of torch.Tensor that has the same length and shape as the output activations in vLLM. But each tensor is the postprocessed HuggingFace output activations to match the output activations in vLLM or copied from the original vLLM output activations if no matching HuggingFace output activations are found.
            - 'vLLM': a tuple of torch.Tensor that is the original vLLM output activations.
    '''
    # === DO NOT CHANGE THE CODE BELOW ===
    # load the activations
    # hf_model_norm_output: Tuple of tensors collected from the HF module's output
    hf_model_norm_output = torch.load('/path/to/activations/agent-training/Llama-3.1-8B-Instruct/dec2/hf_activation_rank0_model_norm_output.pt', weights_only=False)

    # vllm_model_norm_output: Tuple of tensors collected from the vLLM module's output
    vllm_model_norm_output = torch.load('/path/to/activations/agent-training/Llama-3.1-8B-Instruct/dec2/vllm_activation_rank0_model_norm_output.pt', weights_only=False)

    # create the result dict
    result = {"model.norm": {"HF": None, "vLLM": None},}
    # === DO NOT CHANGE THE CODE ABOVE ===
    
    # HuggingFace outputs: tuple with 1 tensor of shape [1, 926, 4096]
    # vLLM outputs: tuple with 2 tensors of shape [926, 4096] each
    
    # Extract the first element from the HF tuple and squeeze the batch dimension
    hf_normalized = hf_model_norm_output[0][0].squeeze(0)  # [1, 926, 4096] -> [926, 4096]
    
    # vLLM returns (normalized_output, residual)
    # HF only returns normalized_output, so we need to copy the residual from vLLM
    vllm_normalized = vllm_model_norm_output[0][0]  # [926, 4096]
    vllm_residual = vllm_model_norm_output[0][1]    # [926, 4096]
    
    # Create the matched HF output tuple
    hf_matched = (hf_normalized, vllm_residual)
    
    # Set the result
    result["model.norm"]["HF"] = hf_matched
    result["model.norm"]["vLLM"] = vllm_model_norm_output[0]
    
    # === DO NOT CHANGE THE CODE BELOW ===
    return result
    # === DO NOT CHANGE THE CODE ABOVE ===