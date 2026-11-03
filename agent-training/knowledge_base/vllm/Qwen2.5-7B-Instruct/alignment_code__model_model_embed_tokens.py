import torch
from typing import Dict, List

def postprocess_hf_activations() -> Dict[str, Dict[str, List[torch.Tensor]]]:
    ''' Post-process HuggingFace hidden states to match vLLM hidden states.
    
    The main difference is that HuggingFace embedding output includes a batch dimension [batch_size, seq_len, hidden_dim]
    while vLLM embedding output does not have a batch dimension [seq_len, hidden_dim].
    This function removes the batch dimension from HuggingFace activations to align with vLLM.
    
    The loaded activations are tuples that may contain nested structures. We extract the actual tensors
    and process them by removing the batch dimension from HuggingFace tensors.
    
    Output:
    - result: A dictionary containing vLLM component names as keys and a dictionary as values with the following keys:
            - 'HF': a tuple of torch.Tensor that has the same length and shape as the output activations in vLLM. 
                   Each tensor is the postprocessed HuggingFace output activations to match the output activations in vLLM 
                   or copied from the original vLLM output activations if no matching HuggingFace output activations are found.
            - 'vLLM': a tuple of torch.Tensor that is the original vLLM output activations.
    '''
    # === DO NOT CHANGE THE CODE BELOW ===
    # load the activations
    # hf_model_embed_tokens_output: Tuple of tensors collected from the HF module's output
    hf_model_embed_tokens_output = torch.load('/path/to/activations/agent-training/Qwen2.5-7B-Instruct/dec2/hf_activation_rank0_model_embed_tokens_output.pt', weights_only=False)

    # vllm_model_embed_tokens_output: Tuple of tensors collected from the vLLM module's output
    vllm_model_embed_tokens_output = torch.load('/path/to/activations/agent-training/Qwen2.5-7B-Instruct/dec2/vllm_activation_rank0_model_embed_tokens_output.pt', weights_only=False)

    # create the result dict
    result = {"model.embed_tokens": {"HF": None, "vLLM": None},}
    # === DO NOT CHANGE THE CODE ABOVE ===
    
    # Helper function to recursively extract tensors and process them
    def extract_and_process_tensors(data, remove_batch_dim=False):
        """Recursively extract tensors from nested tuples/lists and optionally remove batch dimension."""
        if isinstance(data, torch.Tensor):
            # Process tensor: remove batch dimension if needed
            if remove_batch_dim and data.dim() == 3 and data.shape[0] == 1:
                return data.squeeze(0)
            return data
        elif isinstance(data, (tuple, list)):
            # Recursively process each element
            processed = [extract_and_process_tensors(item, remove_batch_dim) for item in data]
            # If all items are tensors, return as tuple
            if all(isinstance(item, torch.Tensor) for item in processed):
                return tuple(processed)
            # Otherwise flatten the structure
            flattened = []
            for item in processed:
                if isinstance(item, (tuple, list)):
                    flattened.extend(item)
                else:
                    flattened.append(item)
            return tuple(flattened)
        else:
            return data
    
    # Process HuggingFace activations (remove batch dimension)
    hf_processed = extract_and_process_tensors(hf_model_embed_tokens_output, remove_batch_dim=True)
    
    # Process vLLM activations (keep as is)
    vllm_processed = extract_and_process_tensors(vllm_model_embed_tokens_output, remove_batch_dim=False)
    
    # Ensure both are tuples
    if not isinstance(hf_processed, tuple):
        hf_processed = (hf_processed,)
    if not isinstance(vllm_processed, tuple):
        vllm_processed = (vllm_processed,)
    
    # Store the processed activations
    result["model.embed_tokens"]["HF"] = hf_processed
    result["model.embed_tokens"]["vLLM"] = vllm_processed
    
    # === DO NOT CHANGE THE CODE BELOW ===
    return result
    # === DO NOT CHANGE THE CODE ABOVE ===