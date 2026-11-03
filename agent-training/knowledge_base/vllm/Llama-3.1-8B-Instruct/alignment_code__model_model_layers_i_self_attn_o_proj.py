import torch
from typing import Dict, List

def postprocess_hf_activations() -> Dict[str, Dict[str, List[torch.Tensor]]]:
    ''' Post-process HuggingFace hidden states to match vLLM hidden states.
    
    The HuggingFace o_proj Linear layer outputs a tensor with shape (1, 926, 4096) 
    that includes a batch dimension. The vLLM o_proj Linear layer outputs a tensor 
    with shape (926, 4096) without the batch dimension. We squeeze the batch 
    dimension from HuggingFace to match vLLM.
    
    Both frameworks capture the Linear layer's output using hooks. The Linear forward 
    method returns F.linear(input, self.weight, self.bias), which is a single tensor.
    
    Output:
    - result: A dictionary containing vLLM component names as keys and a dictionary as values with the following keys:
            - 'HF': a tuple of torch.Tensor that has the same length and shape as the output activations in vLLM. But each tensor is the postprocessed HuggingFace output activations to match the output activations in vLLM or copied from the original vLLM output activations if no matching HuggingFace output activations are found.
            - 'vLLM': a tuple of torch.Tensor that is the original vLLM output activations.
    '''
    # === DO NOT CHANGE THE CODE BELOW ===
    # load the activations
    # hf_model_layers_0_self_attn_o_proj_output: Tuple of tensors collected from the HF module's output
    hf_model_layers_0_self_attn_o_proj_output = torch.load('/path/to/activations/agent-training/Llama-3.1-8B-Instruct/dec2/hf_activation_rank0_model_layers_0_self_attn_o_proj_output.pt', weights_only=False)

    # vllm_model_layers_0_self_attn_o_proj_output: Tuple of tensors collected from the vLLM module's output
    vllm_model_layers_0_self_attn_o_proj_output = torch.load('/path/to/activations/agent-training/Llama-3.1-8B-Instruct/dec2/vllm_activation_rank0_model_layers_0_self_attn_o_proj_output.pt', weights_only=False)

    # create the result dict
    result = {"model.layers.0.self_attn.o_proj": {"HF": None, "vLLM": None},}
    # === DO NOT CHANGE THE CODE ABOVE ===
    
    # Helper function to extract tensors from nested structures
    def extract_tensors(data):
        """Recursively extract tensors from nested lists/tuples"""
        if isinstance(data, torch.Tensor):
            return [data]
        elif isinstance(data, (list, tuple)):
            tensors = []
            for item in data:
                tensors.extend(extract_tensors(item))
            return tensors
        elif data is None:
            return [None]
        else:
            return [data]
    
    # Extract all tensors from HF and vLLM outputs
    hf_tensors = extract_tensors(hf_model_layers_0_self_attn_o_proj_output)
    vllm_tensors = extract_tensors(vllm_model_layers_0_self_attn_o_proj_output)
    
    # Process HF tensors: squeeze batch dimension if present
    processed_hf_tensors = []
    for tensor in hf_tensors:
        if isinstance(tensor, torch.Tensor):
            # Check if tensor has shape [1, seq_len, hidden_size] and squeeze first dim
            if tensor.ndim == 3 and tensor.shape[0] == 1:
                processed_hf_tensors.append(tensor.squeeze(0))
            else:
                processed_hf_tensors.append(tensor)
        else:
            # Keep non-tensor items (e.g., None) as-is
            processed_hf_tensors.append(tensor)
    
    # Convert to tuples for final output
    result["model.layers.0.self_attn.o_proj"]["HF"] = tuple(processed_hf_tensors)
    result["model.layers.0.self_attn.o_proj"]["vLLM"] = tuple(vllm_tensors)
    
    # === DO NOT CHANGE THE CODE BELOW ===
    return result
    # === DO NOT CHANGE THE CODE ABOVE ===