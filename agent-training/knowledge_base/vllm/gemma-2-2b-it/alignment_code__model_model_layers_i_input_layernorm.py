import torch
from typing import Dict, List
# import any other necessary modules here
    
def postprocess_hf_activations() -> Dict[str, Dict[str, List[torch.Tensor]]]:
    ''' Post-process HuggingFace hidden states to match vLLM hidden states.
    
    The HuggingFace RMSNorm output is a tuple containing a tensor with shape [1, seq_len, hidden_size]
    while vLLM RMSNorm output is a tuple containing a tensor with shape [seq_len, hidden_size].
    We need to squeeze the batch dimension (dimension 0) from the HuggingFace output tensor.
    
    Output:
    - result: A dictionary containing vLLM component names as keys and a dictionary as values with the following keys:
            - 'HF': a tuple of torch.Tensor that has the same length and shape as the output activations in vLLM. But each tensor is the postprocessed HuggingFace output activations to match the output activations in vLLM or copied from the original vLLM output activations if no matching HuggingFace output activations are found.
            - 'vLLM': a tuple of torch.Tensor that is the original vLLM output activations.
    '''
    # === DO NOT CHANGE THE CODE BELOW ===
    # load the activations
    # hf_model_layers_0_input_layernorm_output: Tuple of tensors collected from the HF module's output
    hf_model_layers_0_input_layernorm_output = torch.load('/path/to/activations/agent-training/gemma-2-2b-it/dec3/hf_activation_rank0_model_layers_0_input_layernorm_output.pt', weights_only=False)

    # vllm_model_layers_0_input_layernorm_output: Tuple of tensors collected from the vLLM module's output
    vllm_model_layers_0_input_layernorm_output = torch.load('/path/to/activations/agent-training/gemma-2-2b-it/dec3/vllm_activation_rank0_model_layers_0_input_layernorm_output.pt', weights_only=False)

    # create the result dict
    result = {"model.layers.0.input_layernorm": {"HF": None, "vLLM": None},}
    # === DO NOT CHANGE THE CODE ABOVE ===
    
    def extract_and_process_tensors(activation_tuple):
        """Recursively extract tensors from nested tuple structure and squeeze batch dimension."""
        processed = []
        
        def flatten(item):
            if isinstance(item, torch.Tensor):
                if item.dim() == 3 and item.shape[0] == 1:
                    processed.append(item.squeeze(0))
                else:
                    processed.append(item)
            elif isinstance(item, (tuple, list)):
                for sub_item in item:
                    flatten(sub_item)
        
        flatten(activation_tuple)
        return tuple(processed)
    
    def extract_tensors_no_squeeze(activation_tuple):
        """Recursively extract tensors from nested tuple structure without squeezing."""
        processed = []
        
        def flatten(item):
            if isinstance(item, torch.Tensor):
                processed.append(item)
            elif isinstance(item, (tuple, list)):
                for sub_item in item:
                    flatten(sub_item)
        
        flatten(activation_tuple)
        return tuple(processed)
    
    hf_output_postprocessed = extract_and_process_tensors(hf_model_layers_0_input_layernorm_output)
    vllm_output_postprocessed = extract_tensors_no_squeeze(vllm_model_layers_0_input_layernorm_output)
    
    result["model.layers.0.input_layernorm"]["HF"] = hf_output_postprocessed
    result["model.layers.0.input_layernorm"]["vLLM"] = vllm_output_postprocessed
    
    # === DO NOT CHANGE THE CODE BELOW ===
    return result
    # === DO NOT CHANGE THE CODE ABOVE ===