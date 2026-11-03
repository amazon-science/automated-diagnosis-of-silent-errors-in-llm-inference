import torch
from typing import Dict, List

def postprocess_hf_activations() -> Dict[str, Dict[str, List[torch.Tensor]]]:
    ''' Post-process HuggingFace hidden states to match vLLM hidden states.
    
    The key difference is that HuggingFace's Linear layer output includes a batch dimension
    (shape: [1, 926, 4096]) while vLLM's output does not (shape: [926, 4096]). 
    We need to squeeze the batch dimension from HF output.
    
    The outputs are stored as nested tuples: loaded_data[0] contains a tuple of tensors,
    where loaded_data[0][0] is the actual output tensor from the Linear layer.
    
    Output:
    - result: A dictionary containing vLLM component names as keys and a dictionary as values with the following keys:
            - 'HF': a tuple of torch.Tensor that has the same length and shape as the output activations in vLLM. 
                    Each tensor is the postprocessed HuggingFace output activations to match the output activations in vLLM.
            - 'vLLM': a tuple of torch.Tensor that is the original vLLM output activations.
    '''
    # === DO NOT CHANGE THE CODE BELOW ===
    # load the activations
    # hf_model_layers_0_mlp_down_proj_output: Tuple of tensors collected from the HF module's output
    hf_model_layers_0_mlp_down_proj_output = torch.load('/path/to/activations/agent-training/Llama-3.1-8B-Instruct/dec2/hf_activation_rank0_model_layers_0_mlp_down_proj_output.pt', weights_only=False)

    # vllm_model_layers_0_mlp_down_proj_output: Tuple of tensors collected from the vLLM module's output
    vllm_model_layers_0_mlp_down_proj_output = torch.load('/path/to/activations/agent-training/Llama-3.1-8B-Instruct/dec2/vllm_activation_rank0_model_layers_0_mlp_down_proj_output.pt', weights_only=False)

    # create the result dict
    result = {"model.layers.0.mlp.down_proj": {"HF": None, "vLLM": None},}
    # === DO NOT CHANGE THE CODE ABOVE ===
    
    # Access the actual tensors from the nested tuple structure
    # The structure is: loaded_data[0] is a tuple, loaded_data[0][0] is the tensor
    hf_output_tensor = hf_model_layers_0_mlp_down_proj_output[0][0]
    
    # HF shape: (1, 926, 4096) - includes batch dimension
    # vLLM shape: (926, 4096) - no batch dimension
    # Solution: squeeze the batch dimension from HF output
    hf_output_squeezed = hf_output_tensor.squeeze(0)
    
    # Pack into tuples matching vLLM's output structure
    result["model.layers.0.mlp.down_proj"]["HF"] = (hf_output_squeezed,)
    result["model.layers.0.mlp.down_proj"]["vLLM"] = vllm_model_layers_0_mlp_down_proj_output[0]
    
    # === DO NOT CHANGE THE CODE BELOW ===
    return result
    # === DO NOT CHANGE THE CODE ABOVE ===