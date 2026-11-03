import torch
from typing import Dict, List
# import any other necessary modules here
    
def postprocess_hf_activations() -> Dict[str, Dict[str, List[torch.Tensor]]]:
    ''' Post-process HuggingFace hidden states to match vLLM hidden states.
    
    Reasoning:
    - HF MLP output is a nested tuple structure where hf_output[0][0] is a tensor with shape [1, 926, 4096]
    - vLLM MLP output is a nested tuple structure where vllm_output[0][0] is a tensor with shape [926, 4096]
    - The batch dimension (dimension 0) in HF needs to be removed to match vLLM
    - We squeeze dimension 0 from the HF tensor to transform [1, 926, 4096] -> [926, 4096]
    - Both outputs are returned as tuples to match the expected format
    
    Output:
    - result: A dictionary containing vLLM component names as keys and a dictionary as values with the following keys:
            - 'HF': a tuple of torch.Tensor that has the same length and shape as the output activations in vLLM. But each tensor is the postprocessed HuggingFace output activations to match the output activations in vLLM or copied from the original vLLM output activations if no matching HuggingFace output activations are found.
            - 'vLLM': a tuple of torch.Tensor that is the original vLLM output activations.
    '''
    # === DO NOT CHANGE THE CODE BELOW ===
    # load the activations
    # hf_model_layers_0_mlp_output: Tuple of tensors collected from the HF module's output
    hf_model_layers_0_mlp_output = torch.load('/path/to/activations/agent-training/Llama-3.1-8B-Instruct/dec2/hf_activation_rank0_model_layers_0_mlp_output.pt', weights_only=False)

    # vllm_model_layers_0_mlp_output: Tuple of tensors collected from the vLLM module's output
    vllm_model_layers_0_mlp_output = torch.load('/path/to/activations/agent-training/Llama-3.1-8B-Instruct/dec2/vllm_activation_rank0_model_layers_0_mlp_output.pt', weights_only=False)

    # create the result dict
    result = {"model.layers.0.mlp": {"HF": None, "vLLM": None},}
    # === DO NOT CHANGE THE CODE ABOVE ===
    
    # Extract the actual tensor from the nested tuple structure and squeeze the batch dimension
    # hf_model_layers_0_mlp_output is a tuple, hf_model_layers_0_mlp_output[0] is also a tuple
    # hf_model_layers_0_mlp_output[0][0] is the actual tensor with shape [1, 926, 4096]
    hf_tensor = hf_model_layers_0_mlp_output[0][0].squeeze(0)
    
    # vllm_model_layers_0_mlp_output[0][0] is the actual tensor with shape [926, 4096]
    vllm_tensor = vllm_model_layers_0_mlp_output[0][0]
    
    # Create tuples with the processed tensors
    result["model.layers.0.mlp"]["HF"] = (hf_tensor,)
    result["model.layers.0.mlp"]["vLLM"] = (vllm_tensor,)
    
    # === DO NOT CHANGE THE CODE BELOW ===
    return result
    # === DO NOT CHANGE THE CODE ABOVE ===