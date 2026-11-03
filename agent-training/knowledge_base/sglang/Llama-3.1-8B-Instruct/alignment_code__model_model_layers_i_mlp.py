import torch
from typing import Dict, List

def postprocess_hf_activations() -> Dict[str, Dict[str, List[torch.Tensor]]]:
    ''' Post-process HuggingFace hidden states to match vLLM hidden states.
    
    The HF MLP output is a nested tuple: ((tensor[1, 926, 4096],),)
    The vLLM MLP output is a nested tuple: ((tensor[926, 4096],), (tensor[1, 4096],))
    
    We extract the HF tensor and squeeze the batch dimension to match the first vLLM tensor.
    The second vLLM tensor (residual) has no clear HF correspondence, so we use the vLLM tensor directly.
    
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
    vllm_model_layers_0_mlp_output = torch.load('/path/to/activations/agent-training/Llama-3.1-8B-Instruct/dec2/sglang_activation_rank0_model_layers_0_mlp_output.pt', weights_only=False)

    # create the result dict
    result = {"model.layers.0.mlp": {"HF": None, "vLLM": None},}
    # === DO NOT CHANGE THE CODE ABOVE ===
    
    # Extract the actual tensor from nested tuple structure
    # HF: ((tensor[1, 926, 4096],),) -> extract tensor and squeeze batch dim
    hf_tensor = hf_model_layers_0_mlp_output[0][0]
    hf_output_squeezed = hf_tensor.squeeze(0)
    
    # vLLM has 2 output tensors in nested tuple structure
    # Extract them: ((tensor[926, 4096],), (tensor[1, 4096],))
    vllm_tensor_0 = vllm_model_layers_0_mlp_output[0][0]
    vllm_tensor_1 = vllm_model_layers_0_mlp_output[1][0]
    
    # Match HF output[0] to vLLM output[0], use vLLM output[1] as-is (no HF correspondence)
    result["model.layers.0.mlp"]["HF"] = (hf_output_squeezed, vllm_tensor_1)
    result["model.layers.0.mlp"]["vLLM"] = (vllm_tensor_0, vllm_tensor_1)
    
    # === DO NOT CHANGE THE CODE BELOW ===
    return result
    # === DO NOT CHANGE THE CODE ABOVE ===