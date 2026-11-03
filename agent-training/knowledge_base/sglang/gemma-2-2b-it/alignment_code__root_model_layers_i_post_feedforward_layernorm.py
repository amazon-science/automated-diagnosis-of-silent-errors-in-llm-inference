import torch
from typing import Dict, List, Tuple
# import any other necessary modules here
    
def postprocess_hf_activations() -> Dict[str, Dict[str, Tuple[torch.Tensor, ...]]]:
    ''' Post-process HuggingFace hidden states to match vLLM hidden states.
    
    Reasoning:
    - HuggingFace GemmaRMSNorm output structure: ((tensor of shape [1, 1067, 2304],),)
      This is a nested tuple with a single tensor that has a batch dimension.
    - vLLM GemmaRMSNorm output structure: ((tensor1 of shape [1067, 2304],), (tensor2 of shape [1, 2304],))
      This is a tuple with two elements, each containing a tensor.
    - However, the return format should be flat tuples of tensors: (tensor1, tensor2, ...)
    
    Processing steps:
    1. Extract HF's tensor and squeeze batch dimension [1, 1067, 2304] -> [1067, 2304] 
       to match vLLM's first output tensor
    2. For vLLM's second output [1, 2304], there's no matching HF tensor, 
       so we use vLLM's tensor as-is
    3. Return flat tuples: HF should be (squeezed_hf_tensor, vllm_second_tensor)
       and vLLM should be (vllm_first_tensor, vllm_second_tensor)
    
    Output:
    - result: A dictionary containing vLLM component names as keys and a dictionary as values with the following keys:
            - 'HF': a tuple of torch.Tensor that has the same length and shape as the output activations in vLLM.
            - 'vLLM': a tuple of torch.Tensor that is the original vLLM output activations.
    '''
    # === DO NOT CHANGE THE CODE BELOW ===
    # load the activations
    # hf_model_layers_0_post_feedforward_layernorm_output: Tuple of tensors collected from the HF module's output
    hf_model_layers_0_post_feedforward_layernorm_output = torch.load('/path/to/activations/agent-training/gemma-2-2b-it/dec10/hf_activation_rank0_model_layers_0_post_feedforward_layernorm_output.pt', weights_only=False)

    # vllm_model_layers_0_post_feedforward_layernorm_output: Tuple of tensors collected from the vLLM module's output
    vllm_model_layers_0_post_feedforward_layernorm_output = torch.load('/path/to/activations/agent-training/gemma-2-2b-it/dec10/sglang_activation_rank0_model_layers_0_post_feedforward_layernorm_output.pt', weights_only=False)

    # create the result dict
    result = {"model.layers.0.post_feedforward_layernorm": {"HF": None, "vLLM": None},}
    # === DO NOT CHANGE THE CODE ABOVE ===
    
    # Extract the tensor from the nested tuple structure in HF output
    # HF output structure: ((tensor,),) -> extract tensor at [0][0]
    hf_tensor = hf_model_layers_0_post_feedforward_layernorm_output[0][0]
    
    # Squeeze the batch dimension to match vLLM's first output shape
    # HF: [1, 1067, 2304] -> [1067, 2304]
    hf_normalized = hf_tensor.squeeze(0)
    
    # Extract vLLM's two output tensors from nested tuple structure
    # vLLM output structure: ((tensor1,), (tensor2,)) 
    vllm_first_tensor = vllm_model_layers_0_post_feedforward_layernorm_output[0][0]  # [1067, 2304]
    vllm_second_tensor = vllm_model_layers_0_post_feedforward_layernorm_output[1][0]  # [1, 2304]
    
    # Create flat tuples of tensors (not nested tuples)
    # HF tuple: (squeezed HF output to match vLLM's first output, vLLM's second output as-is)
    # vLLM tuple: (vLLM's first output, vLLM's second output)
    result["model.layers.0.post_feedforward_layernorm"]["HF"] = (hf_normalized, vllm_second_tensor)
    result["model.layers.0.post_feedforward_layernorm"]["vLLM"] = (vllm_first_tensor, vllm_second_tensor)
    
    # === DO NOT CHANGE THE CODE BELOW ===
    return result
    # === DO NOT CHANGE THE CODE ABOVE ===