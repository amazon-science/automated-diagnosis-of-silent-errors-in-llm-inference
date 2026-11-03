import torch
from typing import Dict, List

def postprocess_hf_activations() -> Dict[str, Dict[str, List[torch.Tensor]]]:
    ''' Post-process HuggingFace hidden states to match vLLM hidden states.
    
    Reasoning:
    - HuggingFace's model.layers.0 outputs a tuple with one tensor of shape [1, 1067, 2304],
      which represents the final layer output (residual + hidden_states combined)
    - vLLM's model.layers.0 outputs a tuple with two tensors, each of shape [1067, 2304]:
      - vllm_tensor_0: residual connection (input to the layer)
      - vllm_tensor_1: hidden states (transformation output from the layer)
    - In Gemma-2 architecture, the final output is: residual + hidden_states
    - HF provides the final combined output (residual + hidden_states), while vLLM returns 
      both components separately
    - To properly align:
      1. Squeeze the batch dimension from HF output to get [1067, 2304]
      2. The squeezed HF tensor should match vLLM[0] + vLLM[1] (the combined output)
      3. For the HF tuple, we compute: (vLLM[0], HF_squeezed - vLLM[0]) to decompose 
         HF into its residual and hidden_states components
    - This way: HF[0] + HF[1] = vLLM[0] + vLLM[1] = HF_output
    
    Output:
    - result: A dictionary containing vLLM component names as keys and a dictionary as values with the following keys:
            - 'HF': a tuple of torch.Tensor that has the same length and shape as the output activations in vLLM
            - 'vLLM': a tuple of torch.Tensor that is the original vLLM output activations
    '''
    # === DO NOT CHANGE THE CODE BELOW ===
    # load the activations
    # hf_model_layers_0_output: Tuple of tensors collected from the HF module's output
    hf_model_layers_0_output = torch.load('/path/to/activations/agent-training/gemma-2-2b-it/dec3/hf_activation_rank0_model_layers_0_output.pt', weights_only=False)

    # vllm_model_layers_0_output: Tuple of tensors collected from the vLLM module's output
    vllm_model_layers_0_output = torch.load('/path/to/activations/agent-training/gemma-2-2b-it/dec3/vllm_activation_rank0_model_layers_0_output.pt', weights_only=False)

    # create the result dict
    result = {"model.layers.0": {"HF": None, "vLLM": None}}
    # === DO NOT CHANGE THE CODE ABOVE ===
    
    # Extract HuggingFace tensor from nested structure: ((tensor,),)
    hf_tensor = hf_model_layers_0_output[0][0]
    
    # Remove the batch dimension from HuggingFace output
    hf_tensor_squeezed = hf_tensor.squeeze(0)
    
    # Extract vLLM tensors from nested structure: ((tensor0, tensor1),)
    vllm_tensor_0 = vllm_model_layers_0_output[0][0]
    vllm_tensor_1 = vllm_model_layers_0_output[0][1]
    
    # Decompose HF output into residual and hidden_states components
    hf_residual = vllm_tensor_0
    hf_hidden_states = hf_tensor_squeezed - vllm_tensor_0
    
    # Create the HF output tuple to match vLLM structure
    hf_output_tuple = (hf_residual, hf_hidden_states)
    
    # vLLM output tuple remains as is
    vllm_output_tuple = (vllm_tensor_0, vllm_tensor_1)
    
    # Populate the result
    result["model.layers.0"]["HF"] = hf_output_tuple
    result["model.layers.0"]["vLLM"] = vllm_output_tuple
    
    # === DO NOT CHANGE THE CODE BELOW ===
    return result
    # === DO NOT CHANGE THE CODE ABOVE ===