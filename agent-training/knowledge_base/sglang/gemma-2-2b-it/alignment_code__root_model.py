import torch
from typing import Dict, List
# import any other necessary modules here
    
def postprocess_hf_activations() -> Dict[str, Dict[str, List[torch.Tensor]]]:
    ''' Post-process HuggingFace hidden states to match vLLM hidden states.
    
    Reasoning:
    - HuggingFace model output is a tuple where the first element [0] contains a dictionary
      with 'last_hidden_state' (shape: 1, 1067, 2304) and 'past_key_values'
    - vLLM model output is a tuple with 2 elements:
      - vllm_model_output[0][0]: shape (1067, 2304) - hidden states without batch dim
      - vllm_model_output[1][0]: shape (1, 2304) - additional output tensor
    - Alignment strategy:
      1. HF's last_hidden_state (1, 1067, 2304) aligns with vLLM[0][0] (1067, 2304)
         -> Squeeze batch dimension to match
      2. vLLM[1][0] (1, 2304) has no clear HF counterpart
         -> Return original vLLM tensor on the same device as HF
    - Device alignment: Ensure all tensors are on the same device (cuda:0 from HF)
    
    Output:
    - result: A dictionary containing vLLM component names as keys and a dictionary as values with the following keys:
            - 'HF': a tuple of torch.Tensor that has the same length and shape as the output activations in vLLM. But each tensor is the postprocessed HuggingFace output activations to match the output activations in vLLM or copied from the original vLLM output activations if no matching HuggingFace output activations are found.
            - 'vLLM': a tuple of torch.Tensor that is the original vLLM output activations.
    '''
    # === DO NOT CHANGE THE CODE BELOW ===
    # load the activations
    # hf_model_output: Tuple of tensors collected from the HF module's output
    hf_model_output = torch.load('/path/to/activations/agent-training/gemma-2-2b-it/dec10/hf_activation_rank0_model_output.pt', weights_only=False)

    # vllm_model_output: Tuple of tensors collected from the vLLM module's output
    vllm_model_output = torch.load('/path/to/activations/agent-training/gemma-2-2b-it/dec10/sglang_activation_rank0_model_output.pt', weights_only=False)

    # create the result dict
    result = {"model": {"HF": None, "vLLM": None},}
    # === DO NOT CHANGE THE CODE ABOVE ===
    
    # Extract HuggingFace's last_hidden_state from the nested structure
    # Structure: hf_model_output[0][0]['last_hidden_state']
    hf_last_hidden_state = hf_model_output[0][0]['last_hidden_state']
    
    # Remove batch dimension to match vLLM format
    # HF: (1, 1067, 2304) -> (1067, 2304) to match vllm_model_output[0][0]
    hf_hidden_state_squeezed = hf_last_hidden_state.squeeze(0)
    
    # Get vLLM's outputs and move them to the same device as HF tensors
    vllm_first_output = vllm_model_output[0][0].to(hf_hidden_state_squeezed.device)
    vllm_second_output = vllm_model_output[1][0].to(hf_hidden_state_squeezed.device)
    
    # Create the result tuple for HF (matching vLLM structure with 2 tensors)
    # First tensor: processed HF hidden state
    # Second tensor: vLLM second output (no HF counterpart, so we use vLLM's)
    result["model"]["HF"] = (hf_hidden_state_squeezed, vllm_second_output)
    
    # Keep vLLM outputs (ensuring they're on the same device)
    result["model"]["vLLM"] = (vllm_first_output, vllm_second_output)
    
    # === DO NOT CHANGE THE CODE BELOW ===
    return result
    # === DO NOT CHANGE THE CODE ABOVE ===