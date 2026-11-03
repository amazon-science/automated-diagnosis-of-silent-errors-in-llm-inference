import torch
from typing import Dict, List
# import any other necessary modules here
    
def postprocess_hf_activations() -> Dict[str, Dict[str, List[torch.Tensor]]]:
    ''' Post-process HuggingFace hidden states to match vLLM hidden states.
    Output:
    - result: A dictionary containing vLLM component names as keys and a dictionary as values with the following keys:
            - 'HF': a tuple of torch.Tensor that has the same length and shape as the output activations in vLLM. But each tensor is the postprocessed HuggingFace output activations to match the output activations in vLLM or copied from the original vLLM output activations if no matching HuggingFace output activations are found.
            - 'vLLM': a tuple of torch.Tensor that is the original vLLM output activations.
    
    Reasoning:
    - vLLM returns a tuple with two tensors:
      1. First tensor (1051, 3584): Full sequence hidden states without batch dimension
      2. Second tensor (1, 3584): A separately computed output (not necessarily the last token)
    - HuggingFace returns a tuple containing a dictionary with 'last_hidden_state' of shape (1, 1051, 3584)
    - Strategy:
      1. Match HF's full hidden states (squeezed) to vLLM's first output
      2. For vLLM's second output, use the vLLM tensor directly since there's no corresponding HF computation
    '''
    # === DO NOT CHANGE THE CODE BELOW ===
    # load the activations
    # hf_model_output: Tuple of tensors collected from the HF module's output
    hf_model_output = torch.load('/path/to/activations/agent-training/Qwen2.5-7B-Instruct/dec10/hf_activation_rank0_model_output.pt', weights_only=False)

    # vllm_model_output: Tuple of tensors collected from the vLLM module's output
    vllm_model_output = torch.load('/path/to/activations/agent-training/Qwen2.5-7B-Instruct/dec10/sglang_activation_rank0_model_output.pt', weights_only=False)

    # create the result dict
    result = {"model": {"HF": None, "vLLM": None},}
    # === DO NOT CHANGE THE CODE ABOVE ===
    
    # Extract HuggingFace last_hidden_state from the output dictionary
    hf_last_hidden_state = hf_model_output[0][0]['last_hidden_state']  # shape: (1, 1051, 3584)
    
    # Extract vLLM outputs directly
    vllm_tensor_0 = vllm_model_output[0][0]  # shape: (1051, 3584)
    vllm_tensor_1 = vllm_model_output[1][0]  # shape: (1, 3584)
    
    # Get the device of vLLM tensors
    vllm_device = vllm_tensor_0.device
    
    # Remove batch dimension to match vLLM's first output
    hf_hidden_states_no_batch = hf_last_hidden_state.squeeze(0).to(vllm_device)  # shape: (1051, 3584)
    
    # For the second output, since there's no matching HF computation and it appears to be 
    # computed differently in vLLM, use the vLLM tensor directly
    hf_second_output = vllm_tensor_1.clone()  # Use vLLM's value directly
    
    # Create the HF tuple matching vLLM's structure
    hf_processed = (hf_hidden_states_no_batch, hf_second_output)
    
    # Create vLLM tuple
    vllm_processed = (vllm_tensor_0, vllm_tensor_1)
    
    # Store in result dictionary
    result["model"]["HF"] = hf_processed
    result["model"]["vLLM"] = vllm_processed
    
    # === DO NOT CHANGE THE CODE BELOW ===
    return result
    # === DO NOT CHANGE THE CODE ABOVE ===