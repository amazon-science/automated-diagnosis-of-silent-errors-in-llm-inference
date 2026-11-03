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
    - HF output[0][0] has shape (1, 1067, 2304) which corresponds to vLLM output[0][0] with shape (1067, 2304)
      We need to squeeze/remove the batch dimension.
    - vLLM output[1][0] has shape (1, 2304) which appears to be the last token from the sequence.
      We extract this from HF by taking [:, -1:, :] to keep the batch dimension for the last token.
    '''
    # === DO NOT CHANGE THE CODE BELOW ===
    # load the activations
    # hf_model_layers_0_self_attn_output: Tuple of tensors collected from the HF module's output
    hf_model_layers_0_self_attn_output = torch.load('/path/to/activations/agent-training/gemma-2-2b-it/dec10/hf_activation_rank0_model_layers_0_self_attn_output.pt', weights_only=False)

    # vllm_model_layers_0_self_attn_output: Tuple of tensors collected from the vLLM module's output
    vllm_model_layers_0_self_attn_output = torch.load('/path/to/activations/agent-training/gemma-2-2b-it/dec10/sglang_activation_rank0_model_layers_0_self_attn_output.pt', weights_only=False)

    # create the result dict
    result = {"model.layers.0.self_attn": {"HF": None, "vLLM": None},}
    # === DO NOT CHANGE THE CODE ABOVE ===
    
    # Extract HF activations
    hf_attn_output = hf_model_layers_0_self_attn_output[0][0]  # shape: (1, 1067, 2304)
    
    # Process to match vLLM format:
    # vLLM output[0][0]: full sequence without batch dim
    hf_processed_full = hf_attn_output.squeeze(0)  # shape: (1067, 2304)
    
    # vLLM output[1][0]: last token with batch dim
    hf_processed_last = hf_attn_output[:, -1:, :].squeeze(1)  # shape: (1, 2304)
    
    # Create the result tuple for HF
    hf_processed = (hf_processed_full, hf_processed_last)
    
    # Assign to result
    result["model.layers.0.self_attn"]["HF"] = hf_processed
    result["model.layers.0.self_attn"]["vLLM"] = vllm_model_layers_0_self_attn_output[0]
    
    # === DO NOT CHANGE THE CODE BELOW ===
    return result
    # === DO NOT CHANGE THE CODE ABOVE ===