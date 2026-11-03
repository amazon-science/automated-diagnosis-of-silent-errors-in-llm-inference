import torch
from typing import Dict, List
# import any other necessary modules here
    
def postprocess_hf_activations() -> Dict[str, Dict[str, List[torch.Tensor]]]:
    ''' Post-process HuggingFace hidden states to match vLLM hidden states.
    
    Reasoning:
    - vLLM uses a fused gate_up_proj layer that combines gate_proj and up_proj
    - HF has separate gate_proj and up_proj layers
    - vLLM's gate_up_proj output shape is (1067, 4608) where 4608 = 2304 + 2304
    - HF's gate_proj output shape is (1, 1067, 2304) and up_proj is (1, 1067, 2304)
    - The loaded data structure is: list[tuple[tensor]] for both HF and vLLM
    - We need to:
      1. Extract tensors from the nested structure (list[0] -> tuple[0] -> tensor)
      2. Squeeze the batch dimension from HF tensors
      3. Concatenate gate_proj and up_proj along the last dimension
      4. Return both as tuples matching vLLM's output structure
    
    Output:
    - result: A dictionary containing vLLM component names as keys and a dictionary as values with the following keys:
            - 'HF': a tuple of torch.Tensor that has the same length and shape as the output activations in vLLM. But each tensor is the postprocessed HuggingFace output activations to match the output activations in vLLM or copied from the original vLLM output activations if no matching HuggingFace output activations are found.
            - 'vLLM': a tuple of torch.Tensor that is the original vLLM output activations.
    '''
    # === DO NOT CHANGE THE CODE BELOW ===
    # load the activations
    # hf_model_layers_0_mlp_gate_proj_output: Tuple of tensors collected from the HF module's output
    hf_model_layers_0_mlp_gate_proj_output = torch.load('/path/to/activations/agent-training/gemma-2-2b-it/dec3/hf_activation_rank0_model_layers_0_mlp_gate_proj_output.pt', weights_only=False)
    # hf_model_layers_0_mlp_up_proj_output: Tuple of tensors collected from the HF module's output
    hf_model_layers_0_mlp_up_proj_output = torch.load('/path/to/activations/agent-training/gemma-2-2b-it/dec3/hf_activation_rank0_model_layers_0_mlp_up_proj_output.pt', weights_only=False)

    # vllm_model_layers_0_mlp_gate_up_proj_output: Tuple of tensors collected from the vLLM module's output
    vllm_model_layers_0_mlp_gate_up_proj_output = torch.load('/path/to/activations/agent-training/gemma-2-2b-it/dec3/vllm_activation_rank0_model_layers_0_mlp_gate_up_proj_output.pt', weights_only=False)

    # create the result dict
    result = {"model.layers.0.mlp.gate_up_proj": {"HF": None, "vLLM": None},}
    # === DO NOT CHANGE THE CODE ABOVE ===
    
    hf_gate_proj = hf_model_layers_0_mlp_gate_proj_output[0][0]
    hf_up_proj = hf_model_layers_0_mlp_up_proj_output[0][0]
    
    hf_gate_proj_squeezed = hf_gate_proj.squeeze(0)
    hf_up_proj_squeezed = hf_up_proj.squeeze(0)
    
    hf_gate_up_proj = torch.cat([hf_gate_proj_squeezed, hf_up_proj_squeezed], dim=-1)
    
    vllm_output_tuple = vllm_model_layers_0_mlp_gate_up_proj_output[0]
    
    result["model.layers.0.mlp.gate_up_proj"]["HF"] = (hf_gate_up_proj,)
    result["model.layers.0.mlp.gate_up_proj"]["vLLM"] = vllm_output_tuple
    
    # === DO NOT CHANGE THE CODE BELOW ===
    return result
    # === DO NOT CHANGE THE CODE ABOVE ===