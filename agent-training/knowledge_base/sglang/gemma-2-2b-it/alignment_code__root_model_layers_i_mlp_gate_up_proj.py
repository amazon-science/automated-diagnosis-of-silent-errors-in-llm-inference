import torch
from typing import Dict, List, Tuple

def postprocess_hf_activations() -> Dict[str, Dict[str, List[torch.Tensor]]]:
    '''Post-process HuggingFace hidden states to match vLLM hidden states.
    
    The vLLM gate_up_proj module concatenates the gate_proj and up_proj outputs
    along the last dimension. HuggingFace keeps them separate.
    
    Output:
    - result: A dictionary containing vLLM component names as keys and a dictionary as values with the following keys:
            - 'HF': a tuple of torch.Tensor that has the same length and shape as the output activations in vLLM. 
                    Each tensor is the postprocessed HuggingFace output activations to match the output activations in vLLM.
            - 'vLLM': a tuple of torch.Tensor that is the original vLLM output activations.
    '''
    # === DO NOT CHANGE THE CODE BELOW ===
    # load the activations
    # hf_model_layers_0_mlp_gate_proj_output: Tuple of tensors collected from the HF module's output
    hf_model_layers_0_mlp_gate_proj_output = torch.load('/path/to/activations/agent-training/gemma-2-2b-it/dec10/hf_activation_rank0_model_layers_0_mlp_gate_proj_output.pt', weights_only=False)
    # hf_model_layers_0_mlp_up_proj_output: Tuple of tensors collected from the HF module's output
    hf_model_layers_0_mlp_up_proj_output = torch.load('/path/to/activations/agent-training/gemma-2-2b-it/dec10/hf_activation_rank0_model_layers_0_mlp_up_proj_output.pt', weights_only=False)

    # vllm_model_layers_0_mlp_gate_up_proj_output: Tuple of tensors collected from the vLLM module's output
    vllm_model_layers_0_mlp_gate_up_proj_output = torch.load('/path/to/activations/agent-training/gemma-2-2b-it/dec10/sglang_activation_rank0_model_layers_0_mlp_gate_up_proj_output.pt', weights_only=False)

    # create the result dict
    result = {"model.layers.0.mlp.gate_up_proj": {"HF": None, "vLLM": None},}
    # === DO NOT CHANGE THE CODE ABOVE ===
    
    # Extract the actual tensors from HuggingFace outputs
    hf_gate_tensor = hf_model_layers_0_mlp_gate_proj_output[0][0]
    hf_up_tensor = hf_model_layers_0_mlp_up_proj_output[0][0]
    
    # Remove batch dimension and concatenate along the last dimension
    hf_gate_squeezed = hf_gate_tensor.squeeze(0)
    hf_up_squeezed = hf_up_tensor.squeeze(0)
    
    # Concatenate gate and up projections
    hf_combined = torch.cat([hf_gate_squeezed, hf_up_squeezed], dim=-1)
    
    # Extract vLLM tensors
    vllm_tensor_0 = vllm_model_layers_0_mlp_gate_up_proj_output[0][0]
    vllm_tensor_1 = vllm_model_layers_0_mlp_gate_up_proj_output[1][0]
    
    # Create flat tuples of tensors
    hf_processed = (hf_combined, vllm_tensor_1)
    vllm_processed = (vllm_tensor_0, vllm_tensor_1)
    
    # Store the results
    result["model.layers.0.mlp.gate_up_proj"]["HF"] = hf_processed
    result["model.layers.0.mlp.gate_up_proj"]["vLLM"] = vllm_processed
    
    # === DO NOT CHANGE THE CODE BELOW ===
    return result
    # === DO NOT CHANGE THE CODE ABOVE ===