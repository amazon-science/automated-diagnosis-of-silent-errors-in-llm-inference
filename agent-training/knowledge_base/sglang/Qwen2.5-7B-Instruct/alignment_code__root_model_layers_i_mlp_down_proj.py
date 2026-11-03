import torch
from typing import Dict, List

def postprocess_hf_activations() -> Dict[str, Dict[str, List[torch.Tensor]]]:
    ''' Post-process HuggingFace hidden states to match vLLM hidden states.
    Output:
    - result: A dictionary containing vLLM component names as keys and a dictionary as values with the following keys:
            - 'HF': a tuple of torch.Tensor that has the same length and shape as the output activations in vLLM. But each tensor is the postprocessed HuggingFace output activations to match the output activations in vLLM or copied from the original vLLM output activations if no matching HuggingFace output activations are found.
            - 'vLLM': a tuple of torch.Tensor that is the original vLLM output activations.
    '''
    # === DO NOT CHANGE THE CODE BELOW ===
    # load the activations
    # hf_model_layers_0_mlp_down_proj_output: Tuple of tensors collected from the HF module's output
    hf_model_layers_0_mlp_down_proj_output = torch.load('/path/to/activations/agent-training/Qwen2.5-7B-Instruct/dec10/hf_activation_rank0_model_layers_0_mlp_down_proj_output.pt', weights_only=False)

    # vllm_model_layers_0_mlp_down_proj_output: Tuple of tensors collected from the vLLM module's output
    vllm_model_layers_0_mlp_down_proj_output = torch.load('/path/to/activations/agent-training/Qwen2.5-7B-Instruct/dec10/sglang_activation_rank0_model_layers_0_mlp_down_proj_output.pt', weights_only=False)

    # create the result dict
    result = {"model.layers.0.mlp.down_proj": {"HF": None, "vLLM": None},}
    # === DO NOT CHANGE THE CODE ABOVE ===
    
    # HuggingFace output structure: (tensor[1, 1051, 3584],)
    # vLLM output structure: ((tensor[1051, 3584], None), (tensor[1, 3584], None))
    
    # Extract HF tensor and remove batch dimension
    hf_output_tensor = hf_model_layers_0_mlp_down_proj_output[0][0]
    hf_full_sequence = hf_output_tensor.squeeze(0)
    
    # Extract vLLM tensors
    vllm_tensor_0 = vllm_model_layers_0_mlp_down_proj_output[0][0]
    vllm_tensor_1 = vllm_model_layers_0_mlp_down_proj_output[1][0]
    
    # Create flat tuples of tensors
    hf_processed = (hf_full_sequence, vllm_tensor_1)
    vllm_processed = (vllm_tensor_0, vllm_tensor_1)
    
    # Store results
    result["model.layers.0.mlp.down_proj"]["HF"] = hf_processed
    result["model.layers.0.mlp.down_proj"]["vLLM"] = vllm_processed
    
    # === DO NOT CHANGE THE CODE BELOW ===
    return result
    # === DO NOT CHANGE THE CODE ABOVE ===