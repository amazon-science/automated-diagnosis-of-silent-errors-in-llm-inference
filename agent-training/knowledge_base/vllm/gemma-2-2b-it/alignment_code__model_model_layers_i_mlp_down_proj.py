import torch
from typing import Dict, List

def postprocess_hf_activations() -> Dict[str, Dict[str, List[torch.Tensor]]]:
    ''' Post-process HuggingFace hidden states to match vLLM hidden states.
    
    Reasoning:
    - The HuggingFace Linear layer outputs a tuple containing a single tensor with shape (batch_size, seq_len, hidden_dim).
    - The vLLM implementation returns a tuple where the first element is the output tensor with shape (seq_len, hidden_dim)
      (batch dimension removed) and the second element is None.
    - We need to squeeze the batch dimension from HuggingFace output to match vLLM's shape.
    - Based on the activation snapshots, HF output is at [0][0] with shape (1, 1067, 2304) and vLLM output is at [0][0] with shape (1067, 2304).
    
    Output:
    - result: A dictionary containing vLLM component names as keys and a dictionary as values with the following keys:
            - 'HF': a tuple of torch.Tensor that has the same length and shape as the output activations in vLLM.
            - 'vLLM': a tuple of torch.Tensor that is the original vLLM output activations.
    '''
    # === DO NOT CHANGE THE CODE BELOW ===
    # load the activations
    # hf_model_layers_0_mlp_down_proj_output: Tuple of tensors collected from the HF module's output
    hf_model_layers_0_mlp_down_proj_output = torch.load('/path/to/activations/agent-training/gemma-2-2b-it/dec3/hf_activation_rank0_model_layers_0_mlp_down_proj_output.pt', weights_only=False)

    # vllm_model_layers_0_mlp_down_proj_output: Tuple of tensors collected from the vLLM module's output
    vllm_model_layers_0_mlp_down_proj_output = torch.load('/path/to/activations/agent-training/gemma-2-2b-it/dec3/vllm_activation_rank0_model_layers_0_mlp_down_proj_output.pt', weights_only=False)

    # create the result dict
    result = {"model.layers.0.mlp.down_proj": {"HF": None, "vLLM": None},}
    # === DO NOT CHANGE THE CODE ABOVE ===
    
    # Extract HuggingFace output tensor
    # HF structure: tuple of tuple -> [0][0] contains the tensor
    hf_output_tensor = hf_model_layers_0_mlp_down_proj_output[0][0]
    
    # Squeeze the batch dimension to match vLLM shape
    # HF shape: (1, 1067, 2304) -> vLLM shape: (1067, 2304)
    hf_output_tensor_squeezed = hf_output_tensor.squeeze(0)
    
    # Create the postprocessed HF tuple to match vLLM structure
    # vLLM returns a tuple with (tensor, None)
    hf_postprocessed = (hf_output_tensor_squeezed, None)
    
    # Set the results
    result["model.layers.0.mlp.down_proj"]["HF"] = hf_postprocessed
    result["model.layers.0.mlp.down_proj"]["vLLM"] = vllm_model_layers_0_mlp_down_proj_output[0]
    
    # === DO NOT CHANGE THE CODE BELOW ===
    return result
    # === DO NOT CHANGE THE CODE ABOVE ===