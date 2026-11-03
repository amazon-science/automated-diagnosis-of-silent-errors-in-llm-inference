import torch
from typing import Dict, List

def postprocess_hf_activations() -> Dict[str, Dict[str, List[torch.Tensor]]]:
    '''Post-process HuggingFace hidden states to match vLLM hidden states.
    
    Output:
    - result: A dictionary containing vLLM component names as keys and a dictionary as values with the following keys:
            - 'HF': a tuple of torch.Tensor that has the same length and shape as the output activations in vLLM.
            - 'vLLM': a tuple of torch.Tensor that is the original vLLM output activations.
    '''
    # === DO NOT CHANGE THE CODE BELOW ===
    # load the activations
    # hf_model_embed_tokens_output: Tuple of tensors collected from the HF module's output
    hf_model_embed_tokens_output = torch.load('/path/to/activations/agent-training/Qwen2.5-7B-Instruct/dec10/hf_activation_rank0_model_embed_tokens_output.pt', weights_only=False)

    # vllm_model_embed_tokens_output: Tuple of tensors collected from the vLLM module's output
    vllm_model_embed_tokens_output = torch.load('/path/to/activations/agent-training/Qwen2.5-7B-Instruct/dec10/sglang_activation_rank0_model_embed_tokens_output.pt', weights_only=False)

    # create the result dict
    result = {"model.embed_tokens": {"HF": None, "vLLM": None},}
    # === DO NOT CHANGE THE CODE ABOVE ===
    
    hf_tensor = hf_model_embed_tokens_output[0][0]
    hf_embedding = hf_tensor.squeeze(0)
    
    vllm_tensor_0 = vllm_model_embed_tokens_output[0][0]
    vllm_tensor_1 = vllm_model_embed_tokens_output[1][0]
    
    result["model.embed_tokens"]["HF"] = (hf_embedding, vllm_tensor_1)
    result["model.embed_tokens"]["vLLM"] = (vllm_tensor_0, vllm_tensor_1)
    
    # === DO NOT CHANGE THE CODE BELOW ===
    return result
    # === DO NOT CHANGE THE CODE ABOVE ===