# [Bug]: gemma3-27b-it shows degraded accuracy in vLLM v0.11.0 #28539


Your current environment
Current environment
vLLM version: v0.11.0
docker image: vllm/vllm-openai:v0.11.0

The output of python collect_env.py
🐛 Describe the bug
I'm testing open-source models using vLLM v0.11.0 on Nvidia H20 and AMD Instinct MI308X using lm_eval==0.4.9.1 from lm-evaluation-harness. However, I found gemma-3-27b-it shows degraded accuracy compared to Hugging face transformers implementation, similar issue reported in #17689, PR21927 try to address sliding window configuration issue, but it seems does not resolve the original accuracy issue.

HF transformers 4.57.1 results

lm_eval --model hf --model_args pretrained=google/gemma-3-27b-it,dtype=bfloat16,trust_remote_code=True --tasks hellaswag,lambada_openai

Tasks	Version	Filter	n-shot	Metric	 	Value	 	Stderr
hellaswag	1	none	0	acc	↑	0.6503	±	0.0048
 	 	none	0	acc_norm	↑	0.8406	±	0.0037
lambada_openai	1	none	0	acc	↑	0.6953	±	0.0064
 	 	none	0	perplexity	↓	3.7726	±	0.1152
H20 + vLLM 0.11.0 results
lm_eval --model vllm --model_args pretrained=google/gemma-3-27b-it,tensor_parallel_size=2,dtype=bfloat16,gpu_memory_utilization=0.8,trust_remote_code=True,enforce_eager=True,max_gen_toks=256 --tasks hellaswag,lambada_openai

Tasks	Version	Filter	n-shot	Metric	 	Value	 	Stderr
hellaswag	1	none	0	acc	↑	0.3868	±	0.0049
 	 	none	0	acc_norm	↑	0.4858	±	0.0050
lambada_openai	1	none	0	acc	↑	0.2243	±	0.0058
 	 	none	0	perplexity	↓	960.5903	±	76.8435
AMD Instinct 308X Triton Flash Attention + vLLM 0.11.0 results
VLLM_ROCM_USE_AITER=1 VLLM_USE_TRITON_FLASH_ATTN=1 VLLM_ROCM_USE_AITER_MHA=0  VLLM_ROCM_USE_AITER_PAGED_ATTN=0 lm_eval --model vllm --model_args pretrained=google/gemma-3-27b-it,tensor_parallel_size=1,dtype=bfloat16,gpu_memory_utilization=0.8,trust_remote_code=True,enforce_eager=True,,max_gen_toks=256 --tasks hellaswag,lambada_openai

Tasks	Version	Filter	n-shot	Metric	 	Value	 	Stderr
hellaswag	1	none	0	acc	↑	0.3873	±	0.0049
 	 	none	0	acc_norm	↑	0.4858	±	0.0050
lambada_openai	1	none	0	acc	↑	0.2271	±	0.0058
 	 	none	0	perplexity	↓	950.2281	±	77.4239