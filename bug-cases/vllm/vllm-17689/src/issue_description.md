# Original Issue

[Bug]: gemma3 shows degraded accuracy in vLLM v0.8.4 #17689

our current environment
vLLM version: v0.8.4
docker image: nvcr.io/nvidia/pytorch/23.10-py3
install script
pip install -r ./requirements/build.txt
pip install -r ./requirements/common.txt
pip install -r ./requirements/cuda.txt
pip install flash_attn==2.7.4.post1
export VLLM_COMMIT=dc1b4a6f1300003ae27f033afbdff5e2683721ce
export VLLM_PRECOMPILED_WHEEL_LOCATION=https://wheels.vllm.ai/${VLLM_COMMIT}/vllm-1.0.0.dev-cp38-abi3-manylinux1_x86_64.whl
pip install -e .
pip install -U pynvml
Click to expand environment info
🐛 Describe the bug
I'm testing open-source models using vLLM v0.8.4 and lm-evaluation-harness. However, this version shows degraded accuracy when deploying the Gemma 3 series. There were no issues when testing the Qwen 2.5 model. I'm comparing results from vLLM against those from Hugging Face.

Qwen2.5-72B-Instruct
vLLM result
Tasks	Version	Filter	n-shot	Metric		Value		Stderr
hellaswag	1	none	0	acc	↑	0.7042	±	0.0046
none	0	acc_norm	↑	0.8741	±	0.0033
lambada_openai	1	none	0	acc	↑	0.7613	±	0.0059
none	0	perplexity	↓	2.7697	±	0.0560
HF result
gen_kwargs: (None), limit: None, num_fewshot: None, batch_size: 4

Tasks	Version	Filter	n-shot	Metric		Value		Stderr
hellaswag	1	none	0	acc	↑	0.7031	±	0.0046
none	0	acc_norm	↑	0.8735	±	0.0033
lambada_openai	1	none	0	acc	↑	0.7518	±	0.0060
none	0	perplexity	↓	2.7677	±	0.0559
gemma-3-27b-it
vLLM result
Tasks	Version	Filter	n-shot	Metric		Value		Stderr
hellaswag	1	none	0	acc	↑	0.3863	±	0.0049
none	0	acc_norm	↑	0.4862	±	0.0050
lambada_openai	1	none	0	acc	↑	0.2251	±	0.0058
none	0	perplexity	↓	956.6927	±	77.7910
HF resuit
Tasks	Version	Filter	n-shot	Metric		Value		Stderr
hellaswag	1	none	0	acc	↑	0.6508	±	0.0048
none	0	acc_norm	↑	0.8405	±	0.0037
lambada_openai	1	none	0	acc	↑	0.6942	±	0.0064
none	0	perplexity	↓	3.7705	±	0.1152
gemma-3-12b-it
vLLM result
Tasks	Version	Filter	n-shot	Metric		Value		Stderr
hellaswag	1	none	0	acc	↑	0.3351	±	0.0047
none	0	acc_norm	↑	0.3999	±	0.0049
lambada_openai	1	none	0	acc	↑	0.2424	±	0.0060
none	0	perplexity	↓	7298.6355	±	885.6837
HF result
Tasks	Version	Filter	n-shot	Metric		Value		Stderr
hellaswag	1	none	0	acc	↑	0.6285	±	0.0048
none	0	acc_norm	↑	0.8187	±	0.0038
lambada_openai	1	none	0	acc	↑	0.6775	±	0.0065
none	0	perplexity	↓	4.1643	±	0.1289
Before submitting a new issue...

Make sure you already searched for relevant issues, and asked the chatbot living at the bottom right corner of the documentation page, which can answer lots of frequently asked questions.