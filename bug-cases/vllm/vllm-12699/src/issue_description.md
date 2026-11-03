# [Bug]: different logprobs for qwn2-vl when running on transformers and on vllm #12699

Your current environment
Im runnig a finetuned qwen2-vl with transformers and with vllm and getting significantly different logprobs.
vllm is running on runpod vllm-worker and the environment data is in the attached file:
pod_env.txt
tranformers env data:
Package Version



Model Input Dumps
No response

🐛 Describe the bug
prompt:
prompt.txt

vllm logprobs:
token 0:
token=True, logprob=-0.3891778588294983
token=False, logprob=-1.1391777992248535
token=Yes, logprob=-7.0141777992248535
token=true, logprob=-8.139178276062012
token=false, logprob=-8.514178276062012
token 1:
token=, logprob=-0.0007429938414134085
token=., logprob=-7.8757429122924805
token= False, logprob=-9.00074291229248
token=,, logprob=-9.87574291229248
token= True, logprob=-9.93824291229248

transformers logprobs:
Logprob: -0.6362
token id: 4049
Top Logprobs: {'False': -0.6361945271492004, 'True': -0.7611945271492004, 'Yes': -6.886194705963135, 'false': -7.636194705963135, 'true': -7.886194705963135}

Logprob: -0.0007
token id: 151645
Top Logprobs: {'<|im_end|>': -0.0007362039177678525, '.': -7.750736236572266, '\n': -9.000736236572266, '\n\n': -9.375736236572266, ',': -10.750736236572266}

When I insert prompts without images im getting similar logprobs so I think there is something wrong with the vision encoding.

Before submitting a new issue...

Make sure you already searched for relevant issues, and asked the chatbot living at the bottom right corner of the documentation page, which can answer lots of frequently asked questions.