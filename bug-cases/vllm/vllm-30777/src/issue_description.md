[Bug]: whisper-large-v3-turbo have accuracy problem on nightly build #30777
Open
Open
[Bug]: whisper-large-v3-turbo have accuracy problem on nightly build
#30777
@cyysky
Description
cyysky
opened 3 weeks ago
Your current environment
The output of python collect_env.py
🐛 Describe the bug
vllm serve whisper-large-v3-turbo --served-model-name stt_model --gpu-memory-utilization 0.035 --tensor-parallel-size 4 --port 9801

the transcribe output not accurate

switch back to 0.12.0 then output okay.

Before submitting a new issue...

Make sure you already searched for relevant issues, and asked the chatbot living at the bottom right corner of the documentation page, which can answer lots of frequently asked questions.