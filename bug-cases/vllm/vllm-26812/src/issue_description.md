# [Bug]: Inconsistent Results with Seed Setting in Zamba2 #26812

Your current environment
The output of python collect_env.py
🐛 Describe the bug
Despite setting a seed (seed=42) in the script, the results are still not reproducible in a stable manner. After running the code, the outputs from vLLM Zamba2 differ significantly on each execution, even though the environment setup, model files, and other parameters remain constant.

Test script about reproducing the inconsistent result.

Error messages:

INFO 10-14 22:09:48 [parallel_state.py:1208] rank 0 in world size 1 is assigned as DP rank 0, PP rank 0, TP rank 0, EP rank 0
y1: tensor([[[0., 0., 0.],
         [0., 0., 0.],
         [0., 0., 0.]]], device='cuda:0', dtype=torch.bfloat16)
y2: tensor([[[-0.0234,  0.1328, -0.2090],
         [ 0.5391,  0.1357, -0.2002],
         [-0.1875, -0.0513, -0.0018]]], device='cuda:0', dtype=torch.bfloat16)
{'device': 'cuda', 'dtype': 'bfloat16', 'same': False, 'max_abs_diff': 1.8671875}
Besides, I made a pull request that addresses this issue. If there's anything I should be aware of, I would appreciate your feedback.

Before submitting a new issue...

Make sure you already searched for relevant issues, and asked the chatbot living at the bottom right corner of the documentation page, which can answer lots of frequently asked questions.



