# Original Issue

[Bug]: TP4 Accuracy Worse Than TP8 for LLaMa4

Your current environment
N/A

🐛 Describe the bug
We are aware of Scout performing worse on MMLU Pro, GSM8K on TP4 and are actively debugging those.

Before submitting a new issue...

Make sure you already searched for relevant issues, and asked the chatbot living at the bottom right corner of the documentation page, which can answer lots of frequently asked questions.

# Reproducible Example

The issue can be reproduced by running the following counter-example where the HuggingFace and vLLM output under the same random seed and configuration are significantly different.

```
    model_id = "meta-llama/Llama-4-Scout-17B-16E-Instruct"
    llm = LLM(
        model=model_id,
        tensor_parallel_size=4,     # ← shards across 8 ranks automatically
        # tensor_parallel_size=2,     # ← shards across 8 ranks automatically
        dtype="bfloat16",           # or "float16" for older GPUs
        enforce_eager=True,       # disable cuda graph
        gpu_memory_utilization=0.9,  # 90% GPU memory utilization
        max_num_seqs=16,              # max number of sequences per batch
        max_model_len=2048,           # max model length
    )



    sampling_params = SamplingParams(
        temperature=0.0,            # greedy
        top_p=1.0,
        top_k=-1,
        max_tokens=50,
    )
    prompt = "The University of Washington is"
    prompts = [prompt]


    # vLLM returns a list[List[RequestOutput]]
    outputs = llm.generate(prompts, sampling_params)
    # print("Generated text:", [o.outputs[0].text for o in outputs])
    print("Generated text:", [prompt + o.outputs[0].text for prompt, o in zip(prompts, outputs)])
```

## Result
HF TP=4 output
['The University of Washington is a public research university located in Seattle, Washington. It is one of the largest universities on the West Coast and is known for its academic excellence, innovative research, and stunning natural surroundings. Here are some fun facts about the University of Washington:\n\n1.']

vLLM TP=4 output
[' a public research university located in Seattle, Washington. The university was founded in 1861 and has a long history of academic excellence. Here are some key facts about the University of Washington:\n \n \n\n1. Academics: The University of Washington offers']
