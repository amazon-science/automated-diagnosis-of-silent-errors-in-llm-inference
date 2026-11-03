# Reproduction
issue link: https://github.com/vllm-project/vllm/issues/17689
fix link: https://github.com/vllm-project/vllm/pull/21927
## HF

### single request
Generated text: <bos>The University of Washington is committed to providing a safe and inclusive environment for all members of our community. We recognize that individuals may experience harassment, discrimination, or other forms of misconduct.

**Reporting Options:**

*   **UW Office of Human Resources (OHR):**
Logits collected: 50
Example logits shape: torch.Size([262208])

### single request 2, from lm-eval


### lm-eval
2025-09-29:08:36:05 INFO     [loggers.evaluation_tracker:209] Saving results aggregated
vllm (pretrained=google/gemma-3-12b-it,tensor_parallel_size=4,gpu_memory_utilization=0.75,enforce_eager=True,trust_remote_code=True,max_gen_toks=256), gen_kwargs: (None), limit: None, num_fewshot: None, batch_size: 256
|    Tasks     |Version|Filter|n-shot|  Metric  |   |  Value  |   | Stderr |
|--------------|------:|------|-----:|----------|---|--------:|---|-------:|
|hellaswag     |      1|none  |     0|acc       |↑  |   0.3360|±  |  0.0047|
|              |       |none  |     0|acc_norm  |↑  |   0.3990|±  |  0.0049|
|lambada_openai|      1|none  |     0|acc       |↑  |   0.2467|±  |  0.0060|
|              |       |none  |     0|perplexity|↓  |7110.0414|±  |855.1537|


hf (pretrained=google/gemma-3-12b-it,parallelize=True), gen_kwargs: (None), limit: None, num_fewshot: None, batch_size: 16
|  Tasks  |Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr|
|---------|------:|----------------|-----:|-----------|---|-----:|---|------|
|tinyGSM8k|      0|flexible-extract|     5|exact_match|↑  |0.8689|±  |   N/A|
|         |       |strict-match    |     5|exact_match|↑  |0.8527|±  |   N/A|
### lm-eval float32
hf (pretrained=google/gemma-3-12b-it,parallelize=True,dtype=float32), gen_kwargs: (None), limit: None, num_fewshot: None, batch_size: 32
|  Tasks  |Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr|
|---------|------:|----------------|-----:|-----------|---|-----:|---|------|
|tinyGSM8k|      0|flexible-extract|     5|exact_match|↑  |0.8792|±  |   N/A|
|         |       |strict-match    |     5|exact_match|↑  |0.8467|±  |   N/A|

## vLLM

### single request
Generated text: ['The University of Washington is committed to providing a safe and inclusive environment for all members of our community. We recognize that individuals may experience harassment, discrimination, or other forms of misconduct.\n\n**Reporting Options:**\n\n*   **UW Office of Human Resources (OHR):**']

### single request 2, from lm-eval
#### HF TP1
**1. Calculate the cost of the subs:**

* Cost per sub: $7.50
* Number of subs: 2
* Total cost of subs: $7.50 * 2 = $15.00

**2. Calculate the cost of the chips:**

* Cost per bag of chips: $1.50
* Number of bags of chips: 2
* Total cost of chips: $1.50 * 2 = $3.00

**3. Calculate the cost of the cookies:**

* Cost per cookie: $1.00
* Number of cookies: 2
* Total cost of cookies: $1.00 * 2 = $2.00

**4. Calculate the subtotal (cost of all items before delivery fee and tip):**

* Total cost of subs + total cost of chips + total cost of cookies = $15.00 + $3.00 + $2.00 = $20.00

**5. Calculate the delivery fee:**

* Delivery fee percentage: 20%
* Subtotal: $20.00
* Delivery fee amount:

#### HF TP2
**1. Calculate the cost of the subs:**

* Cost per sub: $7.50
* Number of subs: 2
* Total cost of subs: $7.50 * 2 = $15.00

**2. Calculate the cost of the chips:**

* Cost per bag of chips: $1.50
* Number of bags of chips: 2
* Total cost of chips: $1.50 * 2 = $3.00

**3. Calculate the cost of the cookies:**

* Cost per cookie: $1.00
* Number of cookies: 2
* Total cost of cookies: $1.00 * 2 = $2.00

**4. Calculate the subtotal (cost of all items before delivery fee and tip):**

* Total cost of subs + total cost of chips + total cost of cookies = $15.00 + $3.00 + $2.00 = $20.00

**5. Calculate the delivery fee:**

* Delivery fee percentage: 20%
* Subtotal: $20.00
* Delivery fee amount:

#### HF TP4
**1. Calculate the cost of the subs:**

* 2 subs * $7.50/sub = $15.00

**2. Calculate the cost of the chips:**

* 2 bags of chips * $1.50/bag = $3.00

**3. Calculate the cost of the cookies:**

* 2 cookies * $1.00/cookie = $2.00

**4. Calculate the subtotal (cost of all items before fees):**

* $15.00 + $3.00 + $2.00 = $20.00

**5. Calculate the delivery fee:**

* $20.00 * 0.20 = $4.00

**6. Calculate the total cost before the tip:**

* $20.00 + $4.00 = $24.00

**7. Calculate the final cost with the tip:**

* $24.00 + $5.00 = $29.00

**Answer:** Rory's delivery order will cost $29.00.

#### HF TP8

**1. Calculate the cost of the subs:**

* 2 subs * $7.50/sub = $15.00

**2. Calculate the cost of the chips:**

* 2 bags of chips * $1.50/bag = $3.00

**3. Calculate the cost of the cookies:**

* 2 cookies * $1.00/cookie = $2.00

**4. Calculate the subtotal (cost of all items before fees):**

* $15.00 + $3.00 + $2.00 = $20.00

**5. Calculate the delivery fee:**

* $20.00 * 0.20 = $4.00

**6. Calculate the total cost before the tip:**

* $20.00 + $4.00 = $24.00

**7. Calculate the final cost with the tip:**

* $24.00 + $5.00 = $29.00

**Answer:** Rory's delivery order will cost $29.00.

#### HF TP2 float32

**1. Calculate the cost of the subs:**

* 2 subs * $7.50/sub = $15.00

**2. Calculate the cost of the chips:**

* 2 bags of chips * $1.50/bag = $3.00

**3. Calculate the cost of the cookies:**

* 2 cookies * $1.00/cookie = $2.00

**4. Calculate the subtotal (cost of all items before fees):**

* $15.00 + $3.00 + $2.00 = $20.00

**5. Calculate the delivery fee:**

* $20.00 * 0.20 = $4.00

**6. Calculate the total cost before the tip:**

* $20.00 + $4.00 = $24.00

**7. Calculate the final cost with the tip:**

* $24.00 + $5.00 = $29.00

**Therefore, Rory's delivery order will cost $29.00.**

#### HF TP4 float32
**1. Calculate the cost of the subs:**

* 2 subs * $7.50/sub = $15.00

**2. Calculate the cost of the chips:**

* 2 bags of chips * $1.50/bag = $3.00

**3. Calculate the cost of the cookies:**

* 2 cookies * $1.00/cookie = $2.00

**4. Calculate the subtotal (cost of all items before fees):**

* $15.00 + $3.00 + $2.00 = $20.00

**5. Calculate the delivery fee:**

* $20.00 * 0.20 = $4.00

**6. Calculate the total cost before the tip:**

* $20.00 + $4.00 = $24.00

**7. Calculate the final cost with the tip:**

* $24.00 + $5.00 = $29.00

**Therefore, Rory's delivery order will cost $29.00.**

#### HF TP8 float32
**1. Calculate the cost of the subs:**

* 2 subs * $7.50/sub = $15.00

**2. Calculate the cost of the chips:**

* 2 bags of chips * $1.50/bag = $3.00

**3. Calculate the cost of the cookies:**

* 2 cookies * $1.00/cookie = $2.00

**4. Calculate the subtotal (cost of all items before fees):**

* $15.00 + $3.00 + $2.00 = $20.00

**5. Calculate the delivery fee:**

* $20.00 * 0.20 = $4.00

**6. Calculate the total cost before the tip:**

* $20.00 + $4.00 = $24.00

**7. Calculate the final cost with the tip:**

* $24.00 + $5.00 = $29.00

**Therefore, Rory's delivery order will cost $29.00.**


#### vLLM TP1
**1. Calculate the cost of the subs:**\n\n* 2 subs * $7.50/sub = $15.00\n\n**2. Calculate the cost of the chips:**\n\n* 2 bags of chips * $1.50/bag = $3.00\n\n**3. Calculate the cost of the cookies:**\n\n* 2 cookies * $1.00/cookie = $2.00\n\n**4. Calculate the subtotal (cost of all items before fees):**\n\n* $15.00 + $3.00 + $2.00 = $20.00\n\n**5. Calculate the delivery fee:**\n\n* $20.00 * 0.20 = $4.00\n\n**6. Calculate the total cost before the tip:**\n\n* $20.00 + $4.00 = $24.00\n\n**7. Calculate the final cost with the tip:**\n\n* $24.00 + $5.00 = $29.00\n\n**Therefore, Rory's delivery order will cost $29.00.**\n

#### vLLM TP2
**1. Calculate the cost of the subs:**\n\n* 2 subs * $7.50/sub = $15.00\n\n**2. Calculate the cost of the chips:**\n\n* 2 bags of chips * $1.50/bag = $3.00\n\n**3. Calculate the cost of the cookies:**\n\n* 2 cookies * $1.00/cookie = $2.00\n\n**4. Calculate the subtotal (cost of all items before fees):**\n\n* $15.00 + $3.00 + $2.00 = $20.00\n\n**5. Calculate the delivery fee:**\n\n* $20.00 * 0.20 = $4.00\n\n**6. Calculate the total cost before the tip:**\n\n* $20.00 + $4.00 = $24.00\n\n**7. Calculate the final cost with the tip:**\n\n* $24.00 + $5.00 = $29.00\n\n**Therefore, Rory's delivery order will cost $29.00.**\n

#### vLLM TP4
**1. Calculate the cost of the subs:**\n\n* Cost per sub: $7.50\n* Number of subs: 2\n* Total cost of subs: $7.50 * 2 = $15.00\n\n**2. Calculate the cost of the chips:**\n\n* Cost per bag of chips: $1.50\n* Number of bags of chips: 2\n* Total cost of chips: $1.50 * 2 = $3.00\n\n**3. Calculate the cost of the cookies:**\n\n* Cost per cookie: $1.00\n* Number of cookies: 2\n* Total cost of cookies: $1.00 * 2 = $2.00\n\n**4. Calculate the subtotal (cost of all items before delivery fee and tip):**\n\n* Total cost of subs + total cost of chips + total cost of cookies = $15.00 + $3.00 + $2.00 = $20.00\n\n**5. Calculate the delivery fee:**\n\n* Delivery fee percentage: 20%\n* Subtotal: $20.00\n* Delivery fee amount:

#### vLLM TP8

**1. Calculate the cost of the subs:**\n\n* 2 subs * $7.50/sub = $15.00\n\n**2. Calculate the cost of the chips:**\n\n* 2 bags of chips * $1.50/bag = $3.00\n\n**3. Calculate the cost of the cookies:**\n\n* 2 cookies * $1.00/cookie = $2.00\n\n**4. Calculate the subtotal (cost of all items before fees):**\n\n* $15.00 + $3.00 + $2.00 = $20.00\n\n**5. Calculate the delivery fee:**\n\n* $20.00 * 0.20 = $4.00\n\n**6. Calculate the total cost before the tip:**\n\n* $20.00 + $4.00 = $24.00\n\n**7. Calculate the final cost with the tip:**\n\n* $24.00 + $5.00 = $29.00\n\n**Answer:** Rory's delivery order will cost $29.00.

#### vLLM TP2 float32

**1. Calculate the cost of the subs:**\n\n* 2 subs * $7.50/sub = $15.00\n\n**2. Calculate the cost of the chips:**\n\n* 2 bags of chips * $1.50/bag = $3.00\n\n**3. Calculate the cost of the cookies:**\n\n* 2 cookies * $1.00/cookie = $2.00\n\n**4. Calculate the subtotal (cost of all items before fees):**\n\n* $15.00 + $3.00 + $2.00 = $20.00\n\n**5. Calculate the delivery fee:**\n\n* $20.00 * 0.20 = $4.00\n\n**6. Calculate the total cost before the tip:**\n\n* $20.00 + $4.00 = $24.00\n\n**7. Calculate the final cost with the tip:**\n\n* $24.00 + $5.00 = $29.00\n\n**Therefore, Rory's delivery order will cost $29.00.**\n

#### vLLM TP4 float32

**1. Calculate the cost of the subs:**\n\n* 2 subs * $7.50/sub = $15.00\n\n**2. Calculate the cost of the chips:**\n\n* 2 bags of chips * $1.50/bag = $3.00\n\n**3. Calculate the cost of the cookies:**\n\n* 2 cookies * $1.00/cookie = $2.00\n\n**4. Calculate the subtotal (cost of all items before fees):**\n\n* $15.00 + $3.00 + $2.00 = $20.00\n\n**5. Calculate the delivery fee:**\n\n* $20.00 * 0.20 = $4.00\n\n**6. Calculate the total cost before the tip:**\n\n* $20.00 + $4.00 = $24.00\n\n**7. Calculate the final cost with the tip:**\n\n* $24.00 + $5.00 = $29.00\n\n**Therefore, Rory's delivery order will cost $29.00.**\n

#### vLLM TP8 float32

**1. Calculate the cost of the subs:**\n\n* 2 subs * $7.50/sub = $15.00\n\n**2. Calculate the cost of the chips:**\n\n* 2 bags of chips * $1.50/bag = $3.00\n\n**3. Calculate the cost of the cookies:**\n\n* 2 cookies * $1.00/cookie = $2.00\n\n**4. Calculate the subtotal (cost of all items before fees):**\n\n* $15.00 + $3.00 + $2.00 = $20.00\n\n**5. Calculate the delivery fee:**\n\n* $20.00 * 0.20 = $4.00\n\n**6. Calculate the total cost before the tip:**\n\n* $20.00 + $4.00 = $24.00\n\n**7. Calculate the final cost with the tip:**\n\n* $24.00 + $5.00 = $29.00\n\n**Therefore, Rory's delivery order will cost $29.00.**\n

### lm-eval

2025-09-29:08:57:40 INFO     [loggers.evaluation_tracker:209] Saving results aggregated
hf (pretrained=google/gemma-3-12b-it,parallelize=True), gen_kwargs: (None), limit: None, num_fewshot: None, batch_size: 16
|    Tasks     |Version|Filter|n-shot|  Metric  |   |Value |   |Stderr|
|--------------|------:|------|-----:|----------|---|-----:|---|-----:|
|hellaswag     |      1|none  |     0|acc       |↑  |0.6272|±  |0.0048|
|              |       |none  |     0|acc_norm  |↑  |0.8196|±  |0.0038|
|lambada_openai|      1|none  |     0|acc       |↑  |0.6788|±  |0.0065|
|              |       |none  |     0|perplexity|↓  |4.1621|±  |0.1288|


fatal: not a git repository (or any of the parent directories): .git
2025-09-29:09:34:03 INFO     [loggers.evaluation_tracker:209] Saving results aggregated
2025-09-29:09:34:03 INFO     [loggers.evaluation_tracker:298] Saving per-sample results for: tinyGSM8k
vllm (pretrained=google/gemma-3-12b-it,tensor_parallel_size=4,gpu_memory_utilization=0.75,enforce_eager=True,trust_remote_code=True,max_gen_toks=256), gen_kwargs: (None), limit: None, num_fewshot: None, batch_size: 256
|  Tasks  |Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr|
|---------|------:|----------------|-----:|-----------|---|-----:|---|------|
|tinyGSM8k|      0|flexible-extract|     5|exact_match|↑  |0.7687|±  |   N/A|
|         |       |strict-match    |     5|exact_match|↑  |0.7687|±  |   N/A|

### lm-eval float32

25-09-30:17:22:20 INFO     [loggers.evaluation_tracker:209] Saving results aggregated
2025-09-30:17:22:20 INFO     [loggers.evaluation_tracker:298] Saving per-sample results for: tinyGSM8k
vllm (pretrained=google/gemma-3-12b-it,tensor_parallel_size=8,gpu_memory_utilization=0.9,enforce_eager=True,trust_remote_code=True,max_gen_toks=256,max_num_seqs=16,max_model_len=2048,dtype=float32), gen_kwargs: (None), limit: None, num_fewshot: None, batch_size: 32
|  Tasks  |Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr|
|---------|------:|----------------|-----:|-----------|---|-----:|---|------|
|tinyGSM8k|      0|flexible-extract|     5|exact_match|↑  |0.6388|±  |   N/A|
|         |       |strict-match    |     5|exact_match|↑  |0.6551|±  |   N/A|