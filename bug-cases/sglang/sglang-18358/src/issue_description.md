[Bug] DeepSeek OCR 2 Accuracy Issue #18358
Open
Open
[Bug] DeepSeek OCR 2 Accuracy Issue
#18358
@pansicheng
Description
pansicheng
opened 2 weeks ago
Collaborator
Checklist

I searched related issues but found no solution.

The bug persists in the latest version.

Issues without environment info and a minimal reproducible demo are hard to resolve and may receive no feedback.

If this is not a bug report but a general question, please start a discussion at https://github.com/sgl-project/sglang/discussions. Otherwise, it will be closed.

Please use English. Otherwise, it will be closed.
Describe the bug
The OCR results produced by sglang differ from the results produced by transformers

Reproduction
sglang

python -m sglang.launch_server   --model-path /path/to/DeepSeek-OCR-2/   --enable-multimodal   --host 0.0.0.0   --port 30000

curl -s http://localhost:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-ai/DeepSeek-OCR-2",
    "messages": [
      {
        "role": "user",
        "content": [
          { "type": "text", "text": "<image>\n<|grounding|>Convert the document to markdown." },
          { "type": "image_url", "image_url": { "url": "https://raw.githubusercontent.com/sgl-project/sgl-test-files/refs/heads/main/images/ocr-text.png" } }
        ]
      }
    ],
    "max_tokens": 512,
    "temperature": 0
  }'

{"id":"a506ffae636843bebff0ad46658d1994","object":"chat.completion","created":1770367630,"model":"deepseek-ai/DeepSeek-OCR-2","choices":[{"index":0,"message":{"role":"assistant",
"content":" 1. \"SGLang is a high-performance, high-performance, and high-performance, high-performance, and high-performance, high-performance, and high-performance, high-performance, and high-performance, high-performance, and high-performance, high-performance, and high-performance, high-performance, and high-performance, high-performance, and high-performance, high-performance, and high-performance, high-performance, and high-performance, high-performance, and high-performance, high-performance, and high-performance, high-performance, and high-performance, high-performance, and high-performance, high-performance, and high-performance, high-performance, and high-performance, high-performance, and high-performance, high-performance, and high-performance, high-performance, and high-performance, high-performance, and high-performance, high-performance, and high-performance, high-performance, and high-performance, high-performance, and high-performance, high-performance, and high-performance, high-performance, and high-performance, high-performance, and high-performance, high-performance, and high-performance, high-performance, and high-performance, high-performance, and high-performance, high-performance, and high-performance, high-performance, and high-performance, high-performance, and high-performance, high-performance, and high-performance, high-performance, and high-performance, high-performance, and high-performance, high-performance, and high-performance, high-performance, and high-performance, high-performance, and high-performance, high-performance, and high-performance, high-performance, and high-performance, high-performance, and high-performance, high-performance, and high-performance, high-performance, and high-performance, high-performance, and high-performance, high-performance, and high-performance, high-performance, and high-performance, high-performance, and high-performance, high-performance, and high-performance, high-performance, and high-performance, high-performance, and high-performance, high-performance, and high-performance, high-performance, and high-performance, high-performance, and high-performance, high-performance, and high-performance, high-performance, and high-performance, high-performance, and high-performance, high-performance, and high-performance, high-performance, and high-performance, high-performance, and high-performance, high-performance, and high-performance, high-performance, and high-performance, high-performance, and high-performance, high-performance, and high-performance, high-performance, and high-performance, high-performance, and high-performance, high-performance, and high-performance, high-performance, and high-performance, high-performance, and high-performance, high-performance, and high-performance, high-performance, and high-performance, high-performance, and high-performance, high-performance,",
"reasoning_content":null,"tool_calls":null},"logprobs":null,"finish_reason":"length","matched_stop":null}],"usage":{"prompt_tokens":272,"total_tokens":784,"completion_tokens":512,"prompt_tokens_details":null,"reasoning_tokens":0},"metadata":{"weight_version":"default"}}
transformers==4.46.3

from transformers import AutoModel, AutoTokenizer
import torch
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
model_name = '/path/to/DeepSeek-OCR-2'

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModel.from_pretrained(model_name, trust_remote_code=True, use_safetensors=True)
model = model.eval().cuda().to(torch.bfloat16)

prompt = "<image>\n<|grounding|>Convert the document to markdown."
image_file = '/path/to/ocr-text.png'
output_path = '/path/to/ocr-text.dir'

res = model.infer(tokenizer, prompt=prompt, image_file=image_file, output_path = output_path)

<|ref|>sub_title<|/ref|><|det|>[[10, 19, 528, 85]]<|/det|>
## SGLang Documentation

<|ref|>text<|/ref|><|det|>[[7, 120, 974, 249]]<|/det|>
SGLang is a high-performance serving framework for large language models and vision-language models. It is designed to deliver low-latency and high-throughput inference across a wide range of setups, from a single GPU to large distributed clusters. Its core features include:

<|ref|>text<|/ref|><|det|>[[30, 285, 983, 456]]<|/det|>
• Fast Backend Runtime: Provides efficient serving with RadixAttention for prefix caching, a zero-overhead CPU scheduler, prefill-decode disaggregation, speculative decoding, continuous batching, paged attention, tensor/pipeline/expert/data parallelism, structured outputs, chunked prefill, quantization (FP4/FP8/INT4/AWQ/GPTCQ), and multi-LoRA batching.

<|ref|>text<|/ref|><|det|>[[30, 471, 974, 641]]<|/det|>
• Extensive Model Support: Supports a wide range of generative models (Llama, Qwen, DeepSeek, Kimi, GLM, GPT, Gemma, Mistral, etc.), embedding models (e5-mistral, gte, mcdse), and reward models (Skywork), with easy extensibility for integrating new models. Compatible with most Hugging Face models and OpenAI APIs.

<|ref|>text<|/ref|><|det|>[[30, 656, 941, 738]]<|/det|>
• Extensive Hardware Support: Runs on NVIDIA GPUs (GB200/B300/H100/A100/Spark), AMD GPUs (MI355/MI300), Intel Xeon CPUs, Google TPUs, Ascend NPUs, and more.

<|ref|>text<|/ref|><|det|>[[30, 753, 968, 876]]<|/det|>
• Flexible Frontend Language: Offers an intuitive interface for programming LLM applications, supporting chained generation calls, advanced prompting, control flow, multi-modal inputs, parallelism, and external interactions.

<|ref|>text<|/ref|><|det|>[[30, 891, 950, 974]]<|/det|>
• Active Community: SGLang is open-source and supported by a vibrant community with widespread industry adoption, powering over 300,000 GPUs worldwide.