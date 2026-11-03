# Reproduction
## SGLang
repo and branch: https://github.com/IKACE/SGLang/tree/v0.5.8.post1-ported

### setup 
SGLANG_ACTIVATION_SAVE_PATH=$EKKA_ACTIVATION_DIR/bug-cases/sglang/sglang-18358  SGLANG_DISABLE_CUDNN_CHECK=1 python -m sglang.launch_server --model-path deepseek-ai/DeepSeek-OCR-2 --enable-multimodal --host 0.0.0.0 --port 30000 --disable-cuda-graph

### request
curl -s http://localhost:30000/v1/chat/completions   -H "Content-Type: application/json"   -d '{
    "model": "deepseek-ai/DeepSeek-OCR-2",
    "messages": [
      {
        "role": "user",
        "content": [
          { "type": "text", "text": "\n<|grounding|>Convert the document to markdown." },
          { "type": "image_url", "image_url": { "url": "https://raw.githubusercontent.com/sgl-project/sgl-test-files/refs/heads/main/images/ocr-text.png" } }
        ]
      }
    ],
    "max_tokens": 1,
    "temperature": 0
  }'


## vLLM
repo and branch: https://github.com/IKACE/vLLM/tree/v0.16.0rc1-ported

### setup

VLLM_ACTIVATION_SAVE_PATH=$EKKA_ACTIVATION_DIR/bug-cases/sglang/sglang-18358  vllm serve deepseek-ai/DeepSeek-OCR-2  --no-enable-prefix-caching --mm-processor-cache-gb 0 --enforce-eager

### request
curl -s http://localhost:8000/v1/chat/completions   -H "Content-Type: application/json"   -d '{
    "model": "deepseek-ai/DeepSeek-OCR-2",
    "messages": [
      {
        "role": "user",
        "content": [
          { "type": "text", "text": "\n<|grounding|>Convert the document to markdown." },
          { "type": "image_url", "image_url": { "url": "https://raw.githubusercontent.com/sgl-project/sgl-test-files/refs/heads/main/images/ocr-text.png" } }
        ]
      }
    ],
    "max_tokens": 1,
    "temperature": 0
  }'
