### Reproduction

server
vllm serve openai/whisper-large-v3-turbo \
--served-model-name whisper-large-v3-turbo \
--tensor-parallel-size 1 --gpu-memory-utilization 0.9 \
--host 0.0.0.0 --port 21002

client
curl http://localhost:21002/v1/audio/transcriptions \
  -H "Authorization: Bearer EMPTY" \
  -F file=@bug-cases/vllm/vllm-30777/src/tests_jfk.flac \
  -F model=whisper-large-v3-turbo \
  -F task=transcribe