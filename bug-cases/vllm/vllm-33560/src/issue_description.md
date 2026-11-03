Bug]: lm-eval shows significant accuracy differences on RedHatAI/Qwen3-8B-NVFP4 model (Turing vs. Ampere) #33560
Open
#33972
Open
[Bug]: lm-eval shows significant accuracy differences on RedHatAI/Qwen3-8B-NVFP4 model (Turing vs. Ampere)
#33560
#33972
@ir1ka
Description
ir1ka
opened last month · edited by ir1ka
Contributor
Your current environment
The output of python collect_env.py [Turing]
The output of python collect_env.py [Ampere]
🐛 Describe the bug
Run follow lm-eval command on both Turing and Ampere:

lm_eval --model vllm --model_args pretrained=RedHatAI/Qwen3-8B-NVFP4,tensor_parallel_size=2,gpu_memory_utilization=0.8,dtype=auto,attention_backend=FLASHINFER --trust_remote_code --tasks lambada_openai --batch_size auto --seed 42
Completely different results were obtained, and the issue is reproducible with the provided files.
The acc is 1 or 0.9998 (2x RTX2080Ti) and perplexity is NaN on Turing.
And the acc is 0.6546 and perplexity is 4.5249 on Ampere.

The log and results on Ampere
#lm_eval --model vllm --model_args pretrained=/RedHatAI/Qwen3-8B-NVFP4,gpu_memory_utilization=0.8,dtype=auto,attention_backend=FLASHINFER --trust_remote_code --tasks lambada_openai --batch_size auto --seed 42
2026-02-02:02:48:14 INFO     [_cli.run:376] Selected Tasks: ['lambada_openai']
2026-02-02:02:48:14 INFO     [evaluator:211] Setting random seed to 42 | Setting numpy seed to 42 | Setting torch manual seed to 42 | Setting fewshot manual seed to 42
2026-02-02:02:48:14 INFO     [evaluator:236] Initializing vllm model, with arguments: {'pretrained': '/RedHatAI/Qwen3-8B-NVFP4', 'gpu_memory_utilization': 0.8, 'dtype': 'auto', 'attention_backend': 'FLASHINFER', 'trust_remote_code': True}
INFO 02-02 02:48:17 [utils.py:250] non-default args: {'trust_remote_code': True, 'seed': 1234, 'gpu_memory_utilization': 0.8, 'disable_log_stats': True, 'attention_backend': 'FLASHINFER', 'model': '/RedHatAI/Qwen3-8B-NVFP4'}
The argument `trust_remote_code` is to be used with Auto classes. It has no effect here and is ignored.
The argument `trust_remote_code` is to be used with Auto classes. It has no effect here and is ignored.
INFO 02-02 02:48:18 [model.py:532] Resolved architecture: Qwen3ForCausalLM
INFO 02-02 02:48:18 [model.py:1543] Using max model len 40960
INFO 02-02 02:48:19 [scheduler.py:226] Chunked prefill is enabled with max_num_batched_tokens=8192.
INFO 02-02 02:48:19 [vllm.py:669] Asynchronous scheduling is enabled.
(EngineCore_DP0 pid=1748) INFO 02-02 02:48:19 [core.py:96] Initializing a V1 LLM engine (v0.16.0rc1.dev78+gb6bb2842c) with config: model='/RedHatAI/Qwen3-8B-NVFP4', speculative_config=None, tokenizer='/RedHatAI/Qwen3-8B-NVFP4', skip_tokenizer_init=False, tokenizer_mode=auto, revision=None, tokenizer_revision=None, trust_remote_code=True, dtype=torch.bfloat16, max_seq_len=40960, download_dir=None, load_format=auto, tensor_parallel_size=1, pipeline_parallel_size=1, data_parallel_size=1, disable_custom_all_reduce=False, quantization=compressed-tensors, enforce_eager=False, enable_return_routed_experts=False, kv_cache_dtype=auto, device_config=cuda, structured_outputs_config=StructuredOutputsConfig(backend='auto', disable_fallback=False, disable_any_whitespace=False, disable_additional_properties=False, reasoning_parser='', reasoning_parser_plugin='', enable_in_reasoning=False), observability_config=ObservabilityConfig(show_hidden_metrics_for_version=None, otlp_traces_endpoint=None, collect_detailed_traces=None, kv_cache_metrics=False, kv_cache_metrics_sample=0.01, cudagraph_metrics=False, enable_layerwise_nvtx_tracing=False, enable_mfu_metrics=False, enable_mm_processor_stats=False, enable_logging_iteration_details=False), seed=1234, served_model_name=/RedHatAI/Qwen3-8B-NVFP4, enable_prefix_caching=True, enable_chunked_prefill=True, pooler_config=None, compilation_config={'level': None, 'mode': <CompilationMode.VLLM_COMPILE: 3>, 'debug_dump_path': None, 'cache_dir': '', 'compile_cache_save_format': 'binary', 'backend': 'inductor', 'custom_ops': ['none'], 'splitting_ops': ['vllm::unified_attention', 'vllm::unified_attention_with_output', 'vllm::unified_mla_attention', 'vllm::unified_mla_attention_with_output', 'vllm::mamba_mixer2', 'vllm::mamba_mixer', 'vllm::short_conv', 'vllm::linear_attention', 'vllm::plamo2_mamba_mixer', 'vllm::gdn_attention_core', 'vllm::kda_attention', 'vllm::sparse_attn_indexer', 'vllm::rocm_aiter_sparse_attn_indexer', 'vllm::unified_kv_cache_update'], 'compile_mm_encoder': False, 'compile_sizes': [], 'compile_ranges_split_points': [8192], 'inductor_compile_config': {'enable_auto_functionalized_v2': False, 'combo_kernels': True, 'benchmark_combo_kernel': True}, 'inductor_passes': {}, 'cudagraph_mode': <CUDAGraphMode.FULL_AND_PIECEWISE: (2, 1)>, 'cudagraph_num_of_warmups': 1, 'cudagraph_capture_sizes': [1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128, 136, 144, 152, 160, 168, 176, 184, 192, 200, 208, 216, 224, 232, 240, 248, 256, 272, 288, 304, 320, 336, 352, 368, 384, 400, 416, 432, 448, 464, 480, 496, 512], 'cudagraph_copy_inputs': False, 'cudagraph_specialize_lora': True, 'use_inductor_graph_partition': False, 'pass_config': {'fuse_norm_quant': False, 'fuse_act_quant': False, 'fuse_attn_quant': False, 'eliminate_noops': True, 'enable_sp': False, 'fuse_gemm_comms': False, 'fuse_allreduce_rms': False, 'fuse_act_padding': False}, 'max_cudagraph_capture_size': 512, 'dynamic_shapes_config': {'type': <DynamicShapesType.BACKED: 'backed'>, 'evaluate_guards': False, 'assume_32_bit_indexing': False}, 'local_cache_dir': None, 'static_all_moe_layers': []}
(EngineCore_DP0 pid=1748) INFO 02-02 02:48:20 [parallel_state.py:1234] world_size=1 rank=0 local_rank=0 distributed_init_method=tcp://172.18.0.4:35079 backend=nccl
(EngineCore_DP0 pid=1748) INFO 02-02 02:48:20 [parallel_state.py:1445] rank 0 in world size 1 is assigned as DP rank 0, PP rank 0, PCP rank 0, TP rank 0, EP rank N/A
(EngineCore_DP0 pid=1748) INFO 02-02 02:48:20 [gpu_model_runner.py:4112] Starting to load model /RedHatAI/Qwen3-8B-NVFP4...
(EngineCore_DP0 pid=1748) INFO 02-02 02:48:20 [nvfp4_utils.py:85] Using NvFp4LinearBackend.MARLIN for NVFP4 GEMM
(EngineCore_DP0 pid=1748) INFO 02-02 02:48:21 [cuda.py:328] Using AttentionBackendEnum.FLASHINFER backend.
Loading safetensors checkpoint shards:   0% Completed | 0/2 [00:00<?, ?it/s]
Loading safetensors checkpoint shards:  50% Completed | 1/2 [00:00<00:00,  4.50it/s]
Loading safetensors checkpoint shards: 100% Completed | 2/2 [00:01<00:00,  1.60it/s]
Loading safetensors checkpoint shards: 100% Completed | 2/2 [00:01<00:00,  1.77it/s]
(EngineCore_DP0 pid=1748)
(EngineCore_DP0 pid=1748) INFO 02-02 02:48:22 [default_loader.py:291] Loading weights took 1.21 seconds
(EngineCore_DP0 pid=1748) WARNING 02-02 02:48:22 [marlin_utils_fp4.py:150] Your GPU does not have native support for FP4 computation but FP4 quantization is being used. Weight-only FP4 compression will be used leveraging the Marlin kernel. This may degrade performance for compute-heavy workloads.
(EngineCore_DP0 pid=1748) INFO 02-02 02:48:23 [gpu_model_runner.py:4209] Model loading took 6.0 GiB memory and 1.807443 seconds
(EngineCore_DP0 pid=1748) INFO 02-02 02:48:30 [backends.py:860] Using cache directory: /root/.cache/vllm/torch_compile_cache/936f4f19e2/rank_0_0/backbone for vLLM's torch.compile
(EngineCore_DP0 pid=1748) INFO 02-02 02:48:30 [backends.py:920] Dynamo bytecode transform time: 7.27 s
(EngineCore_DP0 pid=1748) INFO 02-02 02:48:38 [backends.py:300] Cache the graph of compile range (1, 8192) for later use
(EngineCore_DP0 pid=1748) INFO 02-02 02:48:43 [backends.py:317] Compiling a graph for compile range (1, 8192) takes 9.49 s
(EngineCore_DP0 pid=1748) INFO 02-02 02:48:43 [monitor.py:34] torch.compile takes 16.77 s in total
(EngineCore_DP0 pid=1748) INFO 02-02 02:48:44 [gpu_worker.py:359] Available KV cache memory: 11.35 GiB
(EngineCore_DP0 pid=1748) INFO 02-02 02:48:44 [kv_cache_utils.py:1307] GPU KV cache size: 82,624 tokens
(EngineCore_DP0 pid=1748) INFO 02-02 02:48:44 [kv_cache_utils.py:1312] Maximum concurrency for 40,960 tokens per request: 2.02x
(EngineCore_DP0 pid=1748) INFO 02-02 02:48:44 [kernel_warmup.py:64] Warming up FlashInfer attention.
Capturing CUDA graphs (mixed prefill-decode, PIECEWISE): 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 51/51 [00:03<00:00, 16.16it/s]
Capturing CUDA graphs (decode, FULL): 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 35/35 [00:01<00:00, 20.01it/s]
(EngineCore_DP0 pid=1748) INFO 02-02 02:48:50 [gpu_model_runner.py:5216] Graph capturing finished in 6 secs, took 0.89 GiB
(EngineCore_DP0 pid=1748) INFO 02-02 02:48:50 [core.py:272] init engine (profile, create kv cache, warmup model) took 27.29 seconds
INFO 02-02 02:48:51 [llm.py:349] Supported tasks: ['generate']
2026-02-02:02:48:56 INFO     [tasks:700] Selected tasks:
2026-02-02:02:48:56 INFO     [tasks:691] Task: lambada_openai (lambada/lambada_openai.yaml)
2026-02-02:02:48:56 INFO     [api.task:311] Building contexts for lambada_openai on rank 0...
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 5153/5153 [00:06<00:00, 763.71it/s]
2026-02-02:02:49:03 INFO     [evaluator:584] Running loglikelihood requests
Adding requests: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 5153/5153 [00:00<00:00, 11088.04it/s]
Processed prompts: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████| 5153/5153 [01:26<00:00, 59.27it/s, est. speed input: 4502.09 toks/s, output: 59.27 toks/s]
Running loglikelihood requests: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 5153/5153 [01:27<00:00, 58.68it/s]
bootstrapping for stddev: perplexity
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 100/100 [00:02<00:00, 43.79it/s]
2026-02-02:02:50:36 INFO     [loggers.evaluation_tracker:316] Output path not provided, skipping saving results aggregated
vllm ({'pretrained': '/RedHatAI/Qwen3-8B-NVFP4', 'gpu_memory_utilization': 0.8, 'dtype': 'auto', 'attention_backend': 'FLASHINFER'}), gen_kwargs: ({}), limit: None, num_fewshot: None, batch_size: auto
|    Tasks     |Version|Filter|n-shot|  Metric  |   |Value |   |Stderr|
|--------------|------:|------|-----:|----------|---|-----:|---|-----:|
|lambada_openai|      1|none  |     0|acc       |↑  |0.6546|±  |0.0066|
|              |       |none  |     0|perplexity|↓  |4.5249|±  |0.1376|
The log and results on Turing
#lm_eval --model vllm --model_args pretrained=/RedHatAI/Qwen3-8B-NVFP4,gpu_memory_utilization=0.8,dtype=auto,attention_backend=FLASHINFER --trust_remote_code --tasks lambada_openai --batch_size auto --seed 42
2026-02-02:02:48:26 INFO     [_cli.run:376] Selected Tasks: ['lambada_openai']
2026-02-02:02:48:26 INFO     [evaluator:211] Setting random seed to 42 | Setting numpy seed to 42 | Setting torch manual seed to 42 | Setting fewshot manual seed to 42
2026-02-02:02:48:26 INFO     [evaluator:236] Initializing vllm model, with arguments: {'pretrained': '/RedHatAI/Qwen3-8B-NVFP4', 'gpu_memory_utilization': 0.8, 'dtype': 'auto', 'attention_backend': 'FLASHINFER', 'trust_remote_code': True}
INFO 02-02 02:48:29 [utils.py:250] non-default args: {'trust_remote_code': True, 'seed': 1234, 'gpu_memory_utilization': 0.8, 'disable_log_stats': True, 'attention_backend': 'FLASHINFER', 'model': '/RedHatAI/Qwen3-8B-NVFP4'}
The argument `trust_remote_code` is to be used with Auto classes. It has no effect here and is ignored.
The argument `trust_remote_code` is to be used with Auto classes. It has no effect here and is ignored.
INFO 02-02 02:48:29 [model.py:532] Resolved architecture: Qwen3ForCausalLM
WARNING 02-02 02:48:29 [model.py:1815] Your device 'NVIDIA GeForce RTX 2080 Ti' (with compute capability 7.5) doesn't support torch.bfloat16. Falling back to torch.float16 for compatibility.
WARNING 02-02 02:48:29 [model.py:1868] Casting torch.bfloat16 to torch.float16.
INFO 02-02 02:48:29 [model.py:1543] Using max model len 40960
INFO 02-02 02:48:30 [scheduler.py:226] Chunked prefill is enabled with max_num_batched_tokens=8192.
INFO 02-02 02:48:30 [vllm.py:669] Asynchronous scheduling is enabled.
(EngineCore_DP0 pid=1746) INFO 02-02 02:48:31 [core.py:96] Initializing a V1 LLM engine (v0.16.0rc1.dev78+gb6bb2842c) with config: model='/RedHatAI/Qwen3-8B-NVFP4', speculative_config=None, tokenizer='/RedHatAI/Qwen3-8B-NVFP4', skip_tokenizer_init=False, tokenizer_mode=auto, revision=None, tokenizer_revision=None, trust_remote_code=True, dtype=torch.float16, max_seq_len=40960, download_dir=None, load_format=auto, tensor_parallel_size=1, pipeline_parallel_size=1, data_parallel_size=1, disable_custom_all_reduce=False, quantization=compressed-tensors, enforce_eager=False, enable_return_routed_experts=False, kv_cache_dtype=auto, device_config=cuda, structured_outputs_config=StructuredOutputsConfig(backend='auto', disable_fallback=False, disable_any_whitespace=False, disable_additional_properties=False, reasoning_parser='', reasoning_parser_plugin='', enable_in_reasoning=False), observability_config=ObservabilityConfig(show_hidden_metrics_for_version=None, otlp_traces_endpoint=None, collect_detailed_traces=None, kv_cache_metrics=False, kv_cache_metrics_sample=0.01, cudagraph_metrics=False, enable_layerwise_nvtx_tracing=False, enable_mfu_metrics=False, enable_mm_processor_stats=False, enable_logging_iteration_details=False), seed=1234, served_model_name=/RedHatAI/Qwen3-8B-NVFP4, enable_prefix_caching=True, enable_chunked_prefill=True, pooler_config=None, compilation_config={'level': None, 'mode': <CompilationMode.VLLM_COMPILE: 3>, 'debug_dump_path': None, 'cache_dir': '', 'compile_cache_save_format': 'binary', 'backend': 'inductor', 'custom_ops': ['none'], 'splitting_ops': ['vllm::unified_attention', 'vllm::unified_attention_with_output', 'vllm::unified_mla_attention', 'vllm::unified_mla_attention_with_output', 'vllm::mamba_mixer2', 'vllm::mamba_mixer', 'vllm::short_conv', 'vllm::linear_attention', 'vllm::plamo2_mamba_mixer', 'vllm::gdn_attention_core', 'vllm::kda_attention', 'vllm::sparse_attn_indexer', 'vllm::rocm_aiter_sparse_attn_indexer', 'vllm::unified_kv_cache_update'], 'compile_mm_encoder': False, 'compile_sizes': [], 'compile_ranges_split_points': [8192], 'inductor_compile_config': {'enable_auto_functionalized_v2': False, 'combo_kernels': True, 'benchmark_combo_kernel': True}, 'inductor_passes': {}, 'cudagraph_mode': <CUDAGraphMode.FULL_AND_PIECEWISE: (2, 1)>, 'cudagraph_num_of_warmups': 1, 'cudagraph_capture_sizes': [1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128, 136, 144, 152, 160, 168, 176, 184, 192, 200, 208, 216, 224, 232, 240, 248, 256, 272, 288, 304, 320, 336, 352, 368, 384, 400, 416, 432, 448, 464, 480, 496, 512], 'cudagraph_copy_inputs': False, 'cudagraph_specialize_lora': True, 'use_inductor_graph_partition': False, 'pass_config': {'fuse_norm_quant': False, 'fuse_act_quant': False, 'fuse_attn_quant': False, 'eliminate_noops': True, 'enable_sp': False, 'fuse_gemm_comms': False, 'fuse_allreduce_rms': False, 'fuse_act_padding': False}, 'max_cudagraph_capture_size': 512, 'dynamic_shapes_config': {'type': <DynamicShapesType.BACKED: 'backed'>, 'evaluate_guards': False, 'assume_32_bit_indexing': False}, 'local_cache_dir': None, 'static_all_moe_layers': []}
(EngineCore_DP0 pid=1746) INFO 02-02 02:48:31 [parallel_state.py:1234] world_size=1 rank=0 local_rank=0 distributed_init_method=tcp://172.18.0.5:40803 backend=nccl
(EngineCore_DP0 pid=1746) INFO 02-02 02:48:31 [parallel_state.py:1445] rank 0 in world size 1 is assigned as DP rank 0, PP rank 0, PCP rank 0, TP rank 0, EP rank N/A
(EngineCore_DP0 pid=1746) INFO 02-02 02:48:32 [gpu_model_runner.py:4112] Starting to load model /RedHatAI/Qwen3-8B-NVFP4...
(EngineCore_DP0 pid=1746) INFO 02-02 02:48:32 [nvfp4_utils.py:85] Using NvFp4LinearBackend.MARLIN for NVFP4 GEMM
(EngineCore_DP0 pid=1746) INFO 02-02 02:48:32 [cuda.py:328] Using AttentionBackendEnum.FLASHINFER backend.
Loading safetensors checkpoint shards:   0% Completed | 0/2 [00:00<?, ?it/s]
Loading safetensors checkpoint shards:  50% Completed | 1/2 [00:00<00:00,  2.62it/s]
Loading safetensors checkpoint shards: 100% Completed | 2/2 [00:01<00:00,  1.20it/s]
Loading safetensors checkpoint shards: 100% Completed | 2/2 [00:01<00:00,  1.31it/s]
(EngineCore_DP0 pid=1746)
(EngineCore_DP0 pid=1746) INFO 02-02 02:48:34 [default_loader.py:291] Loading weights took 1.62 seconds
(EngineCore_DP0 pid=1746) WARNING 02-02 02:48:34 [marlin_utils_fp4.py:150] Your GPU does not have native support for FP4 computation but FP4 quantization is being used. Weight-only FP4 compression will be used leveraging the Marlin kernel. This may degrade performance for compute-heavy workloads.
(EngineCore_DP0 pid=1746) INFO 02-02 02:48:35 [gpu_model_runner.py:4209] Model loading took 6.0 GiB memory and 2.266769 seconds
(EngineCore_DP0 pid=1746) INFO 02-02 02:48:42 [backends.py:860] Using cache directory: /root/.cache/vllm/torch_compile_cache/44d6fd0f98/rank_0_0/backbone for vLLM's torch.compile
(EngineCore_DP0 pid=1746) INFO 02-02 02:48:42 [backends.py:920] Dynamo bytecode transform time: 7.30 s
(EngineCore_DP0 pid=1746) INFO 02-02 02:48:50 [backends.py:300] Cache the graph of compile range (1, 8192) for later use
(EngineCore_DP0 pid=1746) INFO 02-02 02:48:55 [backends.py:317] Compiling a graph for compile range (1, 8192) takes 10.05 s
(EngineCore_DP0 pid=1746) INFO 02-02 02:48:55 [monitor.py:34] torch.compile takes 17.35 s in total
(EngineCore_DP0 pid=1746) INFO 02-02 02:48:57 [gpu_worker.py:359] Available KV cache memory: 9.7 GiB
(EngineCore_DP0 pid=1746) INFO 02-02 02:48:57 [kv_cache_utils.py:1307] GPU KV cache size: 70,624 tokens
(EngineCore_DP0 pid=1746) INFO 02-02 02:48:57 [kv_cache_utils.py:1312] Maximum concurrency for 40,960 tokens per request: 1.72x
(EngineCore_DP0 pid=1746) INFO 02-02 02:48:57 [kernel_warmup.py:64] Warming up FlashInfer attention.
Capturing CUDA graphs (mixed prefill-decode, PIECEWISE): 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 51/51 [00:03<00:00, 13.45it/s]
Capturing CUDA graphs (decode, FULL): 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 35/35 [00:02<00:00, 16.96it/s]
(EngineCore_DP0 pid=1746) INFO 02-02 02:49:03 [gpu_model_runner.py:5216] Graph capturing finished in 7 secs, took 0.89 GiB
(EngineCore_DP0 pid=1746) INFO 02-02 02:49:03 [core.py:272] init engine (profile, create kv cache, warmup model) took 28.59 seconds
INFO 02-02 02:49:04 [llm.py:349] Supported tasks: ['generate']
2026-02-02:02:49:09 INFO     [tasks:700] Selected tasks:
2026-02-02:02:49:09 INFO     [tasks:691] Task: lambada_openai (lambada/lambada_openai.yaml)
2026-02-02:02:49:09 INFO     [api.task:311] Building contexts for lambada_openai on rank 0...
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 5153/5153 [00:06<00:00, 786.20it/s]
2026-02-02:02:49:16 INFO     [evaluator:584] Running loglikelihood requests
Adding requests: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 5153/5153 [00:00<00:00, 11382.45it/s]
Processed prompts: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████| 5153/5153 [01:51<00:00, 46.30it/s, est. speed input: 3516.82 toks/s, output: 46.30 toks/s]
Running loglikelihood requests: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 5153/5153 [01:51<00:00, 46.06it/s]
bootstrapping for stddev: perplexity
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 100/100 [00:02<00:00, 44.54it/s]
2026-02-02:02:51:13 INFO     [loggers.evaluation_tracker:316] Output path not provided, skipping saving results aggregated
vllm ({'pretrained': '/RedHatAI/Qwen3-8B-NVFP4', 'gpu_memory_utilization': 0.8, 'dtype': 'auto', 'attention_backend': 'FLASHINFER'}), gen_kwargs: ({}), limit: None, num_fewshot: None, batch_size: auto
|    Tasks     |Version|Filter|n-shot|  Metric  |   |Value|   |Stderr|
|--------------|------:|------|-----:|----------|---|----:|---|-----:|
|lambada_openai|      1|none  |     0|acc       |↑  |    1|±  |     0|
|              |       |none  |     0|perplexity|↓  |  NaN|±  |   NaN|
Before submitting a new issue...

Make sure you already searched for relevant issues, and asked the chatbot living at the bottom right corner of the documentation page, which can answer lots of frequently asked questions.
