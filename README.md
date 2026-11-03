# Ekka: Automated Diagnosis of Silent Errors in LLM Inference

[![ICML](https://img.shields.io/badge/ICML-2026-2e7d32.svg)](https://icml.cc/virtual/2026/poster/66507)
[![arXiv](https://img.shields.io/badge/arXiv-2606.04594-b31b1b.svg)](https://arxiv.org/abs/2606.04594)

**ICML 2026** · [Paper (arXiv)](https://arxiv.org/abs/2606.04594) · [ICML poster](https://icml.cc/virtual/2026/poster/66507)

Yile Gu<sup>1</sup>, Zhen Zhang<sup>2</sup>, Shaowei Zhu<sup>2</sup>, Xinwei Fu<sup>2</sup>,
Jun Wu<sup>2</sup>, Yida Wang<sup>2</sup>, Baris Kasikci<sup>1</sup>

<sup>1</sup>University of Washington &nbsp;&nbsp;
<sup>2</sup>Amazon Web Services

---

Ekka is an LLM agent that diagnoses **accuracy bugs in LLM inference engines** (vLLM, SGLang). Given a
bug report and a model that produces different outputs in a target engine than in a reference
implementation (Hugging Face `transformers` / `diffusers`), Ekka:

1. **Collects activations** from both frameworks on the same prompt (`drivers/`, `activation_collector/`).
2. **Maps components** between the two model implementations (for example HF `self_attn.q_proj/k_proj/v_proj`
   ↔ vLLM `self_attn.qkv_proj`).
3. **Aligns activations**: writes and validates code that reshapes each reference activation to match the target.
4. **Localizes the bug** by walking the call sequence and finding where the error first grows beyond the
   bf16/fp32 noise floor, then reports the most likely buggy components.

An optional **knowledge base** of alignment examples (`agent-training/knowledge_base/`) obtained from a
training stage on bug-free models helps the alignment step.

## Repository layout

| Path | What it is |
| --- | --- |
| `agent-langgraph/my_agent/diagnose.py` | Entry point of the diagnosis agent; the implementation is the `my_agent/diagnosis/` package |
| `common/` | Shared helpers: config loading with `${VAR}` expansion, LLM provider factory |
| `drivers/` | Run a model in HF / vLLM / SGLang from a YAML config and dump activations |
| `activation_collector/` | Forward hooks that record module inputs/outputs and the call sequence (HF side) |
| `bug-cases/{vllm,sglang}/<case>/` | Benchmark bug cases (see below) |
| `agent-training/` | Bug-free training cases and the resulting knowledge base |
| `evaluator/` | Scores a diagnosis run against ground truth |
| `baseline/`, `docker-scripts/` | General-purpose coding-agent baselines and their container |
| `scripts/` | Utilities: class-index builder, module-tree extraction, activation post-processing |

Each bug case contains:

```
bug-cases/vllm/vllm-26812/
├── config_no_kb.yaml / config_with_kb.yaml   # diagnosis configs (without / with the knowledge base)
├── src/
│   ├── issue_description.md                  # the bug report given to the agent
│   ├── hf_config.yaml, hf_config_fp32.yaml   # reference driver configs (bf16 and fp32)
│   └── vllm_config.yaml | sglang_config.yaml # target driver config
├── model_tree/   model_impl/                 # module trees and model source for both frameworks
├── call_sequence_{hf,vllm|sglang}/           # module call order recorded by the drivers
└── evaluation/                               # ground-truth mapping and buggy component, eval config
```

## Setup

Requirements: Linux, NVIDIA GPUs (most cases need 1–4 GPUs, some need 8), Python 3.10+.

```bash
git clone <this repo> Ekka && cd Ekka
cp .env.example .env        # then fill in credentials and, optionally, EKKA_ACTIVATION_DIR
```

### 1. Python environments

The agent, HF reference, and each engine are easiest to keep in separate environments:

```bash
# Agent + evaluator
python -m venv .venv-agent && . .venv-agent/bin/activate
pip install -r agent-langgraph/my_agent/requirements.txt

# Reference (Hugging Face) driver
pip install torch transformers accelerate pyyaml

# Target engines: the activation hooks live in Ekka's forks of the engines.
#   vLLM:   https://github.com/IKACE/vLLM   (e.g. branch v0.11.0-baseline-docker)
#   SGLang: https://github.com/IKACE/SGLang (e.g. branch v0.5.6-baseline-docker)
# Install each fork in its own environment (`pip install -e .` for vLLM, `pip install -e python` for SGLang).
```

Use the fork branch that still contains the bug: for example vllm-26812 needs the v0.11.0 line
(`v0.11.0-ported`/`v0.11.0-baseline*`); v0.13 already has the fix and shows no divergence. When the fork
branch only changes Python files, you can skip the CUDA build by reusing the upstream release kernels:

```bash
VLLM_USE_PRECOMPILED=1 \
VLLM_PRECOMPILED_WHEEL_LOCATION=https://github.com/vllm-project/vllm/releases/download/v0.11.0/vllm-0.11.0-cp38-abi3-manylinux1_x86_64.whl \
pip install -e .
pip install "transformers==4.57.3" accelerate   # keep transformers on the version the engine was built against
```

The same environment can run the HF reference driver.

`docker-scripts/Dockerfile` builds an image with both forks preinstalled (see `docker-scripts/README.md`).

### 2. Paths

All configs use two variables, expanded when the config is loaded:

- `EKKA_ROOT`: repository root (defaults to the checkout location)
- `EKKA_ACTIVATION_DIR`: activation storage (defaults to `$EKKA_ROOT/activations`; point it at a large disk)

### 3. LLM backend

Set `provider` in a diagnosis config, or `EKKA_LLM_PROVIDER` to override all configs:

| provider | Credentials | Model id |
| --- | --- | --- |
| `bedrock` (default in the shipped configs) | standard AWS chain (`AWS_PROFILE`, keys, role) + `AWS_REGION` | Bedrock ID, e.g. `global.anthropic.claude-sonnet-4-5-20250929-v1:0` |
| `anthropic` | `ANTHROPIC_API_KEY` | API name, e.g. `claude-sonnet-4-5-20250929` (Bedrock IDs are converted automatically) |

`EKKA_MODEL_ID` overrides `model_id` in every config. List the Bedrock inference profiles you can use with
`aws bedrock list-inference-profiles --region us-east-1 --query "inferenceProfileSummaries[].inferenceProfileId"`
(for example `global.anthropic.claude-sonnet-5`). Check the backend before a long run:

```bash
python -c "import sys; sys.path[:0]=['.']; from common.llm import make_chat_model; m=make_chat_model('bedrock'); print(m.model_id, m.invoke('say hi').content)"
```

Bedrock requests use a 30-minute read timeout (`EKKA_BEDROCK_READ_TIMEOUT`, in seconds), since responses with
large `max_tokens` arrive in one piece. Sonnet 5 and newer models reject conversations that end on an
assistant message; `common/llm.py` appends a short user turn in that case, so the LangGraph agents work unchanged.

## Running a bug case

```bash
CASE=bug-cases/vllm/vllm-26812

# 1) Collect activations (reference bf16 + fp32, then the target engine)
#    The HF driver must be launched with torchrun (it reads RANK/WORLD_SIZE), even on one GPU.
torchrun --nproc-per-node 1 drivers/driver_hf_v2.py --config $CASE/src/hf_config.yaml
torchrun --nproc-per-node 1 drivers/driver_hf_v2.py --config $CASE/src/hf_config_fp32.yaml
python drivers/driver_vllm_v2.py --config $CASE/src/vllm_config.yaml    # or driver_sglang.py for SGLang cases
#   multi-GPU HF runs: torchrun --nproc-per-node <N> ...
#   Each driver config needs `activation_collector.enabled: true` and a save_path equal to the diagnosis
#   config's activation_path; keep generation at 1 token so all runs capture the same prefill pass.
#   The vLLM fork always writes the `vllm_activation` prefix and ignores `save_prefix`.
#   vision-language cases: drivers/driver_hf_v2_vl.py and drivers/driver_sglang_vl.py

# 2) Build class indexes the agent uses to look up source code
scripts/update_index/build_index.sh reference \
    "$(python -c 'import transformers, os; print(os.path.dirname(transformers.__file__))')" $CASE
scripts/update_index/build_index.sh target /path/to/vllm/vllm $CASE

# 3) Diagnose
python agent-langgraph/my_agent/diagnose.py --config_path $CASE/config_with_kb.yaml
#   -> $CASE/diagnosis_output_<timestamp>/ (component_mapping.json, alignment_results.json,
#      detection_result.json, diagnosis_log.log, ...)

# 4) Evaluate
python evaluator/evaluator.py --config $CASE/evaluation/eval_config.yaml \
    --result-dir $CASE/diagnosis_output_<timestamp>
evaluator/run_all_evals.sh      # every diagnosis_output_* under bug-cases/
```

Diagnosis configs support `mode: diagnosis` (full workflow), `mode: training` (builds alignment examples
on bug-free models, see `agent-training/cases/`), and `mode: baseline_1` (static analysis of the model
code only). Other options: `detection_top_k`, `component_mapping_max_trials`,
`activation_alignment_max_trials`, `alignment_multi_trial`, and `checkpoint_path` + `next_stage`
(`mapping` | `alignment` | `analysis`) to resume from an earlier run's output directory.

### Troubleshooting

| Symptom | Cause |
| --- | --- |
| `KeyError: Environment variable ... is not set` | a config references a variable you have not exported or put in `.env` |
| `No such file ... model_tree_hf.json` / `call_sequence_*` | `--config_path` points at the wrong case |
| activation file not found during alignment | `activation_path` or the `*_activation_prefix` keys do not match the files, or a driver config had `activation_collector` disabled |
| `KeyError: 'RANK'` from `driver_hf_v2.py` | launched with `python` instead of `torchrun` |
| `... requires accelerate` | `pip install accelerate` in the HF environment |
| Bedrock `AccessDeniedException` / `ValidationException: ... model identifier is invalid` | wrong region, or a model ID that is not an inference profile you can use |
| Bedrock `ValidationException: ... does not support assistant message prefill` | building the model without `common.llm.make_chat_model` |
| Bedrock `Read timeout` | raise `EKKA_BEDROCK_READ_TIMEOUT` |

### Adding a bug case

Copy an existing case, update the model and prompt in `src/*_config.yaml`, write
`src/issue_description.md`, run the drivers, then copy the `call_sequence_*` and `model_tree` folders the
drivers write next to the activations into the case directory and add the model sources to `model_impl/`.

## Baselines

See `baseline/README.md` for running mini-swe-agent and OpenCode on the same cases.

## Citation

If you use Ekka in your research, please cite:

```bibtex
@article{gu2026ekka,
  title={EKKA: Automated Diagnosis of Silent Errors in LLM Inference},
  author={Gu, Yile and Zhang, Zhen and Zhu, Shaowei and Fu, Xinwei and Wu, Jun and Wang, Yida and Kasikci, Baris},
  journal={arXiv preprint arXiv:2606.04594},
  year={2026}
}
```

## License

See [LICENSE](LICENSE).
