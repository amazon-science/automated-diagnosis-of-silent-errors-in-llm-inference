# Ekka diagnosis agent

Entry point: `my_agent/diagnose.py` (see the repository README for setup).

```bash
python agent-langgraph/my_agent/diagnose.py --config_path bug-cases/vllm/vllm-26812/config_with_kb.yaml
# equivalently, from agent-langgraph/ with the repo root on PYTHONPATH:
PYTHONPATH=.. python -m my_agent.diagnosis --config_path ../bug-cases/vllm/vllm-26812/config_with_kb.yaml
# several configs in sequence:
agent-langgraph/my_agent/scripts/run.sh bug-cases/vllm/*/config_with_kb.yaml
```

## Code layout

The pipeline is a LangGraph graph with three stages:
**component mapping → activation alignment → analysis**.

```
my_agent/
├── diagnose.py                     CLI entry point
├── diagnosis/
│   ├── cli.py                      load config, build initial state, run_diagnosis(), main()
│   ├── workflow.py                 diagnose_workflow() (full graph) + single-stage debug graphs
│   ├── nodes.py                    one graph node per stage; writes each stage's JSON output and
│   │                               handles resuming from checkpoint_path / next_stage
│   ├── state.py                    DiagnosisState schema
│   ├── runtime.py                  shared run-time state: LLM handle, loaded activations, KB root
│   ├── agent/                      AnalystAgent = one mixin per stage
│   │   ├── component_mapping.py    stage 1: map reference (HF) modules to target-engine modules
│   │   ├── activation_io.py        find/load activation files; gather module source for prompts
│   │   ├── activation_alignment.py stage 2: LLM writes postprocess_hf_activations() to align tensors
│   │   ├── analysis.py             stage 3: error ratios along the call sequence, root-cause detection
│   │   ├── baseline.py             static baseline (model source only)
│   │   └── base.py                 shared init (LLM, token tracer)
│   └── tools/                      LangChain tools available to the LLM
│       ├── code_lookup.py          get_class_definition (reference_index / target_index)
│       ├── tensors.py              list/compare registered activation tensors
│       └── knowledge_base.py       list/read alignment examples (agent-training/knowledge_base)
├── output_structure.py             structured-output schemas for each stage
├── utils/globals.py                thresholds and retry limits
├── utils/token_tracer.py           token usage accounting
└── scripts/                        source-extraction helpers used by the stages
```

Each run writes to `<output_path>_<timestamp>/`:
`component_mapping.json`, `alignment_results.json`, `analysis_results.json`,
`call_sequence_aligned.json`, `detection_result.json`, and `diagnosis_log.log`.
