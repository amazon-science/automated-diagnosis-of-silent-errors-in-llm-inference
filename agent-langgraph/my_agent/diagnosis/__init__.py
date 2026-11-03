"""Ekka diagnosis workflow.

Package layout (one module per concern):

    cli.py              config loading, initial state, ``run_diagnosis`` and ``main``
    workflow.py         LangGraph graphs (full pipeline and single-stage debug graphs)
    nodes.py            graph nodes: run a stage, persist its JSON output, support checkpoints
    state.py            ``DiagnosisState`` schema
    runtime.py          shared run-time state (LLM handle, loaded activations, knowledge-base root)
    agent/              ``AnalystAgent``, composed of one mixin per stage:
        component_mapping.py     stage 1: map reference modules to target-engine modules
        activation_io.py         locating/loading activations and module source for prompts
        activation_alignment.py  stage 2: LLM-written code aligning reference and target activations
        analysis.py              stage 3: error ratios along the call sequence, root-cause detection
        baseline.py              static baseline (model source only)
    tools/              LangChain tools the LLM can call (class lookup, tensor inspection, knowledge base)
"""

from my_agent.diagnosis import runtime  # noqa: F401  (registers DTensor for torch.load)
