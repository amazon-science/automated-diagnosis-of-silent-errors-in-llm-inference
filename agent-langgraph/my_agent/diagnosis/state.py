"""LangGraph state schema for the diagnosis workflow."""

from typing import TypedDict, List, Dict, Required

from my_agent.output_structure import ComponentMappingOutput


class DiagnosisState(TypedDict, total=False):
    problem_description: Required[str]
    error_threshold: Required[float]
    next_action: Required[str]
    component_mappings: Required[ComponentMappingOutput]
    alignment_results: Required[List]
    analysis_results: Required[List]
    model_tree: Required[Dict[str, str]]
    model_impl: Required[Dict[str, str]]
    input_path: Required[str]
    output_path: Required[str]
    activation_path: Required[str]
    ref_activation_prefix: Required[str]
    ref_activation_fp32_prefix: Required[str]
    target_activation_prefix: Required[str]
    suspicious_components: Required[List]
    mode: Required[str]
    engine_type: Required[str]
    checkpoint_path: Required[str]
    next_stage: Required[str]
    knowledge_base_path: Required[str]
    original_target_activation_prefix: Required[str]
    original_activation_path: Required[str]
    alignment_multi_trial: Required[bool]
    alignment_multi_trial_count: Required[int]
    # Trial/feedback configuration (optional; defaulted from globals if absent)
    component_mapping_max_trials: int
    component_mapping_max_error_feedback: int
    activation_alignment_max_trials: int
    activation_alignment_max_error_feedback: int
    detection_top_k: int


def set_defaults(state: DiagnosisState) -> DiagnosisState:
    return {
        "next_action": "analyze",
        "component_mappings": {},
        "model_tree": {},
        "model_impl": {},
        "input_path": "",
        "output_path": "",
        "activation_path": "",
        "ref_activation_prefix": "",
        "target_activation_prefix": "",
        "knowledge_base_path": "",
        **state  # override defaults with provided fields
    }
