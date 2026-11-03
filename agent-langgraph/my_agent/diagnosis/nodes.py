"""LangGraph node functions: one per diagnosis stage, each wrapping an AnalystAgent call and persisting its output."""

import json
import os
import logging

from my_agent.output_structure import ComponentMappingOutput
from my_agent.diagnosis.state import DiagnosisState
from my_agent.diagnosis.agent import AnalystAgent


def baseline_static_agent_node(state: DiagnosisState) -> DiagnosisState:
    """Static baseline: diagnose from model source only."""
    analyst = AnalystAgent()
    result = analyst.baseline_static_agent(state)
    logging.info(result)
    return state


def create_component_mapping_node(state: DiagnosisState) -> DiagnosisState:
    """Stage 1. Writes component_mapping.json; reuses a checkpoint's mapping when next_stage is set."""
    analyst = AnalystAgent()
    if state["next_stage"] == "alignment" or state["next_stage"] == "analysis" or state["next_stage"] == "mapping":
        # try component_mapping.json or gt_component_mapping.json
        if os.path.exists(os.path.join(state["checkpoint_path"], "component_mapping.json")):
            with open(os.path.join(state["checkpoint_path"], "component_mapping.json"), "r") as f:
                save_result = json.load(f)
        elif os.path.exists(os.path.join(state["checkpoint_path"], "gt_component_mapping.json")):
            with open(os.path.join(state["checkpoint_path"], "gt_component_mapping.json"), "r") as f:
                save_result = json.load(f)
        else:
            raise ValueError(f"No component mapping found in {state['checkpoint_path']}")
        state["component_mappings"] = ComponentMappingOutput(component_mapping=save_result["component_mapping"], variables=save_result["variables"], reasoning=save_result["reasoning"])
        with open(os.path.join(state["output_path"], "component_mapping.json"), "w") as f:
            json.dump(save_result, f, indent=4)
        return state
    result = analyst.create_component_mapping(state)
    logging.info(result)
    state["component_mappings"] = result
    save_result = {
        "mapping_size": len(result.component_mapping),
        "component_mapping": result.component_mapping,
        "variables": result.variables,
        "reasoning": result.reasoning
    }
    with open(os.path.join(state.get("output_path", ""), "component_mapping.json"), "w") as f:
        json.dump(save_result, f, indent=4)
    return state


def activation_alignment_node(state: DiagnosisState) -> DiagnosisState:
    """Stage 2. Writes alignment_results.json; reuses a checkpoint's results when next_stage == 'analysis'."""
    analyst = AnalystAgent()
    if state["next_stage"] == "analysis":
        with open(os.path.join(state["checkpoint_path"], "alignment_results.json"), "r") as f:
            save_result = json.load(f)
        state["alignment_results"] = save_result
        with open(os.path.join(state["output_path"], "alignment_results.json"), "w") as f:
            json.dump(save_result, f, indent=4)
        return state
    result = analyst.align_hf_activations_with_vllm(state)
    state["alignment_results"] = result
    with open(os.path.join(state.get("output_path", ""), "alignment_results.json"), "w") as f:
        json.dump(result, f, indent=4)
    logging.info(result)
    return state


def analysis_node(state: DiagnosisState) -> DiagnosisState:
    """Stage 3. Writes analysis_results.json, call_sequence_aligned.json and detection_result.json."""
    analyst = AnalystAgent()

    analysis_results = analyst.analyze_alignment_results(state)
    with open(os.path.join(state.get("output_path", ""), "analysis_results.json"), "w") as f:
        json.dump(analysis_results, f, indent=4)
    logging.info(analysis_results)
    state["analysis_results"] = analysis_results
    call_sequence_aligned, detection_result = analyst.detect_abnormal_components(state)
    with open(os.path.join(state.get("output_path", ""), "call_sequence_aligned.json"), "w") as f:
        json.dump(call_sequence_aligned, f, indent=4)
    state["call_sequence_aligned"] = call_sequence_aligned
    with open(os.path.join(state.get("output_path", ""), "detection_result.json"), "w") as f:
        json.dump(detection_result, f, indent=4)
    state["detection_result"] = detection_result
    return state
