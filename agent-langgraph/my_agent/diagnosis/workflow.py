"""Graph construction: the full diagnosis pipeline and single-stage graphs for debugging."""

from langgraph.graph import StateGraph

from my_agent.diagnosis.state import DiagnosisState, set_defaults
from my_agent.diagnosis.nodes import baseline_static_agent_node, create_component_mapping_node, activation_alignment_node, analysis_node


def diagnose_workflow():
    """Full pipeline: component mapping -> activation alignment -> analysis."""
    g = StateGraph(DiagnosisState)

    g.add_node("create_component_mapping", create_component_mapping_node)
    g.add_node("activation_alignment", activation_alignment_node)
    g.add_node("analyze", analysis_node)
    g.set_entry_point("create_component_mapping")
    g.add_edge("create_component_mapping", "activation_alignment")
    g.add_edge("activation_alignment", "analyze")
    return g.compile()


def build_baseline_graph():
    """Static baseline only (diagnose from the model source, no activations)."""
    g = StateGraph(DiagnosisState)
    g.add_node("set_defaults", set_defaults)
    g.add_node("baseline_static_agent", baseline_static_agent_node)
    g.set_entry_point("set_defaults")
    g.add_edge("set_defaults", "baseline_static_agent")
    return g.compile()


def build_component_mapping_graph():
    """Stage 1 only, for debugging."""
    g = StateGraph(DiagnosisState)
    g.add_node("set_defaults", set_defaults)
    g.add_node("analyst", create_component_mapping_node)
    g.set_entry_point("set_defaults")
    g.add_edge("set_defaults", "analyst")
    return g.compile()


def build_alignment_graph():
    """Stage 2 only (use with checkpoint_path + next_stage), for debugging."""
    g = StateGraph(DiagnosisState)
    g.add_node("set_defaults", set_defaults)
    g.add_node("analyst", activation_alignment_node)
    g.set_entry_point("set_defaults")
    g.add_edge("set_defaults", "analyst")
    return g.compile()


def build_analysis_graph():
    """Stage 3 only (use with checkpoint_path + next_stage), for debugging."""
    g = StateGraph(DiagnosisState)
    g.add_node("set_defaults", set_defaults)
    g.add_node("analyst", analysis_node)
    g.set_entry_point("set_defaults")
    g.add_edge("set_defaults", "analyst")
    return g.compile()
