"""The analyst agent, assembled from one mixin per diagnosis stage."""

from my_agent.diagnosis.agent.base import AgentBase
from my_agent.diagnosis.agent.baseline import BaselineMixin
from my_agent.diagnosis.agent.component_mapping import ComponentMappingMixin
from my_agent.diagnosis.agent.activation_io import ActivationIOMixin
from my_agent.diagnosis.agent.activation_alignment import ActivationAlignmentMixin
from my_agent.diagnosis.agent.analysis import AnalysisMixin


class AnalystAgent(
    BaselineMixin,
    ComponentMappingMixin,
    ActivationIOMixin,
    ActivationAlignmentMixin,
    AnalysisMixin,
    AgentBase,
):
    """LLM-backed agent that performs each diagnosis stage."""


__all__ = ["AnalystAgent"]
