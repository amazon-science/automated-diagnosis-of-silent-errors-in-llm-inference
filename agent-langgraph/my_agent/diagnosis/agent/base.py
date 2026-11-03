"""Common initialisation for the analyst agent."""

import logging

from my_agent.utils.token_tracer import TokenTracer
from my_agent.diagnosis import runtime


class AgentBase:
    """Holds the LLM handle and token accounting shared by all stages."""

    def __init__(self):
        self.llm = runtime.MODEL
        logging.info(f"LLM: {self.llm}")
        self.token_tracer = TokenTracer()
