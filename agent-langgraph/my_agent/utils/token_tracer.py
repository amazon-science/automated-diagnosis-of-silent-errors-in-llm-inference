import logging
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from typing import Any

class TokenTracer(BaseCallbackHandler):
        # self.token_count = 0

    # def on_llm_start(self, serialized, messages, **kwargs):
    #     # This is right BEFORE an LLM call
    #     # messages is already the list that will be sent
    #     print(f"on_llm_start: {messages}")
    #     num_tokens = self.llm.get_num_tokens_from_messages(messages)
    #     print(f"num_tokens = {num_tokens}")
    #     logging.info(f"[LLM call] input tokens = {num_tokens}")
    #     # self.token_count += num_tokens

    #     # keep a copy if you want
    #     self.current_messages = messages
        
    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        """Fires after each successful LLM call"""
        # print(f"on_llm_end: {response}")
        logging.info(f"[TokenTracer] [on_llm_end] prompt tokens = {response.llm_output['usage']['prompt_tokens']}, completion tokens = {response.llm_output['usage']['completion_tokens']}, total tokens = {response.llm_output['usage']['total_tokens']}")
        logging.info(f"[TokenTracer] [on_llm_end] full usage = {response.llm_output['usage']}")

    # def on_tool_end(self, output, **kwargs):
    #     print("on_tool_end")
    #     # A tool just ran and produced `output`
    #     # In many agent setups, the framework will next add a ToolMessage
    #     tool_msg = ToolMessage(content=str(output), tool_call_id=kwargs.get("tool_call_id"))
    #     msgs_with_tool = list(self.current_messages) + [tool_msg]
    #     num_tokens = self.llm.get_num_tokens_from_messages(msgs_with_tool)
    #     logging.info(f"[After tool output] tokens if we call LLM now = {num_tokens}")
    #     # self.token_count += num_tokens