"""Baseline stage: static diagnosis from the model implementations only."""

from typing import Dict, Any
import logging

from my_agent.diagnosis.state import DiagnosisState


class BaselineMixin:
    """Static baseline: diagnose from the model source only, without activations."""

    def baseline_static_agent(self, state: DiagnosisState) -> Dict[str, Any]:
        """
        Baseline static agent that relies only on the model definition
        """
        model_impl = state.get("model_impl", {})
        hf_model_impl = model_impl["hf"]
        vllm_model_impl = model_impl["vllm"]
        
        prompt = f"""
        
        You are an expert ML engineer diagnosing accuracy differences between inference engines for the same model.
        Your overall task is to analyze the implementation differences between the two inference engines based on the information provided to you as well as your own investigation and decide which components may be causing the accuracy differences.
        Issue description: 
        {state['problem_description']}
        
        Here are the model definitions for the two inference engines:
        HF model definition:
        {hf_model_impl}
        vLLM model definition:
        {vllm_model_impl}
        
        You can assume that the HF model definition is the ground truth one and the vLLM model definition is the one that is causing the accuracy differences.
        
        You should output the component class name that you suspect is the root cause component of the accuracy differences.
        In addition, you should give reasoning for your decision, for each reasoning, you should outline the line number of the code in the class definition that you think is the root cause of the accuracy differences.
        
        Give top-3 suspected components that you think are the root cause components of the accuracy differences.
        
        Output format:
        ```json
        {{
            "suspected_components": List[str] = Field(description="The component class names that you suspect are the root cause components of the accuracy differences."),
            "reasoning": Dict[str, str] = Field(description="The reasoning for why the components are suspected to be the root cause of the issue, key is the component name in suspected_components and value is the reasoning for why the component is suspected to be the root cause of the issue. You should mention code line numbers in the reasoning for each component.")
        }}
        ```
        """
        response = self.llm.invoke(prompt)
        logging.info(f"[Baseline Static Agent] Response: {response}")
        return response
