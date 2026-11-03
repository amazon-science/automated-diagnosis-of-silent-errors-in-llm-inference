"""Stage 2 - activation alignment: the LLM writes postprocess_hf_activations() so reference activations match the target layout."""

from langgraph.prebuilt import create_react_agent
from typing import List, Dict, Any, Tuple
from langchain_core.messages import HumanMessage
import json
import os
import logging

from my_agent.utils.globals import (
    ACTIVATION_ALIGNMENT_MAX_RETRY,
    ACTIVATION_ALIGNMENT_MAX_ERROR_FEEDBACK,
)
from my_agent.output_structure import ComponentMappingOutput, AlignActivationOutput
from my_agent.diagnosis import runtime
from my_agent.diagnosis.state import DiagnosisState
from my_agent.diagnosis.tools.tensors import get_tensor_pair_sum, infer_tensor_match, list_tensor_keys
from my_agent.diagnosis.tools.knowledge_base import list_alignment_examples, get_alignment_example_code


class ActivationAlignmentMixin:
    """Stage 2: generate and validate code that aligns reference activations with the target's."""

    def format_alignment_prompt(
        self,
        problem_description: str,
        hf_components: List[str],
        vllm_components: List[str],
        hf_preview: str,
        vllm_preview: str,
        forward_context: str,
        function_signature: str,
        error_threshold: float,
        mode: str,
    ) -> str:
        def indent_block(block: str) -> str:
            if not block:
                return "    (no data available)"
            return "\n".join(("    " + line) if line else "" for line in block.splitlines())

        hf_block = indent_block(hf_preview)
        vllm_block = indent_block(vllm_preview)
        forward_block = indent_block(forward_context)
        if mode == "training":
            error_ratio_prompt = f"""Implement the following function so that HuggingFace output activations can be postprocessed to match vLLM output activations within an L2-relative tolerance of {error_threshold}:
{function_signature}
"""
        else:
            error_ratio_prompt = f"""Implement the following function so that HuggingFace output activations can be postprocessed to match vLLM output activations:
{function_signature}
"""

        return f"""

{problem_description}

We are aligning output activations for the following component mapping:
  - HuggingFace components: {hf_components}
  - vLLM components: {vllm_components}


You will be given the HuggingFace and vLLM activations during one forward pass of the model. It could be a prefill pass or a decode pass. Both the HuggingFace and vLLM activations will be in the form of a tuple of tensors, collected from the module's output.

{error_ratio_prompt}

Requirements and guidelines for the function implementation:
- Do not fabricate placeholder tensors.
- Ensure HuggingFace and vLLM output activations have matching shapes before returning.
- Avoid computing L2 metrics inside this function.
- You may add temporary debug prints, but remove them when asked. Your debug prints will be fed back to you in the error feedback loop.
- For vLLM, retrieve the output activations from the activations loaded from the file path.
- For HuggingFace, retrieve the output activations from the activations loaded from the file path and additionally do processing to match vLLM's output activations.
- The return result is formatted as a dictionary based on the vLLM components.
- You can modify the docstring of the function but do not change the function signature.
- If knowledge base is provided, you should use the alignment examples to guide your implementation. Available tools: list_alignment_examples, get_alignment_example_code.
- Steps:
  1. You should look at the forward definitions and check if the output variables in vLLM are aligned with the output variables in HuggingFace.
  2. If you find activations in HuggingFace that are aligned with the output variables in vLLM, you should postprocess them to match the output variables in vLLM.
  3. If you find activations in HuggingFace that are not aligned with the output variables in vLLM, you should return the original activations in vLLM.
  4. In both cases, you should return the reasoning for why you choose such actions.

Activation snapshots (first few entries shown for guidance):
HuggingFace:
{hf_block}

vLLM:
{vllm_block}

Relevant forward definitions:
{forward_block}

Return the python code enclosed in a python code block starting with ```python and ending with ```.
"""

    def format_alignment_summarization_prompt(
        self,
        original_prompt: str,
        examples: List[Dict[str, Any]],
        function_signature: str,
        error_threshold: float,
        mode: str,
    ) -> str:
        """
        Format a summarization prompt that takes the original alignment prompt and multiple
        example alignment codes with their error ratios to generate a final alignment code.
        
        Args:
            original_prompt: The original alignment prompt
            examples: List of dictionaries, each containing:
                - "function_code": The alignment function code
                - "error_ratio": The calculated error ratio for this component
                - "reasoning": The reasoning from the alignment attempt
            function_signature: The function signature to implement
            error_threshold: The error threshold for alignment
            mode: Mode of operation ("training" or "diagnosis")
        """
        examples_text = ""
        for i, example in enumerate(examples, 1):
            error_ratio = example.get("error_ratio", "N/A")
            function_code = example.get("function_code", "")
            reasoning = example.get("reasoning", "No reasoning provided")
            
            examples_text += f"""
Example {i}:
Error Ratio: {error_ratio}
Reasoning: {reasoning}
Function Code:
```python
{function_code}
```
"""
        
        if mode == "training":
            error_ratio_prompt = f"""Implement the following function so that HuggingFace output activations can be postprocessed to match vLLM output activations within an L2-relative tolerance of {error_threshold}:
{function_signature}
"""
        else:
            error_ratio_prompt = f"""Implement the following function so that HuggingFace output activations can be postprocessed to match vLLM output activations:
{function_signature}
"""
        
        return f"""
You are given an original alignment task and multiple example alignment implementations with their error ratios.

Original Alignment Task:
{original_prompt}

{error_ratio_prompt}

Multiple alignment attempts have been made for this component pair. Below are {len(examples)} example implementations with their calculated error ratios:

{examples_text}

Your task is to analyze these examples and generate a final, improved alignment function that:
1. Takes the best aspects from the examples
2. Avoids the obvious mistakes or inefficiencies present in the examples
3. Follows the same function signature and requirements as specified in the original task

Requirements:
- Do not fabricate placeholder tensors
- Ensure HuggingFace and vLLM output activations have matching shapes before returning
- Avoid computing L2 metrics inside this function
- The return result is formatted as a dictionary based on the vLLM components
- You can modify the docstring of the function but do not change the function signature

Return the python code enclosed in a python code block starting with ```python and ending with ```.
"""

    def run_single_alignment_trial(
        self,
        prompt: str,
        react_agent: Any,
        component_pair: Tuple[List[str], List[str]],
        component_mappings: ComponentMappingOutput,
        activation_path: str,
        ref_activation_prefix: str,
        ref_activation_fp32_prefix: str,
        target_activation_prefix: str,
        mode: str,
        error_threshold: float,
        max_retries: int = ACTIVATION_ALIGNMENT_MAX_RETRY,
        max_error_feedback: int = ACTIVATION_ALIGNMENT_MAX_ERROR_FEEDBACK,
    ) -> Dict[str, Any]:
        """
        Run a single alignment trial and return the result with error ratio.
        
        Returns:
            Dictionary with keys:
                - "function_code": The alignment function code
                - "reasoning": The reasoning from the alignment
                - "error_ratio": The calculated error ratio (average across components)
                - "alignment_results": Full alignment results
                - "success": Whether the alignment was successful
        """
        error_feedback = ""
        error_feedback_attempt = 0
        attempt = 1
        success = False
        function_code = ""
        structured = None
        
        while not success and attempt <= max_retries:
            try:
                inputs = {"messages": [HumanMessage(content=prompt + error_feedback)]}
                response = react_agent.invoke(inputs, config={"callbacks": [self.token_tracer]})
                structured = response["structured_response"]
                
                function_code = self.extract_function_code(structured.code)
                execution_result = self.execute_function(function_code)
                if execution_result["error"]:
                    raise ValueError(
                        f"Execution failed: {execution_result['error']} \n"
                        f"Output before failure: {execution_result['stdout']} \n"
                        f"Error output: {execution_result['stderr']}"
                    )
                
                results = self.calculate_component_error_ratio(
                    component_pair=component_pair,
                    function_code=function_code,
                    component_mapping_variables=component_mappings.variables,
                    activation_path=activation_path,
                    ref_activation_prefix=ref_activation_prefix,
                    ref_activation_fp32_prefix=ref_activation_fp32_prefix,
                    target_activation_prefix=target_activation_prefix,
                    mode=mode,
                )
                
                cleanup_feedback = f"""
The function executed successfully and the activations are aligned within the tolerance of {error_threshold}.
Please remove any print statements from the function.

function code: {function_code}
""" 
                inputs = {"messages": [HumanMessage(content=prompt + cleanup_feedback)]}
                cleanup_response = react_agent.invoke(inputs, config={"callbacks": [self.token_tracer]})
                structured = cleanup_response["structured_response"]
                function_code = self.extract_function_code(structured.code)
                execution_result = self.execute_function(function_code)
                if execution_result["error"]:
                    raise ValueError(
                        f"Execution failed: {execution_result['error']} \n"
                        f"Output before failure: {execution_result['stdout']} \n"
                        f"Error output: {execution_result['stderr']}"
                    )
                
                results = self.calculate_component_error_ratio(
                    component_pair=component_pair,
                    function_code=function_code,
                    component_mapping_variables=component_mappings.variables,
                    activation_path=activation_path,
                    ref_activation_prefix=ref_activation_prefix,
                    ref_activation_fp32_prefix=ref_activation_fp32_prefix,
                    target_activation_prefix=target_activation_prefix,
                    mode=mode,
                )
                
                # Calculate average error ratio across all components
                error_ratios = []
                for component_name, component_results in results.items():
                    if isinstance(component_results, dict):
                        # Handle case with layer indices
                        for layer_key, layer_results in component_results.items():
                            if isinstance(layer_results, dict) and "ratio" in layer_results:
                                error_ratios.append(layer_results["ratio"])
                    elif isinstance(component_results, dict) and "ratio" in component_results:
                        error_ratios.append(component_results["ratio"])
                
                avg_error_ratio = sum(error_ratios) / len(error_ratios) if error_ratios else float('inf')
                
                success = True
                return {
                    "function_code": function_code,
                    "reasoning": structured.reasoning,
                    "error_ratio": avg_error_ratio,
                    "alignment_results": results,
                    "success": True,
                    "attempt": attempt
                }
                
            except Exception as e:
                logging.info(f"Error in alignment trial attempt {attempt}: {e} \n function code: {function_code}")
                error_feedback_attempt += 1
                error_feedback = f"""

Previous attempt failed with error: {e}

Previous attempt function code: {function_code}

Please fix the implementation and try again.
"""
                if error_feedback_attempt > max_error_feedback:
                    logging.info(
                        "Failed alignment trial after "
                        f"{max_error_feedback} error feedback attempts, "
                        f"doing a fresh attempt {attempt+1}/{max_retries}"
                    )
                    attempt += 1
                    error_feedback_attempt = 0
                    error_feedback = ""
                    if attempt > max_retries:
                        logging.info(f"Failed alignment trial after {max_retries} attempts")
                        return {
                            "function_code": function_code,
                            "reasoning": structured.reasoning if structured else "No reasoning provided",
                            "error_ratio": float('inf'),
                            "alignment_results": None,
                            "success": False,
                            "attempt": attempt
                        }
                    continue
                continue
        
        return {
            "function_code": function_code,
            "reasoning": structured.reasoning if structured else "No reasoning provided",
            "error_ratio": float('inf'),
            "alignment_results": None,
            "success": False,
            "attempt": attempt
        }

    def align_hf_activations_with_vllm(self, state: DiagnosisState) -> Dict[str, Any]:
        component_mappings = state["component_mappings"]
        activation_path = state.get("activation_path")
        ref_activation_prefix = state.get("ref_activation_prefix")
        ref_activation_fp32_prefix = state.get("ref_activation_fp32_prefix")
        target_activation_prefix = state.get("target_activation_prefix")
        if not activation_path or not ref_activation_prefix or not target_activation_prefix:
            raise ValueError("activation_path and ref_activation_prefix and target_activation_prefix must be provided in the configuration to locate activation tensors.")

        hf_index_path = os.path.join(state.get("input_path", ""), "reference_index")
        vllm_index_path = os.path.join(state.get("input_path", ""), "target_index")

        base_llm = runtime.MODEL
        if state["knowledge_base_path"] is not None:
            tools = [
                get_tensor_pair_sum,
                infer_tensor_match,
                list_tensor_keys,
                list_alignment_examples,
                get_alignment_example_code,
            ]
        else:
            tools = [
                get_tensor_pair_sum,
                infer_tensor_match,
                list_tensor_keys,
            ]
        react_agent = create_react_agent(
            model=base_llm,
            tools=tools,
            prompt="You are a helpful assistant",
            response_format=AlignActivationOutput
        )
        alignment_results_checkpointed = None
        
        # Configuration for activation-alignment trials and feedback attempts
        alignment_max_trials = state.get(
            "activation_alignment_max_trials",
            ACTIVATION_ALIGNMENT_MAX_RETRY,
        )
        alignment_max_error_feedback = state.get(
            "activation_alignment_max_error_feedback",
            ACTIVATION_ALIGNMENT_MAX_ERROR_FEEDBACK,
        )
        
        if state["next_stage"] == "alignment":
            with open(os.path.join(state["checkpoint_path"], "alignment_results.json"), "r") as f:
                alignment_results_checkpointed = json.load(f)
        alignment_results: List[Dict[str, Any]] = []

        for component_pair in component_mappings.component_mapping:
            # if "qk_norm" not in component_pair[0][0]:
            #     continue
            log_msg = f"Aligning components: {component_pair}"
            logging.info(log_msg)

            hf_components_raw = component_pair[0]
            vllm_components_raw = component_pair[1]
            if alignment_results_checkpointed is not None:
                cache_hit = False
                for cached_result in alignment_results_checkpointed:
                    if cached_result["component_pair"][0] == component_pair[0] and cached_result["component_pair"][1] == component_pair[1] and cached_result["aligned"]:
                        alignment_results.append(cached_result)
                        cache_hit = True
                        break
                if cache_hit:
                    logging.info(f"[align_hf_activations_with_vllm] Cache hit for {component_pair} alignment results, skipping alignment")
                    continue

            hf_components = [self.expand_component_name(comp, component_mappings) for comp in hf_components_raw]
            vllm_components = [self.expand_component_name(comp, component_mappings) for comp in vllm_components_raw]

            hf_class_names = [
                self.get_class_name_from_module_tree(comp, component_mappings, state["model_tree"]["hf"])
                for comp in hf_components_raw
            ]
            vllm_class_names = [
                self.get_class_name_from_module_tree(comp, component_mappings, state["model_tree"]["vllm"])
                for comp in vllm_components_raw
            ]

            hf_forward_map = self.gather_forward_snippets(hf_class_names, hf_index_path)
            vllm_forward_map = self.gather_forward_snippets(vllm_class_names, vllm_index_path)
            forward_context = self.format_forward_context(hf_forward_map, vllm_forward_map)

            try:
                hf_files = self.resolve_activation_files(hf_components, activation_path, ref_activation_prefix)
                vllm_files = self.resolve_activation_files(vllm_components, activation_path, target_activation_prefix)
            except Exception as e:
                logging.error(f"Error resolving activation files: {e}")
                continue

            runtime.ACTIVATION_STORE.clear()
            hf_preview = self.load_and_register_activation_set(hf_files, "hf")
            vllm_preview = self.load_and_register_activation_set(vllm_files, "vllm")
            
            function_signature = self.get_function_signature(hf_files, vllm_files)

            prompt = self.format_alignment_prompt(
                state["problem_description"],
                hf_components,
                vllm_components,
                hf_preview,
                vllm_preview,
                forward_context,
                function_signature,
                state["error_threshold"],
                state["mode"],
            )
            
            logging.info(f"Generated alignment prompt: {prompt}")

            # Check if multi-trial mode is enabled
            use_multi_trial = state.get("alignment_multi_trial", False)
            multi_trial_count = state.get("alignment_multi_trial_count", 3)
            
            if use_multi_trial:
                logging.info(f"Using multi-trial alignment mode with {multi_trial_count} trials for {component_pair}")
                
                # Run multiple trials
                trial_examples = []
                for trial_num in range(1, multi_trial_count + 1):
                    logging.info(f"Running alignment trial {trial_num}/{multi_trial_count} for {component_pair}")
                    trial_result = self.run_single_alignment_trial(
                        prompt=prompt,
                        react_agent=react_agent,
                        component_pair=component_pair,
                        component_mappings=component_mappings,
                        activation_path=activation_path,
                        ref_activation_prefix=ref_activation_prefix,
                        ref_activation_fp32_prefix=ref_activation_fp32_prefix,
                        target_activation_prefix=target_activation_prefix,
                        mode=state["mode"],
                        error_threshold=state["error_threshold"],
                        max_retries=alignment_max_trials,
                        max_error_feedback=alignment_max_error_feedback,
                    )
                    
                    if trial_result["success"]:
                        trial_examples.append({
                            "function_code": trial_result["function_code"],
                            "error_ratio": trial_result["error_ratio"],
                            "reasoning": trial_result["reasoning"],
                        })
                        # Persist each successful trial's code for inspection.
                        code_postfix = ""
                        for vllm_component in component_pair[1]:
                            code_postfix += f"_{self.sanitize_component_name(vllm_component)}"
                        trial_filename = os.path.join(
                            state.get("output_path", ""),
                            f"alignment_code_trial{trial_num}{code_postfix}.py",
                        )
                        with open(trial_filename, "w") as f:
                            f.write(trial_result["function_code"])
                        logging.info(f"Trial {trial_num} completed with error ratio: {trial_result['error_ratio']}")
                    else:
                        logging.warning(f"Trial {trial_num} failed, skipping for summarization")
                
                if not trial_examples:
                    logging.error(f"All {multi_trial_count} trials failed for {component_pair}, falling back to single trial")
                    use_multi_trial = False
                else:
                    # Generate summarization prompt
                    summarization_prompt = self.format_alignment_summarization_prompt(
                        original_prompt=prompt,
                        examples=trial_examples,
                        function_signature=function_signature,
                        error_threshold=state["error_threshold"],
                        mode=state["mode"],
                    )
                    logging.info(f"Generated summarization prompt: {summarization_prompt}")
                    
                    logging.info(f"Generating final alignment code from {len(trial_examples)} examples using summarization")
                    
                    # Generate final alignment code using summarization
                    summarization_attempt = 1
                    final_success = False
                    final_function_code = ""
                    final_structured = None
                    
                    while not final_success and summarization_attempt <= alignment_max_trials:
                        try:
                            inputs = {"messages": [HumanMessage(content=summarization_prompt)]}
                            response = react_agent.invoke(inputs, config={"callbacks": [self.token_tracer]})
                            final_structured = response["structured_response"]
                            
                            final_function_code = self.extract_function_code(final_structured.code)
                            execution_result = self.execute_function(final_function_code)
                            if execution_result["error"]:
                                raise ValueError(
                                    f"Execution failed: {execution_result['error']} \n"
                                    f"Output before failure: {execution_result['stdout']} \n"
                                    f"Error output: {execution_result['stderr']}"
                                )
                            
                            final_results = self.calculate_component_error_ratio(
                                component_pair=component_pair,
                                function_code=final_function_code,
                                component_mapping_variables=component_mappings.variables,
                                activation_path=activation_path,
                                ref_activation_prefix=ref_activation_prefix,
                                ref_activation_fp32_prefix=ref_activation_fp32_prefix,
                                target_activation_prefix=target_activation_prefix,
                                mode=state["mode"],
                            )
                            
                            final_success = True
                            logging.info(f"Successfully generated final alignment code from summarization for {component_pair}")
                            
                            alignment_results.append({
                                "function_code": final_function_code,
                                "reasoning": final_structured.reasoning,
                                "alignment_results": final_results,
                                "component_pair": component_pair,
                                "aligned": True,
                                "attempt": summarization_attempt,
                                "multi_trial": True,
                                "trial_count": len(trial_examples),
                                "trial_examples": trial_examples,
                            })
                            with open(os.path.join(state.get("output_path", ""), f"alignment_results.json"), "w") as f:
                                json.dump(alignment_results, f, indent=4)
                            
                            # Save final summarization function code to file
                            code_postfix = ""
                            for vllm_component in component_pair[1]:
                                code_postfix += f"_{self.sanitize_component_name(vllm_component)}"
                            with open(os.path.join(state.get("output_path", ""), f"alignment_code_{code_postfix}.py"), "w") as f:
                                f.write(final_function_code)
                            
                            continue  # Skip the single-trial logic below
                            
                        except Exception as e:
                            logging.info(f"Error in summarization attempt {summarization_attempt}: {e}")
                            summarization_attempt += 1
                            if summarization_attempt > alignment_max_trials:
                                logging.error(
                                    "Failed to generate final alignment code from summarization "
                                    f"after {alignment_max_trials} attempts"
                                )
                                # Fall back to using the best trial example
                                best_trial = min(trial_examples, key=lambda x: x["error_ratio"])
                                logging.info(f"Using best trial example with error ratio: {best_trial['error_ratio']}")
                                
                                final_function_code = best_trial["function_code"]
                                final_results = self.calculate_component_error_ratio(
                                    component_pair=component_pair,
                                    function_code=final_function_code,
                                    component_mapping_variables=component_mappings.variables,
                                    activation_path=activation_path,
                                    ref_activation_prefix=ref_activation_prefix,
                                    ref_activation_fp32_prefix=ref_activation_fp32_prefix,
                                    target_activation_prefix=target_activation_prefix,
                                    mode=state["mode"],
                                )
                                
                                alignment_results.append({
                                    "function_code": final_function_code,
                                    "reasoning": best_trial["reasoning"],
                                    "alignment_results": final_results,
                                    "component_pair": component_pair,
                                    "aligned": True,
                                    "attempt": summarization_attempt,
                                    "multi_trial": True,
                                    "trial_count": len(trial_examples),
                                    "trial_examples": trial_examples,
                                    "summarization_failed": True,
                                })
                                with open(os.path.join(state.get("output_path", ""), f"alignment_results.json"), "w") as f:
                                    json.dump(alignment_results, f, indent=4)
                                
                                code_postfix = ""
                                for vllm_component in component_pair[1]:
                                    code_postfix += f"_{self.sanitize_component_name(vllm_component)}"
                                with open(os.path.join(state.get("output_path", ""), f"alignment_code_{code_postfix}.py"), "w") as f:
                                    f.write(final_function_code)
                                
                                continue  # Skip the single-trial logic below
            
            # Single-trial mode (original behavior)
            if not use_multi_trial:
                error_feedback = ""
                error_feedback_attempt = 0
                attempt = 1
                success = False
                function_code = ""
                structured = None

                # Use configurable limits for attempts and feedback within this component pair
                single_max_trials = alignment_max_trials
                single_max_error_feedback = alignment_max_error_feedback

                while not success and attempt <= single_max_trials:
                    try:
                        inputs = {"messages": [HumanMessage(content=prompt + error_feedback)]}
                        response = react_agent.invoke(inputs, config={"callbacks": [self.token_tracer]})
                        # logging.info(f"ReAct Agent attempt {attempt}: {response}")
                        structured = response["structured_response"]

                        function_code = self.extract_function_code(structured.code)
                        execution_result = self.execute_function(function_code)
                        if execution_result["error"]:
                            raise ValueError(
                                f"Execution failed: {execution_result['error']} \n"
                                f"Output before failure: {execution_result['stdout']} \n"
                                f"Error output: {execution_result['stderr']}"
                            )

                        # results = execution_result["result"]
                        # self.check_alignment_errors(results, state["error_threshold"], state["mode"])
                        results = self.calculate_component_error_ratio(
                            component_pair=component_pair,
                            function_code=function_code,
                            component_mapping_variables=component_mappings.variables,
                            activation_path=activation_path,
                            ref_activation_prefix=ref_activation_prefix,
                            ref_activation_fp32_prefix=ref_activation_fp32_prefix,
                            target_activation_prefix=target_activation_prefix,
                            mode=state["mode"],
                        )
                
                        cleanup_feedback = f"""
The function executed successfully and the activations are aligned within the tolerance of {state["error_threshold"]}.
Please remove any print statements from the function.

function code: {function_code}
""" 
                        inputs = {"messages": [HumanMessage(content=prompt + cleanup_feedback)]}
                        cleanup_response = react_agent.invoke(inputs, config={"callbacks": [self.token_tracer]})
                        logging.info(f"ReAct Agent attempt {attempt}: {cleanup_response}")
                        structured = cleanup_response["structured_response"]
                        function_code = self.extract_function_code(structured.code)
                        execution_result = self.execute_function(function_code)
                        if execution_result["error"]:
                            raise ValueError(
                                f"Execution failed: {execution_result['error']} \n"
                                f"Output before failure: {execution_result['stdout']} \n"
                                f"Error output: {execution_result['stderr']}"
                            )
                            
                        results = self.calculate_component_error_ratio(
                            component_pair=component_pair,
                            function_code=function_code,
                            component_mapping_variables=component_mappings.variables,
                            activation_path=activation_path,
                            ref_activation_prefix=ref_activation_prefix,
                            ref_activation_fp32_prefix=ref_activation_fp32_prefix,
                            target_activation_prefix=target_activation_prefix,
                            mode=state["mode"],
                        )
                
                        
                        success = True
                        logging.info(f"Successfully aligned activations for {component_pair} in {attempt} attempts")
                        alignment_results.append({
                            "function_code": function_code,
                            "reasoning": structured.reasoning,
                            "alignment_results": results,
                            "component_pair": component_pair,
                            "aligned": True,
                            "attempt": attempt
                        })
                        with open(os.path.join(state.get("output_path", ""), f"alignment_results.json"), "w") as f:
                            json.dump(alignment_results, f, indent=4)
                        logging.info(f"Alignment results: {alignment_results}")
                        # save function code to file
                        code_postfix = ""
                        for vllm_component in component_pair[1]:
                            code_postfix += f"_{self.sanitize_component_name(vllm_component)}"
                        with open(os.path.join(state.get("output_path", ""), f"alignment_code_{code_postfix}.py"), "w") as f:
                            f.write(function_code)


                    except Exception as e:
                        logging.info(f"Error in activation alignment attempt {attempt}: {e} \n function code: {function_code}")
                        error_feedback_attempt += 1
                        error_feedback = f"""

Previous attempt failed with error: {e}

Previous attempt function code: {function_code}

Please fix the implementation and try again.
"""
                        if error_feedback_attempt > single_max_error_feedback:
                            logging.info(
                                "Failed to align activations for "
                                f"{component_pair} after {single_max_error_feedback} error "
                                f"feedback attempts, doing a fresh attempt {attempt+1}/{single_max_trials}"
                            )
                            attempt += 1
                            error_feedback_attempt = 0
                            error_feedback = ""
                            if attempt > single_max_trials:
                                logging.info(
                                    "Failed to align activations for "
                                    f"{component_pair} terminating this component alignment as it "
                                    f"reached the maximum trial limit: {single_max_trials}"
                                )
                                alignment_results.append({
                                    "function_code": function_code,
                                    "reasoning": structured.reasoning if structured else "No reasoning provided",
                                    "alignment_results": None,
                                    "component_pair": component_pair,
                                    "aligned": False,
                                    "attempt": attempt
                                })
                                with open(os.path.join(state.get("output_path", ""), f"alignment_results.json"), "w") as f:
                                    json.dump(alignment_results, f, indent=4)
                                break
                            continue
                        continue

        return alignment_results

    def extract_function_code(self, response):
        """Extract Python function code from LLM response"""
        import re
        
        response_str = str(response)
        
        # Look for code blocks
        code_match = re.search(r'```python\n(.*?)\n```', response_str, re.DOTALL)
        if code_match:
            return code_match.group(1)
        
        # # If no code block, look for function definition
        # func_match = re.search(r'def postprocess_hf_activations.*?return.*', response_str, re.DOTALL)
        # if func_match:
        #     return func_match.group(0)
        
        return response_str

    def execute_function(self, function_code):
        """Execute the generated function and return results"""
        import torch
        from typing import Tuple, List
        import sys
        import traceback
        from io import StringIO
        
        # Create string buffers to capture output
        stdout_capture = StringIO()
        stderr_capture = StringIO()
        
        # Store original stdout/stderr
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        
        # Add line numbers to the function code for better error reporting
        numbered_code = '\n'.join(f"{i+1}: {line}" for i, line in enumerate(function_code.split('\n')))
        # print("Executing function with line numbers:")
        # print(numbered_code)
        # print("-" * 80)
        
        try:
            # Redirect stdout/stderr to our captures
            sys.stdout = stdout_capture
            sys.stderr = stderr_capture
            
            # Create execution namespace
            exec_globals = {
                'torch': torch,
                'Tuple': Tuple,
                'List': List
            }
            
            # Execute function and call it
            exec(function_code, exec_globals)
            result = exec_globals['postprocess_hf_activations']()
            
            # Get captured output
            stdout_output = stdout_capture.getvalue()
            stderr_output = stderr_capture.getvalue()
            
            return {
                'result': result,
                'stdout': stdout_output,
                'stderr': stderr_output,
                'error': None,
                'code': numbered_code
            }
            
        except Exception as e:
            # Capture the exception info including full traceback
            stdout_output = stdout_capture.getvalue()
            stderr_output = stderr_capture.getvalue()
            
            # Get the line number from the traceback
            tb = traceback.extract_tb(sys.exc_info()[2])
            error_line = None
            for frame in tb:
                if frame.filename == "<string>":
                    error_line = frame.lineno
                    break
            
            error_context = f"""
    Error occurred in generated function at line {error_line}:
    {numbered_code}
    ----------------------------------------
    Error on line {error_line}: {str(e)}
    Full traceback:
    {traceback.format_exc()}
            """
            
            return {
                'result': None,
                'stdout': stdout_output,
                'stderr': stderr_output,
                'error': {
                    'message': str(e),
                    'traceback': error_context,
                    'line_number': error_line
                },
                'code': numbered_code
            }
            
        finally:
            # Restore original stdout/stderr
            sys.stdout = old_stdout
            sys.stderr = old_stderr
            
            # Clean up
            stdout_capture.close()
            stderr_capture.close()

    def check_pairwise_alignment_errors(self, hf_activations, vllm_activations):
        """Check if alignment errors are within floating point tolerance"""
        import torch
        
        result = {}
        
        # Ensure both are lists of tensors
        if isinstance(hf_activations, torch.Tensor):
            hf_activations = [hf_activations]
        if isinstance(vllm_activations, torch.Tensor):
            vllm_activations = [vllm_activations]
        if not isinstance(hf_activations, list) and not isinstance(hf_activations, tuple):
            raise ValueError(f"[ERROR] hf_activations should be list or tuple type, but got {type(hf_activations)}")
        if not isinstance(vllm_activations, list) and not isinstance(vllm_activations, tuple):
            raise ValueError(f"[ERROR] vllm_activations should be list or tuple type, but got {type(vllm_activations)}")

        
        l2_diffs = []
        for i in range(min(len(hf_activations), len(vllm_activations))):
            hf_tensor = hf_activations[i]
            vllm_tensor = vllm_activations[i]
            
            if type(hf_tensor) != type(vllm_tensor):
                raise ValueError(f"[ERROR] Type mismatch between HuggingFace and vLLM activations at index {i}: {type(hf_tensor)} vs {type(vllm_tensor)}")

            if not isinstance(hf_tensor, torch.Tensor):
                raise ValueError(f"[ERROR] HuggingFace activation at index {i} is not a torch.Tensor but {type(hf_tensor)}")

            if not isinstance(vllm_tensor, torch.Tensor):
                raise ValueError(f"[ERROR] vLLM activation at index {i} is not a torch.Tensor but {type(vllm_tensor)}")

            if hf_tensor.shape != vllm_tensor.shape:
                raise ValueError(f"[ERROR] Shape mismatch between HuggingFace and vLLM activations at index {i}: {hf_tensor.shape} vs {vllm_tensor.shape}")
            
            # # assume shape of (seq_len, hidden_size), as we are going to norm over the last dimension
            # hf_tensor = hf_tensor.view(hf_tensor.shape[0], -1)
            # vllm_tensor = vllm_tensor.view(vllm_tensor.shape[0], -1)
            
            # flatten the tensor along the last dimension
            hf_tensor = hf_tensor.flatten()
            vllm_tensor = vllm_tensor.flatten()
            
            # to bfloat16
            hf_tensor = hf_tensor.to(torch.bfloat16)
            vllm_tensor = vllm_tensor.to(torch.bfloat16)
            
            # print(hf_tensor, vllm_tensor)
            # exit()
            
            # Handle DTensor conversion
            if hasattr(hf_tensor, 'to_local'):
                hf_tensor = hf_tensor.to_local()
            if hasattr(vllm_tensor, 'to_local'):
                vllm_tensor = vllm_tensor.to_local()

            # Calculate relative differences
            # rel_diff = torch.where(
            #     hf_tensor != 0,
            #     torch.abs((hf_tensor - vllm_tensor) / hf_tensor),
            #     torch.abs(hf_tensor - vllm_tensor)
            # )
            
            # max_rel_error = rel_diff.max().item()
            # result['max_rel_error'] = max(result['max_rel_error'], max_rel_error)
            
            # if max_rel_error > tolerance:
            #     result['within_tolerance'] = False
            
            l2_diff = torch.norm(hf_tensor - vllm_tensor, dim=0) / torch.norm(hf_tensor, dim=0)
            l2_diff = l2_diff.mean().item()
            l2_diffs.append(l2_diff)
        # print(l2_diffs)
        result["mean_l2_diff"] = torch.mean(torch.tensor(l2_diffs)).item()
        result["max_l2_diff"] = torch.max(torch.tensor(l2_diffs)).item()
        result["min_l2_diff"] = torch.min(torch.tensor(l2_diffs)).item()
        result["std_l2_diff"] = torch.std(torch.tensor(l2_diffs)).item()

        return result

    def check_alignment_errors(self, results):
        alignment_results = {}
        for vllm_component, component_results in results.items():
            try:
                # current = "input"
                # input_alignment = self.check_pairwise_alignment_errors(component_results["input"]["HF"], component_results["input"]["vLLM"])
                # current = "output"
                # output_alignment = self.check_pairwise_alignment_errors(component_results["output"]["HF"], component_results["output"]["vLLM"])
                output_alignment = self.check_pairwise_alignment_errors(component_results["HF"], component_results["vLLM"])
            except Exception as e:
                raise ValueError(f"when checking alignment error for results[{vllm_component}]['HF'] and results[{vllm_component}]['vLLM'], the following error is encountered {e}")
            alignment_results[vllm_component] = {
                "output_alignment": output_alignment
            }
            logging.info(f"[check_alignment_errors] Activations for {vllm_component} output_alignment: {output_alignment['max_l2_diff']}")
        return alignment_results
