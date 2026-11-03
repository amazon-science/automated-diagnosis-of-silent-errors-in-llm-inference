"""Stage 3 - analysis: error ratios along the call sequence and final root-cause detection."""

from typing import List, Dict, Any, Optional, Tuple
from langchain_core.messages import HumanMessage
import json
import os
import re
from itertools import product
import logging

from my_agent.utils.globals import (
    ERROR_THRESHOLD,
    EPS_DENOMINATOR,
)
from my_agent.output_structure import FinalAnalysisOutput
from my_agent.diagnosis.state import DiagnosisState


class AnalysisMixin:
    """Stage 3: compute per-component error ratios and localise the buggy component."""

    def calculate_instantiated_component_error_ratio(
        self,
        function_code_hf_vllm: str,
        ref_activation_path: str,
        ref_activation_fp32_path: str,
        mode: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Process instantiated component error ratio. It should receive function code with variable replaced.
        
        Args:
            function_code_hf_vllm: The function code to execute
            ref_activation_path: Path to BF16 reference activation files
            ref_activation_fp32_path: Path to FP32 reference activation files
            mode: Mode of operation ("training" or "diagnosis")
            
        Returns:
            Dictionary with analysis result for this execution, or None if processing failed
        """
        
        # Execute with BF16 reference
        execution_result_hf_bf16_vllm = self.execute_function(function_code_hf_vllm)
        if execution_result_hf_bf16_vllm["error"]:
            raise ValueError(f"[ERROR][calculate_instantiated_component_error_ratio] Failed to execute function code: {execution_result_hf_bf16_vllm['error']}, function code: {function_code_hf_vllm}")
        results_hf_bf16_vllm = execution_result_hf_bf16_vllm["result"]
        
        # Execute with FP32 reference
        function_code_hf_fp32_vllm = function_code_hf_vllm.replace(ref_activation_path, ref_activation_fp32_path)
        execution_result_hf_fp32_vllm = self.execute_function(function_code_hf_fp32_vllm)
        if execution_result_hf_fp32_vllm["error"]:
            raise ValueError(f"[ERROR][calculate_instantiated_component_error_ratio] Failed to execute function code: {execution_result_hf_fp32_vllm['error']}, function code: {function_code_hf_fp32_vllm}")
        results_hf_fp32_vllm = execution_result_hf_fp32_vllm["result"]
        
        # Assemble BF16 vs FP32 comparison
        results_hf_bf16_fp32 = {}
        for key in results_hf_fp32_vllm:
            results_hf_bf16_fp32[key] = {
                "vLLM": results_hf_bf16_vllm[key]["HF"],  # bf16 ref
                "HF": results_hf_fp32_vllm[key]["HF"]  # fp32 ref
            }
        
        try:
            alignment_result_hf_fp32_vllm = self.check_alignment_errors(results_hf_fp32_vllm)
            alignment_result_hf_bf16_fp32 = self.check_alignment_errors(results_hf_bf16_fp32)
        except Exception as e:
            raise ValueError(f"[ERROR][calculate_instantiated_component_error_ratio] Failed to check alignment errors: {e}")
        
        # Calculate error ratio for each component
        result_dict = {}
        for key in alignment_result_hf_bf16_fp32:
            max_l2_diff_hf_fp32_vllm = alignment_result_hf_fp32_vllm[key]["output_alignment"]["max_l2_diff"]
            max_l2_diff_hf_bf16_fp32 = alignment_result_hf_bf16_fp32[key]["output_alignment"]["max_l2_diff"]
            ratio = max_l2_diff_hf_fp32_vllm / (max_l2_diff_hf_bf16_fp32 + EPS_DENOMINATOR)
            abnormal = ratio > ERROR_THRESHOLD
            
            if mode == "training" and abnormal:
                raise ValueError(f"[ERROR][calculate_instantiated_component_error_ratio] Abnormal component: {key}, ratio: {ratio}, expected: {ERROR_THRESHOLD}, function code: {function_code_hf_vllm}")
            
            result_dict[key] = {
                "alignment_result_hf_bf16_fp32": alignment_result_hf_bf16_fp32[key],
                "alignment_result_hf_vllm": alignment_result_hf_fp32_vllm[key],
                "max_l2_diff_hf_fp32_vllm": max_l2_diff_hf_fp32_vllm,
                "max_l2_diff_hf_bf16_fp32": max_l2_diff_hf_bf16_fp32,
                "ratio": ratio,
                "abnormal": abnormal
            }
        
        return result_dict

    def calculate_component_error_ratio(
        self,
        component_pair: Tuple[List[str], List[str]],
        function_code: str,
        component_mapping_variables: Dict[str, List[str]],
        activation_path: str,
        ref_activation_prefix: str,
        ref_activation_fp32_prefix: str,
        target_activation_prefix: str,
        mode: str,
        original_target_activation_prefix: Optional[str] = None,
        original_activation_path: Optional[str] = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        Calculate error ratio for a single alignment result component.
        
        This function takes an alignment result and calculates the error ratio by:
        1. Executing the alignment function code with BF16 and FP32 reference activations
        2. Computing alignment errors between HF-FP32-vLLM and HF-BF16-FP32
        3. Calculating the ratio to determine if the component is abnormal
        
        Args:
            alignment_result: A dictionary containing:
                - "function_code": The alignment function code
                - "component_pair": Tuple of (hf_components, vllm_components)
                - "aligned": Whether alignment was successful
            component_mapping_variables: Dictionary mapping variable names to their possible values
            activation_path: Base path to activation files
            ref_activation_prefix: Prefix for reference (BF16) activation files
            ref_activation_fp32_prefix: Prefix for FP32 reference activation files
            target_activation_prefix: Prefix for target activation files
            original_target_activation_prefix: Prefix for original target activation files
            original_activation_path: Path to original activation files
            mode: Mode of operation ("training" or "diagnosis")
            
        Returns:
            Dictionary mapping component names to their analysis results, where each result contains:
                - "alignment_result_hf_bf16_fp32": Alignment result between HF BF16 and FP32
                - "alignment_result_hf_vllm": Alignment result between HF FP32 and vLLM
                - "max_l2_diff_hf_fp32_vllm": Max L2 difference between HF FP32 and vLLM
                - "max_l2_diff_hf_bf16_fp32": Max L2 difference between HF BF16 and FP32
                - "ratio": Error ratio (hf_fp32_vllm / hf_bf16_fp32)
                - "abnormal": Boolean indicating if ratio > 1.5
        """
        hf_components = component_pair[0]
        vllm_components = component_pair[1]
        hf_components_model_removed = [".".join(component.split(".")[1:]) for component in hf_components]
        vllm_components_model_removed = [".".join(component.split(".")[1:]) for component in vllm_components]
        
        if original_target_activation_prefix is None:
            original_target_activation_prefix = target_activation_prefix
            
        if original_activation_path is None:
            original_activation_path = activation_path
            
        if len(vllm_components) > 1:
            logging.info(f"[WARNING][calculate_component_error_ratio] Multiple components in the component pair: {vllm_components}, is not supported yet")
            return {}
        
        # Extract variable information from all components
        # Find all unique variables used across HF and vLLM components
        all_placeholders = set()
        for component in hf_components_model_removed + vllm_components_model_removed:
            placeholders = re.findall(r"\{([^{}]+)\}", component)
            all_placeholders.update(placeholders)
        
        variables_used = sorted(list(all_placeholders))  # Sort for consistent ordering
        
        # Check that all variables exist in component_mapping_variables
        missing_vars = [var for var in variables_used if var not in component_mapping_variables]
        if missing_vars:
            raise ValueError(f"Variable(s) {missing_vars} not found in component_mapping_variables: {component_mapping_variables}")
        
        # Get value lists for all variables
        value_lists = [component_mapping_variables[var] for var in variables_used] if variables_used else None
        analysis_result = {}
        
        if not value_lists:
            # No variable - process once
            function_code_hf_vllm = function_code.replace(original_activation_path, activation_path)
            # print(f"function_code_hf_vllm: {function_code_hf_vllm}")
            # print(f"original_activation_path: {original_activation_path}")
            # print(f"activation_path: {activation_path}")
            # exit()
            ref_activation_path = os.path.join(activation_path, ref_activation_prefix)
            ref_activation_fp32_path = os.path.join(activation_path, ref_activation_fp32_prefix)
            
            processed_result = self.calculate_instantiated_component_error_ratio(
                function_code_hf_vllm=function_code_hf_vllm,
                ref_activation_path=ref_activation_path,
                ref_activation_fp32_path=ref_activation_fp32_path,
                mode=mode,
            )
            
            if processed_result is None:
                return {}
            
            component_name = vllm_components[0]
            if component_name not in analysis_result:
                analysis_result[component_name] = {}
            
            # Store results with "no_layer_idx" key (processed_result has component names as keys)
            # Since there's only one component, extract the single result
            assert len(processed_result) == 1, f"Expected single component result, got {len(processed_result)}"
            for result_key in processed_result:
                analysis_result[component_name]["no_layer_idx"] = processed_result[result_key]
        else:
            # Has variables - process for each combination of variable values
            function_code_hf_vllm = function_code.replace(original_activation_path, activation_path)
            
            ref_activation_path = os.path.join(activation_path, ref_activation_prefix)
            ref_activation_fp32_path = os.path.join(activation_path, ref_activation_fp32_prefix)
            target_activation_path = os.path.join(activation_path, target_activation_prefix)
            original_target_activation_path = os.path.join(activation_path, original_target_activation_prefix)
            hf_initial_components = []
            vllm_initial_components = []
            # we always use the first value for each variable for alignment stage
            initial_values = [values[0] for values in value_lists]
            for hf_component in hf_components_model_removed:
                hf_initial_component = hf_component
                for var_name, initial_value in zip(variables_used, initial_values):
                    hf_initial_component = hf_initial_component.replace(f"{{{var_name}}}", initial_value)
                hf_initial_components.append(ref_activation_path + f"_{self.sanitize_component_name(hf_initial_component)}")
            for vllm_component in vllm_components_model_removed:
                vllm_initial_component = vllm_component
                for var_name, initial_value in zip(variables_used, initial_values):
                    vllm_initial_component = vllm_initial_component.replace(f"{{{var_name}}}", initial_value)
                vllm_initial_components.append(original_target_activation_path + f"_{self.sanitize_component_name(vllm_initial_component)}")
            
            # Generate all combinations of variable values (cartesian product)
            for value_combination in product(*value_lists):
                hf_current_components = []
                vllm_current_components = []
                for hf_component in hf_components_model_removed:
                    hf_current_component = hf_component
                    for var_name, value in zip(variables_used, value_combination):
                        hf_current_component = hf_current_component.replace(f"{{{var_name}}}", value)
                    hf_current_components.append(ref_activation_path + f"_{self.sanitize_component_name(hf_current_component)}")
                for vllm_component in vllm_components_model_removed:
                    vllm_current_component = vllm_component
                    for var_name, value in zip(variables_used, value_combination):
                        vllm_current_component = vllm_current_component.replace(f"{{{var_name}}}", value)
                    vllm_current_components.append(target_activation_path + f"_{self.sanitize_component_name(vllm_current_component)}")
                current_functio_code_hf_vllm = function_code_hf_vllm
                for i, hf_current_component in enumerate(hf_current_components):
                    current_functio_code_hf_vllm = current_functio_code_hf_vllm.replace(hf_initial_components[i], hf_current_component)
                for i, vllm_current_component in enumerate(vllm_current_components):
                    current_functio_code_hf_vllm = current_functio_code_hf_vllm.replace(vllm_initial_components[i], vllm_current_component)
                processed_result = self.calculate_instantiated_component_error_ratio(
                    function_code_hf_vllm=current_functio_code_hf_vllm,
                    ref_activation_path=ref_activation_path,
                    ref_activation_fp32_path=ref_activation_fp32_path,
                    mode=mode,
                )
                
                if processed_result is None:
                    continue
                
                component_name = vllm_components[0]
                if component_name not in analysis_result:
                    analysis_result[component_name] = {}
                
                # Store results with combination key (processed_result has component names as keys)
                # For single variable, use the value directly; for multiple variables, use underscore-joined string
                if len(value_combination) == 1:
                    combination_key = str(value_combination[0])
                else:
                    combination_key = "_".join(str(v) for v in value_combination)
                
                # Since there's only one component, extract the single result
                assert len(processed_result) == 1, f"Expected single component result, got {len(processed_result)}"
                for result_key in processed_result:
                    analysis_result[component_name][combination_key] = processed_result[result_key]
        
        return analysis_result

    def analyze_alignment_results(self, state: DiagnosisState) -> List[str]:
        alignment_results_hf_vllm = state["alignment_results"]
        component_mapping = state["component_mappings"]
        
        # Extract variables from dict or object
        if isinstance(component_mapping, dict):
            component_mapping_variables = component_mapping["variables"]
        else:
            component_mapping_variables = component_mapping.variables
        
        analysis_results = []
        for alignment_result_hf_vllm in alignment_results_hf_vllm:
            if alignment_result_hf_vllm.get("aligned", False) == False:
                continue
            analysis_result = self.calculate_component_error_ratio(
                component_pair=alignment_result_hf_vllm["component_pair"],
                function_code=alignment_result_hf_vllm["function_code"],
                component_mapping_variables=component_mapping_variables,
                activation_path=state["activation_path"],
                ref_activation_prefix=state["ref_activation_prefix"],
                ref_activation_fp32_prefix=state["ref_activation_fp32_prefix"],
                target_activation_prefix=state["target_activation_prefix"],
                mode=state["mode"],
                original_target_activation_prefix=state["original_target_activation_prefix"],
                original_activation_path=state["original_activation_path"]
            )
            if analysis_result:
                analysis_results.append(analysis_result)
        
        return analysis_results

    def detect_abnormal_components(self, state: DiagnosisState) -> List[str]:
        call_sequence_path = os.path.join(os.path.join(state.get("input_path", ""), f"call_sequence_{state['engine_type']}"), "call_sequence_rank0.log")
        call_sequence_aligned = []
        with open(call_sequence_path, "r") as f:
            lines = f.readlines()
            for line in lines:
                call_sequence_aligned.append((line.strip(), False))

        analysis_results = state["analysis_results"]
        
        suspicious_components = []
        for alignment_result in analysis_results:
            for component_name in alignment_result:
                for key in alignment_result[component_name]:
                    # if alignment_result[layer_idx][component_name]["abnormal"]:
                    ratio = alignment_result[component_name][key]["ratio"]
                    if key != "no_layer_idx":
                        # Find all placeholders in component_name and replace with key values
                        placeholders = re.findall(r"\{([^{}]+)\}", component_name)
                        if placeholders:
                            component_name_expanded = component_name
                            # key is a string: single value for one variable, or underscore-joined for multiple variables
                            # The key was created using sorted variable order, so we need to sort placeholders to match
                            sorted_placeholders = sorted(placeholders)
                            if len(placeholders) > 1:
                                # Multiple variables: split the key and map to sorted placeholders
                                # Check if splitting by "_" gives the right number of parts
                                key_str = str(key)
                                key_values = key_str.split("_")
                                if len(key_values) == len(placeholders):
                                    # Create a mapping from sorted placeholders to their values
                                    var_to_value = dict(zip(sorted_placeholders, key_values))
                                    # Replace placeholders in the original order they appear in component_name
                                    for var_name in placeholders:
                                        component_name_expanded = component_name_expanded.replace(f"{{{var_name}}}", str(var_to_value[var_name]))
                                else:
                                    # Fallback: treat as single variable (value might contain underscores)
                                    component_name_expanded = component_name_expanded.replace(f"{{{sorted_placeholders[0]}}}", key_str)
                            else:
                                # Single variable case
                                component_name_expanded = component_name_expanded.replace(f"{{{sorted_placeholders[0]}}}", str(key))
                        else:
                            component_name_expanded = component_name
                    else:
                        component_name_expanded = component_name
                    suspicious_components.append((component_name_expanded, ratio))

        for suspicious_component, ratio in suspicious_components:
            for i, component_call in enumerate(call_sequence_aligned):
                if suspicious_component == component_call[0]:
                    call_sequence_aligned[i] = (component_call[0], ratio)
        
        # save
        with open(os.path.join(state.get("output_path", ""), "call_sequence_aligned.json"), "w") as f:
            json.dump(call_sequence_aligned, f, indent=4)
        prompt = f"""
{state["problem_description"]}

You are given an *aligned call sequence* from one forward pass, where each entry is:
  - (component_path, error_ratio) or (component_path, false)

Definitions:
- error_ratio = max ||HF_fp32 - vLLM_bf16||_2 / max ||HF_bf16 - HF_fp32||_2 for that component’s output hidden states.
- Typical numerical noise exists: error ratio could be as large as {ERROR_THRESHOLD} even if the component is normal.
  A ratio > {ERROR_THRESHOLD} is *more likely* beyond pure roundoff, but not necessarily indicates a bug.

Important caveats:
- Some alignments are missing or failed: value == false means “no reliable aligned ratio for this component.”
- Some aligned ratios can be *false positives* due to incorrect alignment (even if the ratio is huge), looking like a spike.
- Your goal is to identify components that most likely correspond to an *actual bug root cause* in model/kernel implementation.

Core diagnostic rule (what you should optimize for):
- A true buggy component usually appears as a **change point**: after it, the error_ratio of modules immediately after it in the call sequence becomes **consistently higher than before** (sustained elevation), not just a one-off spike.

How to reason (be explicit and conservative):
1) Treat the call sequence as ordered in execution time.
2) Ignore entries with `false` when computing statistics, but they still matter for “where a change might have happened.”
3) Find candidate change points where:
   - The median/mean level of error_ratio shifts upward and stays elevated for a while.
   - The increase is sustained across multiple consecutive subsequent components that are after it in the call sequence.
4) Penalize “spiky” modules:
   - If a component has a very large ratio but surrounding components before/after do NOT show a sustained step-up, treat it as likely alignment noise.
   - Prefer the *earliest* component in the step-up region that is *plausible* (e.g., a projection/kernel boundary) over later components that merely inherit the error.
5) If multiple step-ups exist, prefer the earliest major step-up that explains the downstream elevated region.
6) You MUST provide evidence by citing nearby entries (a small window before/after) and describe why earlier components are unlikely.

Give ***top-{state["detection_top_k"]}*** suspected components, ordered by your confidence of it being the root cause buggy component.

### In-context examples (learn the pattern, module-agnostic)

Example A — SPIKE / ALIGNMENT NOISE (NOT root cause):
Call sequence excerpt:
1. op_01.layer_norm              -> 0.73
2. op_02.matmul                  -> 0.76
3. op_03.elementwise_nonlin       -> 640.50   (HUGE)
4. op_04.add_residual             -> 0.74     (IMMEDIATELY back to baseline)
5. op_05.layer_norm              -> 0.71
6. op_06.matmul                  -> 0.78

Correct interpretation:
- op_03 is a spike / likely false-positive alignment.
- Because the error immediately returns to baseline right after it and does NOT stay elevated.
Correct behavior:
- Do NOT rank op_03 as the most likely root cause.

Example B — TRUE CHANGE POINT / ROOT CAUSE (sustained elevation):
Call sequence excerpt:
1. op_10.layer_norm              -> 0.70
2. op_11.matmul                  -> 0.75
3. op_12.softmax_or_reduce        -> 12.40    (first jump)
4. op_13.matmul                  -> 25.10
5. op_14.add_residual             -> 21.80
6. op_15.layer_norm              -> 19.90
7. op_16.mlp_or_pointwise         -> 23.60
8. op_17.matmul                  -> 20.70

Correct interpretation:
- op_12 is the best root-cause candidate because it starts a sustained elevated region.
Correct behavior:
- Rank op_12 above later ops (op_13–op_17) since those likely inherit the error.

Example C — MULTIPLE SPIKES BEFORE A TRUE ROOT CAUSE:
Call sequence excerpt:
1. op_20.elementwise_nonlin       -> 510.0    (spike)
2. op_21.matmul                  -> 0.81     (back to baseline)
3. op_22.reduce_or_gather         -> 420.0    (spike)
4. op_23.add_residual             -> 0.79     (back to baseline)
5. op_24.embedding_or_indexing    -> 14.8     (true change point)
6. op_25.layer_norm              -> 28.9
7. op_26.matmul                  -> 24.7
8. op_27.mlp_or_pointwise         -> 26.2

Correct interpretation:
- Even if spikes are the largest numbers, they are NOT root cause if the sequence returns low immediately after.
- The true root cause is op_24 because it begins the sustained-high region.


Call sequence (ordered):
{call_sequence_aligned}
"""
        self.llm = self.llm.with_structured_output(FinalAnalysisOutput)
        response = self.llm.invoke([HumanMessage(content=prompt)])
        logging.info(response)
        detection_result = {
            "suspected_components": response.suspected_components,
            "reasoning": response.reasoning
        }
        return call_sequence_aligned, detection_result
