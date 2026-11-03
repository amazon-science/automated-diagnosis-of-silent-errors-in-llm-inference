"""Stage 1 - component mapping between the reference and target model implementations."""

from langgraph.prebuilt import create_react_agent
from typing import List, Dict, Any, Optional
from langchain_core.messages import HumanMessage
import os
import re
from itertools import product
import logging

from my_agent.utils.globals import (
    COMPONENT_MAPPING_MAX_RETRY,
    COMPONENT_MAPPING_MAX_ERROR_FEEDBACK,
)
from my_agent.output_structure import ComponentMappingOutput
from scripts.generate_module_tree.get_model_tree import expand_node, compressed_tree_to_pretty_string, remove_mapped_components
from my_agent.diagnosis.state import DiagnosisState
from my_agent.diagnosis.tools.code_lookup import get_class_definition


class ComponentMappingMixin:
    """Stage 1: map reference (HF) modules to target-engine modules."""

    def construct_component_mapping_prompt(self, hf_model_tree: Dict[str, Any], vllm_model_tree: Dict[str, Any], state: DiagnosisState) -> str:
        hf_index_path = os.path.join(state.get("input_path", ""), "reference_index")
        vllm_index_path = os.path.join(state.get("input_path", ""), "target_index")
        
        context = f"""
        Available components to investigate: 
        
        HF model tree with remaining unmapped components:
{hf_model_tree}
        
        vLLM model tree with remaining unmapped components:
{vllm_model_tree}
        
        Reference index (HuggingFace/HF): {hf_index_path}
        Target index (vLLM): {vllm_index_path}
        
        """
        
        prompt = f"""
        
        {state['problem_description']}
                
        You are given the PyTorch model architectures from two different inference frameworks. Your task is to construct a component-level mapping between the two architectures for diagnostic analysis.

        1. Mapping Requirements:
        - Components in the two frameworks may have different names, so you must output explicit mapping pairs.
        - Valid mappings include one-to-one and one-to-many (e.g., due to operator fusion).
        - Many-to-many mappings are NOT allowed. If you encounter a many-to-many relationship, you must decompose it into valid one-to-one or one-to-many mappings.

        2. Variable Abstraction:
        Component mappings often repeat across layers. To avoid redundancy, introduce variables (e.g., "i", "j") to represent repeated indices. For each variable, you must output the complete list of values it can take.

        3. Source-Code Extraction Tool:
        You are provided a tool that can extract the source code of a component given its class name and an index. Matching component names across frameworks does not guarantee semantic equivalence. If necessary, extract and compare source code to determine whether two components truly correspond.

        4. Mapping Strategy:
        - Begin with leaf components (lowest-level modules).
        - Then proceed upward to intermediate components.
        - Ensure alignment of both leaf and intermediate components when they are semantically equivalent.
        - You do not need to map the entire model tree, you can generate partial mappings for a trial. And the remaining unmapped components as well as the current mapping will be provided to you in the next iteration.
        - Make sure to provide the full name for the component, including the "root" prefix.

        5. Completeness Requirement:
        Every component in both frameworks must appear either:
        (a) in the component_mapping output, or
        (b) in the reasoning output explaining why the component does not have a valid mapping.
        You must provide reasoning separately for the HF side and the vLLM side.

        6. Examples:

        Example 1 (one-to-one mapping repeated across layers):
        component_mapping: [
        (["root.model.model.layers.{{"i"}}"], ["root.model.model.layers.{{"i"}}"])
        ]
        variables: {{"i": ["0", "1", ..., "47"]}}

        Example 2 (one-to-many mapping via operator fusion):
        component_mapping: [
        (
            [
            "root.model.layers.{{"i"}}.feed_forward.shared_expert.gate_proj",
            "root.model.layers.{{"i"}}.feed_forward.shared_expert.up_proj"
            ],
            [
            "root.model.layers.{{"i"}}.feed_forward.shared_expert.gate_up_proj"
            ]
        )
        ]
        variables: {{"i": ["0", "1", ..., "47"]}}

        Example 3 (one-to-many mapping for components present only in subset of layers):
        component_mapping: [
        (
            ["root.model.layers.{{"j"}}.self_attn.qk_norm"],
            [
            "root.model.layers.{{"j"}}.self_attn.q_norm",
            "root.model.layers.{{"j"}}.self_attn.k_norm"
            ]
        )
        ]
        variables: {{"j": ["0", "1", "2", "4", "5", "6", "8", "9", "10", "12", "13", "14", "16", "17", "18", "20", "21", "22", "24", "25", "26", "28", "29", "30", "32", "33", "34", "36", "37", "38", "40", "41", "42", "44", "45", "46"]}}

        Follow the instructions above to construct a complete and correct component mapping, assign variables where appropriate, and explain unmapped components in the reasoning field.

        {context}
        """
        
        # prompt = f"""
        
        # {state['problem_description']}

        # Initially, you will be given the torch model architecture from the two frameworks.
        # You will need to create a component mapping between the two model architectures to for further investigation.
        # Notice that the components are not necessarily named the same in both frameworks, so you need to output pairs.
        # Also, the components may not be one-to-one mapping but one-to-many mapping due to operator fusion.
        # Notably, many-to-many mapping is not allowed. When you encounter many-to-many mapping, you should split the components into either one-to-one mapping or one-to-many mapping.
        
        # Additionally, component mapping may be repeatitive across layers, so you need to assign variables to the components to avoid repeatitive mapping. But you should also give the values the variables can take.
        
        # You are also given tool to extract the source code of the components given the component class name and the index you would like to search. Sometimes component names that match between the two frameworks do not necessarily mean they are the same component. You can use this tool to extract the source code of the components and compare the source code to decide if they are the same component.
        
        # You can start with the leaf components and then move to the intermediate components. You should align not only the leaf components but also the intermediate components.
        
        # You should make sure that all components are either mapped in the component mapping or explained in the reasoning why they are not mapped.
        
        # Here are some examples.
        # Example 1 (one-to-one mapping), layer mapping, which is valid for every layer:
        # component_mapping: [(["model.model.layers.{{"i"}}"], ["model.model.layers.{{"i"}}"])], (...)]
        # variables: {{"i": ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "21", "22", "23", "24", "25", "26", "27", "28", "29", "30", "31", "32", "33", "34", "35", "36", "37", "38", "39", "40", "41", "42", "43", "44", "45", "46", "47"], ...}}
        
        # Example 2 (one-to-many mapping), gate and up projection of the shared expert in the MoE layer:
        # component_mapping: [(["model.model.layers.{{"i"}}.feed_forward.shared_expert.gate_proj", "model.model.layers.{{"i"}}.feed_forward.shared_expert.up_proj"], ["model.model.layers.{{"i"}}.feed_forward.shared_expert.gate_up_proj"]), (...)]
        # variables: {{"i": ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "21", "22", "23", "24", "25", "26", "27", "28", "29", "30", "31", "32", "33", "34", "35", "36", "37", "38", "39", "40", "41", "42", "43", "44", "45", "46", "47"], ...}}
        
        # Example 3 (one-to-many mapping), qk_norm that is only used in part of the layers:
        # component_mapping: [(["model.model.layers.{{"j"}}.self_attn.qk_norm"], ["model.model.layers.{{"j"}}.self_attn.q_norm", "model.model.layers.{{"j"}}.self_attn.k_norm"])], (...)]
        # variables: {{"j": ["0", "1", "2", "4", "5", "6", "8", "9", "10", "12", "13", "14", "16", "17", "18", "20", "21", "22", "24", "25", "26", "28", "29", "30", "32", "33", "34", "36", "37", "38", "40", "41", "42", "44", "45", "46"], ...}}
        
        
        # {context}
       
        # """
        
        return prompt

    def create_component_mapping(self, state: DiagnosisState) -> Dict[str, Any]:
        """
        Create a component mapping between the HF and vLLM model trees.
        
        This method now mirrors the trial / feedback structure used in
        `align_hf_activations_with_vllm`:
        - Within a trial, we provide iterative error feedback to the model.
        - After a configurable number of feedback attempts in a single trial,
          we start a fresh trial from scratch.
        - We run up to a configurable maximum number of trials in total.
        - Across all trials, we keep track of the best (largest) valid partial
          mapping and return it if we never achieve a fully complete mapping.
        """

        component_mapping_agent = create_react_agent(
            model=self.llm,
            tools=[get_class_definition],
            prompt="You are a helpful assistant",
            response_format=ComponentMappingOutput,
        )

        best_response: Optional[ComponentMappingOutput] = None
        best_num_pairs: int = -1
        
        max_trials = state.get("component_mapping_max_trials", COMPONENT_MAPPING_MAX_RETRY)
        max_feedback_attempts = state.get(
            "component_mapping_max_error_feedback",
            COMPONENT_MAPPING_MAX_ERROR_FEEDBACK,
        )

        for trial_idx in range(1, max_trials + 1):
            logging.info(f"[create_component_mapping] Starting trial {trial_idx}/{max_trials} with {max_feedback_attempts} feedback attempts")

            # Per-trial state
            mapped_components_hf: List[str] = []
            mapped_components_vllm: List[str] = []
            variables: Dict[str, Any] = {}
            error_feedback = ""
            feedback_attempt = 0
            response: Optional[ComponentMappingOutput] = None

            # Within a trial, allow multiple feedback refinements
            while feedback_attempt < max_feedback_attempts:
                try:
                    hf_model_tree = remove_mapped_components(
                        state["model_tree"]["hf"], mapped_components_hf, variables
                    )
                    vllm_model_tree = remove_mapped_components(
                        state["model_tree"]["vllm"], mapped_components_vllm, variables
                    )
                    hf_model_tree_str = compressed_tree_to_pretty_string(hf_model_tree)
                    vllm_model_tree_str = compressed_tree_to_pretty_string(vllm_model_tree)
                    prompt = self.construct_component_mapping_prompt(
                        hf_model_tree_str, vllm_model_tree_str, state
                    )
                    inputs = {"messages": [HumanMessage(content=prompt + error_feedback)]}
                    # logging.info(f"[create_component_mapping] Input: {inputs}")
                    raw_response = component_mapping_agent.invoke(
                        inputs, config={"callbacks": [self.token_tracer]}
                    )
                    # logging.info(f"[create_component_mapping] Raw Response: {raw_response}")
                    response = raw_response["structured_response"]
                    logging.info(f"[create_component_mapping] Response: {response}")

                    completed, flat_paths_hf_matched, flat_paths_vllm_matched = (
                        self.check_component_mapping(response, state["model_tree"])
                    )

                    # Checkpoint the best (largest) mapping we've seen so far
                    try:
                        current_num_pairs = len(response.component_mapping)
                    except Exception:
                        current_num_pairs = 0

                    if current_num_pairs > best_num_pairs:
                        best_num_pairs = current_num_pairs
                        best_response = response
                        
                    logging.info(f"[create_component_mapping] Trial {trial_idx} Attempt {feedback_attempt + 1} Best component mapping length: {best_num_pairs}")

                    if completed:
                        logging.info(
                            f"[create_component_mapping] Successfully created full component mapping "
                            f"in trial {trial_idx} after {feedback_attempt + 1} feedback attempts, with{len(response.component_mapping)} component pairs"
                        )
                        return response

                    # Not completed: update mapped components and provide error feedback
                    mapped_components_hf = [
                        key for key in flat_paths_hf_matched if flat_paths_hf_matched[key]
                    ]
                    mapped_components_vllm = [
                        key for key in flat_paths_vllm_matched if flat_paths_vllm_matched[key]
                    ]

                    error_feedback = f"""
The component mapping is not complete, please try again.
Current component mapping: {response}
"""
                    feedback_attempt += 1
                    logging.info(
                        f"[create_component_mapping] Incomplete mapping in trial {trial_idx}, "
                        f"feedback attempt {feedback_attempt}/{max_feedback_attempts}"
                    )
                    continue

                except Exception as e:
                    logging.info(f"[create_component_mapping] Error in trial {trial_idx}, "
                                 f"feedback attempt {feedback_attempt + 1}: {e}")
                    error_feedback = f"""
Here is the incorrect component mapping generated from the previous attempt: {response}

The error is: {e}

Please try again.
"""
                    feedback_attempt += 1
                    if feedback_attempt >= max_feedback_attempts:
                        logging.info(
                            f"[create_component_mapping] Reached max feedback attempts "
                            f"{max_feedback_attempts} for trial {trial_idx}, "
                            "starting a new trial if available"
                        )
                        break
                    continue

            # Finished this trial (either by exhausting feedback attempts or breaking early)
            logging.info(
                f"[create_component_mapping] Ending trial {trial_idx}/{max_trials} with "
                f"{feedback_attempt} feedback attempts; best so far has "
                f"{best_num_pairs if best_num_pairs >= 0 else 0} component pairs"
            )

        # All trials exhausted; return the best partial mapping we've seen.
        if best_response is not None:
            logging.info(
                f"[create_component_mapping] Failed to create full component mapping after "
                f"{max_trials} trials; returning best partial mapping with "
                f"{best_num_pairs} component pairs"
            )
            return best_response

        # No valid response at all
        logging.info(
            f"[create_component_mapping] Failed to create any valid component mapping "
            f"after {max_trials} trials; returning empty ComponentMappingOutput"
        )
        return ComponentMappingOutput(component_mapping=[], variables={})

    def get_class_name_from_module_tree(self, component_name: str, component_mappings: ComponentMappingOutput, module_tree: Dict[str, str]) -> str:
        # Find all placeholders in the component name
        placeholders = re.findall(r"\{([^{}]+)\}", component_name)
        if placeholders:
            missing_vars = [var for var in placeholders if var not in component_mappings.variables]
            if missing_vars:
                raise ValueError(f"Variable(s) {missing_vars} not found in variables: {component_mappings.variables}")
            # Expand all variables by taking the first value for each
            component_name_expanded = component_name
            for var_name in placeholders:
                var_value = component_mappings.variables[var_name][0]
                component_name_expanded = component_name_expanded.replace(f"{{{var_name}}}", var_value)
        else:
            component_name_expanded = component_name
        component_name_splits = component_name_expanded.split(".")
        current_nodes = [module_tree]
        found = False
        for split in component_name_splits:
            for node in current_nodes:
                if split in node["names"]:
                    if split == component_name_splits[-1]:
                        return node["type"]
                    current_nodes = node["children"]
                    found = True
                    break
            if not found:
                raise ValueError(f"Component {component_name_expanded} not found in module tree")
        raise ValueError(f"Component {component_name_expanded} not found in module tree")

    def check_component_mapping(self, output: ComponentMappingOutput, model_tree: Dict[str, Dict[str, str]]) -> bool:
        """
        Check if the component mapping is valid.
        Return True if the mapping is complete, otherwise return False.
        Throw exception if the mapping is invalid.
        """
        
        hf_architecture = model_tree["hf"]
        vllm_architecture = model_tree["vllm"]
        
        component_mapping = output.component_mapping
        variables = output.variables
        reasoning = output.reasoning
        
        flat_paths_hf = expand_node(hf_architecture, prefixes=[""])
        flat_paths_vllm = expand_node(vllm_architecture, prefixes=[""])
        # logging.info(f"Flat paths HF: {flat_paths_hf}")
        # logging.info(f"Flat paths vLLM: {flat_paths_vllm}")
        flat_paths_hf_matched = {}
        flat_paths_vllm_matched = {}
        for flat_path in flat_paths_hf:
            flat_paths_hf_matched[flat_path] = False
        for flat_path in flat_paths_vllm:
            flat_paths_vllm_matched[flat_path] = False
            
        def expand_component(component: str) -> List[str]:
            """Expand a component pattern containing one or more {variables}.

            We generate the cartesian product of all variable values so every
            placeholder in the component string is substituted simultaneously.
            """
            placeholders = re.findall(r"\{([^{}]+)\}", component)
            if not placeholders:
                return [component]

            missing_vars = [var for var in placeholders if var not in variables]
            if missing_vars:
                raise ValueError(f"Variable(s) {missing_vars} not found in variables: {variables}")

            value_lists = [variables[var] for var in placeholders]
            expanded_components: List[str] = []
            for values in product(*value_lists):
                expanded = component
                for var, val in zip(placeholders, values):
                    expanded = expanded.replace(f"{{{var}}}", val)
                expanded_components.append(expanded)
            return expanded_components

        for component_pair in component_mapping:
            if len(component_pair) != 2:
                raise ValueError(f"Component pair must have two components: {component_pair}")
            hf_components = component_pair[0]
            vllm_components = component_pair[1]
            
            if len(hf_components) != 1 and len(vllm_components) != 1:
                raise ValueError(f"Component pair must be one-to-one mapping or one-to-many mapping, but got {hf_components} and {vllm_components}")
            
            
            try:
                # check if it is a valid HF component
                for hf_component in hf_components:
                    hf_components_expanded = expand_component(hf_component)
        
                    
                    for hf_component_expanded in hf_components_expanded:
                        hf_component_splits = hf_component_expanded.split(".")
                        current_nodes = [hf_architecture]
                        for split in hf_component_splits:
                            found = False
                            for node in current_nodes:
                                if split in node["names"]:
                                    current_nodes = node["children"]
                                    found = True
                                    break
                            if not found:
                                raise ValueError(f"HF component {hf_component_expanded} not found in architecture")
                        
                # check if the HF component is in the flat paths
                for hf_component_expanded in hf_components_expanded:
                    if hf_component_expanded in flat_paths_hf_matched:
                        flat_paths_hf_matched[hf_component_expanded] = True
                        
            except Exception as e:
                raise ValueError(f"[check_component_mapping] Error in checking HF component {hf_components}: {e}")
                
            
            try:
                # check if it is a valid vLLM component
                for vllm_component in vllm_components:
                    vllm_components_expanded = expand_component(vllm_component)
                        
                    for vllm_component_expanded in vllm_components_expanded:
                        vllm_component_splits = vllm_component_expanded.split(".")
                        current_nodes = [vllm_architecture]
                        for split in vllm_component_splits:
                            found = False
                            for node in current_nodes:
                                if split in node["names"]:
                                    current_nodes = node["children"]
                                    found = True
                                    break
                            if not found:
                                raise ValueError(f"vLLM component {vllm_component_expanded} not found in architecture")
                            
                    # check if the vLLM component is in the flat paths
                    for vllm_component_expanded in vllm_components_expanded:
                        if vllm_component_expanded in flat_paths_vllm_matched:
                            flat_paths_vllm_matched[vllm_component_expanded] = True
            except Exception as e:
                raise ValueError(f"[check_component_mapping] Error in checking vLLM component {vllm_components}: {e}")
                
        
        
        
        # for leftovers in reasoning, check if they are in the flat paths
        for hf_component in reasoning["HF"]:
            hf_components_expanded = expand_component(hf_component)
            
            for hf_component_expanded in hf_components_expanded:
                if hf_component_expanded in flat_paths_hf_matched:
                    flat_paths_hf_matched[hf_component_expanded] = True
            
        for vllm_component in reasoning["vLLM"]:
            vllm_components_expanded = expand_component(vllm_component)
            
            for vllm_component_expanded in vllm_components_expanded:
                if vllm_component_expanded in flat_paths_vllm_matched:
                    flat_paths_vllm_matched[vllm_component_expanded] = True
                        
        # finally, anything not mapped in the flat paths is an error
        for key in flat_paths_hf_matched:
            if not flat_paths_hf_matched[key]:
                # raise ValueError(f"HF component {key} is not found in either the component mapping or the reasoning")
                return False, flat_paths_hf_matched, flat_paths_vllm_matched
        for key in flat_paths_vllm_matched:
            if not flat_paths_vllm_matched[key]:
                # raise ValueError(f"vLLM component {key} is not found in either the component mapping or the reasoning")
                return False, flat_paths_hf_matched, flat_paths_vllm_matched
                
        return True, flat_paths_hf_matched, flat_paths_vllm_matched

    def verify_component_mapping_with_gt(self, output: ComponentMappingOutput, gt_component_mapping: List) -> bool:
        
        def is_valid_component_pair(hf_components: List[str], vllm_components: List[str]) -> bool:
            hf_components_set = set()
            for hf_component in hf_components:
                hf_components_set.add(hf_component)
            vllm_components_set = set()
            for vllm_component in vllm_components:
                vllm_components_set.add(vllm_component)
            for gt_component_pair in gt_component_mapping:
                if set(gt_component_pair[0]) == hf_components_set and set(gt_component_pair[1]) == vllm_components_set:
                    return True
            return False
        
        component_mapping = output.component_mapping
        variables = output.variables
        
        for component_pair in component_mapping:
            hf_components = component_pair[0]
            vllm_components = component_pair[1]
            
            # special case for rotary_emb
            if "rotary_emb" in hf_components[0]:
                if len(vllm_components) != 1:
                    raise ValueError(f"rotary_emb should be one-to-one mapping, but got {vllm_components}")
                if "{" and "}" not in vllm_components[0]:
                    raise ValueError(f"rotary_emb should use variable for vLLM, but got {vllm_components}")
                var_name = vllm_components[0].split("{")[1].split("}")[0]
                if var_name not in variables:
                    raise ValueError(f"Variable {var_name} not found in variables: {variables}")
                vllm_component_expanded = [vllm_components[0].replace("{" + var_name + "}", var_value) for var_value in variables[var_name]]
                if not is_valid_component_pair(hf_components, vllm_component_expanded):
                    raise ValueError(f"Invalid component pair: {hf_components} and {vllm_component_expanded}")
                continue
                
            
            has_variable = False
            for hf_component in hf_components:
                if "{" and "}" in hf_component:
                    has_variable = True
                    var_name = hf_component.split("{")[1].split("}")[0]
                    break
            # all components should use the same variable
            if has_variable:
                for hf_component in hf_components:
                    if "{" + var_name + "}" not in hf_component:
                        raise ValueError(f"Variable {var_name} not found in hf_components: {hf_component}")
                for vllm_component in vllm_components:
                    if "{" + var_name + "}" not in vllm_component:
                        raise ValueError(f"Variable {var_name} not found in vllm_components: {vllm_component}")
                for var_value in variables[var_name]:
                    hf_components_expanded = [hf_component.replace("{" + var_name + "}", var_value) for hf_component in hf_components]
                    vllm_components_expanded = [vllm_component.replace("{" + var_name + "}", var_value) for vllm_component in vllm_components]
                    if not is_valid_component_pair(hf_components_expanded, vllm_components_expanded):
                        raise ValueError(f"Invalid component pair: {hf_components_expanded} and {vllm_components_expanded}")
            else:
                if not is_valid_component_pair(hf_components, vllm_components):
                    raise ValueError(f"Invalid component pair: {hf_components} and {vllm_components}")
