"""Entry point: load a config, build the initial state and run the diagnosis graph."""

import argparse
import json
import os
from pathlib import Path
from datetime import datetime
import logging

from my_agent.utils.globals import (
    COMPONENT_MAPPING_MAX_RETRY,
    COMPONENT_MAPPING_MAX_ERROR_FEEDBACK,
    ACTIVATION_ALIGNMENT_MAX_RETRY,
    ACTIVATION_ALIGNMENT_MAX_ERROR_FEEDBACK,
)
from my_agent.diagnosis import runtime
from common.config import load_config
from common.llm import make_chat_model
from my_agent.diagnosis.state import DiagnosisState
from my_agent.diagnosis.workflow import diagnose_workflow


def run_diagnosis(cfg: dict):
    if cfg.get("mode", "training") == "training":
        problem_description = f"""
        You are an expert ML engineer diagnosing accuracy differences between inference engines for the same model.
        You are learning how to map the components between the two frameworks that have correct implementations of the same model.
        Your task is to create an accurate component mapping between the two frameworks and align the activations of the two frameworks for each component pair within the error threshold.
        """
        
    elif cfg.get("mode", "training") == "baseline_1":
        with open(cfg["issue_path"], "r") as f:
            issue_description = f.read()
        problem_description = issue_description
    elif cfg.get("mode", "training") == "diagnosis":
        problem_description = f"""
        You are an expert ML engineer diagnosing accuracy differences between inference engines for the same model.
        Your overall task is to analyze the activation differences based on the information provided to you as well as your own investigation and decide which components may be causing the accuracy differences.
        Issue description: 
        """
        with open(cfg["issue_path"], "r") as f:
            issue_description = f.read()
        problem_description += "\n\n" + issue_description
    else:
        raise ValueError(f"Invalid mode: {cfg.get('mode', 'training')}, supported modes: training, baseline_1, diagnosis \n training: training stage to obtain mapping and alignment examples \n baseline_1: static agent that diagnose bugs using only model implementation \n diagnosis: full diagnosis workflow to diagnose bugs")
    
    if cfg.get("checkpoint_path", None) is not None:
        if cfg.get("next_stage", None) is None or cfg.get("next_stage", None) not in ["mapping", "alignment", "analysis"]:
            raise ValueError(f"next_stage is required to determine what stage to execute when checkpoint_path is provided, valid values: mapping, alignment, analysis, got {cfg.get('next_stage', None)}")
        
    model_tree = {
        "hf": json.load(open(cfg["input_path"] + "/model_tree/model_tree_hf.json")),
        "vllm": json.load(open(cfg["input_path"] + f"/model_tree/model_tree_{cfg['engine_type']}.json"))
    }
    
    model_impl = {}
    with open(cfg["input_path"] + "/model_impl/model_impl_hf.py", "r") as f:
        model_impl["hf"] = f.read()
    with open(cfg["input_path"] + f"/model_impl/model_impl_{cfg['engine_type']}.py", "r") as f:
        model_impl["vllm"] = f.read()
    
    runtime.MODEL = make_chat_model(
        provider=cfg.get("provider", "bedrock"),
        model_id=cfg.get("model_id"),
        model_kwargs=cfg.get("model_kwargs"),
        region=cfg.get("region"),
    )

    # Set alignment example root from config.
    if "knowledge_base_path" in cfg:
        logging.info(f"Knowledge base path: {cfg['knowledge_base_path']}")
        runtime.KNOWLEDGE_BASE_PATH = Path(cfg["knowledge_base_path"])
    else:
        runtime.KNOWLEDGE_BASE_PATH = None
        logging.info("No knowledge base path provided, do not use alignment examples")
    
    state = DiagnosisState(
        problem_description=problem_description,
        error_threshold=cfg["error_threshold"],
        next_action="analyze",
        component_mappings={},
        model_tree=model_tree,
        model_impl=model_impl,
        input_path=cfg["input_path"],
        output_path=cfg["output_path"],
        activation_path=cfg.get("activation_path", cfg["input_path"]),
        ref_activation_prefix=cfg.get("ref_activation_prefix", ""),
        ref_activation_fp32_prefix=cfg.get("ref_activation_fp32_prefix", ""),
        target_activation_prefix=cfg.get("target_activation_prefix", ""),
        suspicious_components=[],
        mode=cfg.get("mode", "training"),
        engine_type=cfg.get("engine_type", "vllm"),
        checkpoint_path=cfg.get("checkpoint_path", None),
        next_stage=cfg.get("next_stage", None),
        knowledge_base_path=cfg.get("knowledge_base_path", None),
        original_target_activation_prefix=cfg.get("original_target_activation_prefix", None),
        original_activation_path=cfg.get("original_activation_path", None),
        alignment_multi_trial=cfg.get("alignment_multi_trial", False),
        alignment_multi_trial_count=cfg.get("alignment_multi_trial_count", 3),
        component_mapping_max_trials=cfg.get(
            "component_mapping_max_trials",
            COMPONENT_MAPPING_MAX_RETRY,
        ),
        component_mapping_max_error_feedback=cfg.get(
            "component_mapping_max_error_feedback",
            COMPONENT_MAPPING_MAX_ERROR_FEEDBACK,
        ),
        activation_alignment_max_trials=cfg.get(
            "activation_alignment_max_trials",
            ACTIVATION_ALIGNMENT_MAX_RETRY,
        ),
        activation_alignment_max_error_feedback=cfg.get(
            "activation_alignment_max_error_feedback",
            ACTIVATION_ALIGNMENT_MAX_ERROR_FEEDBACK,
        ),
        detection_top_k=cfg.get(
            "detection_top_k",
            3
        )
    )
    graph = diagnose_workflow()
    return graph.invoke(state)


def main():
    parser = argparse.ArgumentParser(description="Run the Ekka diagnosis workflow on a bug case.")
    parser.add_argument("--config_path", type=str, required=True, help="Path to a diagnosis config YAML (e.g. bug-cases/vllm/vllm-26812/config_with_kb.yaml)")
    args = parser.parse_args()

    # Expands ${EKKA_ROOT} / ${EKKA_ACTIVATION_DIR} and other environment variables in config values.
    cfg = load_config(args.config_path)

    output_path = cfg["output_path"] + "_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    cfg["output_path"] = output_path
    log_path = output_path + "/diagnosis_log.log"
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    logging.basicConfig(
        filename=log_path,
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    logging.info(f"Config: {cfg}")

    run_diagnosis(cfg)
