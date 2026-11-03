#!/usr/bin/env python3
"""
Run vLLM from a YAML config with timestamped logging.
Supports two modes:
  - generate (default): standard text / token generation
  - reranker: sequence scoring via LLM.score for query-document pairs

Usage:
    python driver_vllm_v2.py --config config.yaml
"""

import os
import argparse
import yaml
import random
import numpy as np
import torch
import logging
from datetime import datetime

from vllm import LLM, SamplingParams, TokensPrompt


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def resolve_output_path(output_path_from_cfg: str | None) -> str:
    """Return a concrete log file path, always timestamped when directory-like."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    if output_path_from_cfg is None:
        # default: logs/vllm_run_<ts>.log
        base_dir = "logs"
        os.makedirs(base_dir, exist_ok=True)
        return os.path.join(base_dir, f"vllm_run_{ts}.log")

    # user gave something
    # 1) explicit file with extension -> use as-is
    # 2) directory-like -> drop timestamp file inside
    # simple check: has '.', not at start, and not a dir
    is_file_like = (
        "." in os.path.basename(output_path_from_cfg)
        and not output_path_from_cfg.endswith("/")
    )

    if is_file_like:
        # ensure parent exists
        parent = os.path.dirname(output_path_from_cfg)
        if parent:
            os.makedirs(parent, exist_ok=True)
        return output_path_from_cfg
    else:
        # treat as directory
        os.makedirs(output_path_from_cfg, exist_ok=True)
        return os.path.join(output_path_from_cfg, f"vllm_run_{ts}.log")


def setup_logger(final_output_path: str):
    """Set up a logger that logs to both console and file."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(final_output_path, mode="w"),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def maybe_set_env(env_cfg: dict, logger: logging.Logger):
    if not env_cfg:
        return
    for k, v in env_cfg.items():
        os.environ[str(k)] = str(v)
        logger.info(f"Set environment variable: {k}={v}")


def build_llm(cfg: dict, logger: logging.Logger, mode: str) -> LLM:
    model_cfg = cfg["model"]
    model_name = model_cfg["name"]

    llm_kwargs = {}
    for key in model_cfg.keys():
        if key == "name":
            continue
        llm_kwargs[key] = model_cfg[key]

    if mode == "reranker" and "task" not in llm_kwargs:
        llm_kwargs["task"] = "score"

    logger.info(f"Initializing LLM with model={model_name}, kwargs={llm_kwargs} (mode={mode})")
    llm = LLM(model_name, **llm_kwargs)
    return llm


def build_sampling(cfg: dict) -> SamplingParams:
    samp_cfg = cfg.get("sampling", {})
    return SamplingParams(
        temperature=samp_cfg.get("temperature", 0.0),
        top_p=samp_cfg.get("top_p", 1.0),
        max_tokens=samp_cfg.get("max_tokens", 512),
        stop=samp_cfg.get("stop", None),
    )


DEFAULT_RERANKER_INSTRUCTION = (
    "Given a web search query, retrieve relevant passages that answer the query"
)
DEFAULT_RERANKER_PREFIX = (
    '<|im_start|>system\nJudge whether the Document meets the requirements based on the '
    'Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n'
    "<|im_start|>user\n"
)
DEFAULT_RERANKER_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
DEFAULT_QUERY_TEMPLATE = "{prefix}<Instruct>: {instruction}\n<Query>: {query}\n"
DEFAULT_DOCUMENT_TEMPLATE = "<Document>: {doc}{suffix}"


def build_reranker_inputs(reranker_cfg: dict) -> tuple[list[str], list[str]]:
    queries = reranker_cfg.get("queries", None)
    documents = reranker_cfg.get("documents", None)
    if queries is None or documents is None:
        raise ValueError("reranker.queries and reranker.documents must be provided in YAML")
    if len(queries) != len(documents):
        raise ValueError("reranker.queries and reranker.documents must have the same length")

    instruction = reranker_cfg.get("instruction", DEFAULT_RERANKER_INSTRUCTION)
    prefix = reranker_cfg.get("prefix", DEFAULT_RERANKER_PREFIX)
    suffix = reranker_cfg.get("suffix", DEFAULT_RERANKER_SUFFIX)
    query_template = reranker_cfg.get("query_template", DEFAULT_QUERY_TEMPLATE)
    document_template = reranker_cfg.get("document_template", DEFAULT_DOCUMENT_TEMPLATE)

    formatted_queries = [
        query_template.format(prefix=prefix, instruction=instruction, query=query)
        for query in queries
    ]
    formatted_documents = [
        document_template.format(doc=doc, suffix=suffix)
        for doc in documents
    ]
    return formatted_queries, formatted_documents


def run_reranker(llm: LLM, reranker_cfg: dict, logger: logging.Logger):
    if not reranker_cfg:
        raise ValueError("reranker mode selected but reranker config block is missing")

    queries, documents = build_reranker_inputs(reranker_cfg)
    logger.info(f"Scoring {len(queries)} query-document pair(s).")
    outputs = llm.score(queries, documents)

    logger.info("=== Reranker Scores ===")
    for i, output in enumerate(outputs):
        score = output.outputs.score
        logger.info(f"Pair {i}:")
        logger.info(f"Query:\n{queries[i]}")
        logger.info(f"Document:\n{documents[i]}")
        logger.info(f"Score: {score}")
    logger.info("=" * 40)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="YAML config path")
    args = parser.parse_args()

    cfg = load_config(args.config)

    # figure out log path (with timestamp if needed)
    output_path_from_cfg = cfg.get("output_path", None)
    final_log_path = resolve_output_path(output_path_from_cfg)

    # setup logger
    logger = setup_logger(final_log_path)
    logger.info(f"Loaded config from {args.config}")
    logger.info(f"Logs will be saved to {final_log_path}")

    # env
    maybe_set_env(cfg.get("env", {}), logger)

        
    # seed
    seed = cfg.get("seed", 42)
    set_seed(seed)
    logger.info(f"Random seed set to {seed}")

    # mode
    driver_mode = cfg.get("mode", "generate")
    logger.info(f"Driver mode: {driver_mode}")

    # model
    llm = build_llm(cfg, logger, driver_mode)

    if driver_mode == "reranker":
        reranker_cfg = cfg.get("reranker", {})
        run_reranker(llm, reranker_cfg, logger)
    else:
        # sampling
        sampling_params = build_sampling(cfg)

        # prompts
        prompt_cfg = cfg.get("prompt", {})
        prompt_mode = prompt_cfg.get("mode", "text")

        if prompt_mode == "text":
            prompts = prompt_cfg.get("texts", [])
            if not prompts:
                raise ValueError("prompt.mode='text' but no prompt.texts provided")
            logger.info(f"Generating for {len(prompts)} text prompt(s).")
            outputs = llm.generate(prompts, sampling_params)
        elif prompt_mode == "tokens":
            token_lists = prompt_cfg.get("token_ids", [])
            if not token_lists:
                raise ValueError("prompt.mode='tokens' but no prompt.token_ids provided")
            logger.info(f"Generating for {len(token_lists)} token prompt(s).")
            token_prompts = [TokensPrompt(prompt_token_ids=toks) for toks in token_lists]
            outputs = llm.generate(token_prompts, sampling_params=sampling_params)
        else:
            raise ValueError(f"Unknown prompt.mode={prompt_mode}")

        # log results
        for i, out in enumerate(outputs):
            req_out = out.outputs[0]
            logger.info("=" * 40)
            logger.info(f"Prompt {i}:")
            if prompt_mode == "text":
                logger.info(f"Input:\n{prompt_cfg['texts'][i]}")
            else:
                logger.info(f"Input token ids:\n{prompt_cfg['token_ids'][i]}")
            logger.info(f"Generated:\n{req_out.text}")
            logging.info(f"Generated IDs: {req_out.token_ids}")

    logger.info("### vLLM run completed successfully.")


if __name__ == "__main__":
    main()
