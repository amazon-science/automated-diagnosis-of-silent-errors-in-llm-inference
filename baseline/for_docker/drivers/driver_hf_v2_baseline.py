#!/usr/bin/env python3
"""
Run Hugging Face models from a YAML config with timestamped logging.
Supports two modes:
  - generate: Text generation with AutoModelForCausalLM
  - reranker: Sequence classification with AutoModelForSequenceClassification for query-document reranking

Usage:
    python driver_hf_v2.py --config config.yaml
"""

import os
import sys
import json
import argparse
import yaml
import logging
from datetime import datetime

import torch
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer, set_seed

# ---------- utilities ----------

def set_global_seed(seed: int):
    set_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def resolve_output_path(output_path_from_cfg: str | None) -> str:
    """Return a concrete log file path, always timestamped when directory-like."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    if output_path_from_cfg is None:
        base_dir = "logs"
        os.makedirs(base_dir, exist_ok=True)
        return os.path.join(base_dir, f"hf_run_{ts}.log")

    is_file_like = (
        "." in os.path.basename(output_path_from_cfg)
        and not output_path_from_cfg.endswith("/")
    )

    if is_file_like:
        parent = os.path.dirname(output_path_from_cfg)
        if parent:
            os.makedirs(parent, exist_ok=True)
        return output_path_from_cfg
    else:
        os.makedirs(output_path_from_cfg, exist_ok=True)
        return os.path.join(output_path_from_cfg, f"hf_run_{ts}.log")


def setup_logger(final_output_path: str):
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


# ---------- model / gen ----------

def load_hf_model_and_tokenizer(model_cfg: dict, logger: logging.Logger, mode: str = "generate"):
    model_name = model_cfg["name"]

    # For reranker, use left padding; for causal LM, use right padding (default)
    padding_side = "left" if mode == "reranker" else "right"
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side=padding_side)

    # pass through extra kwargs to from_pretrained
    model_kwargs = {}
    for k, v in model_cfg.items():
        if k == "name":
            continue
        model_kwargs[k] = v

    logger.info(f"Loading HF model {model_name} with kwargs={model_kwargs} (mode={mode})")
    
    if mode == "reranker":
        model = AutoModelForSequenceClassification.from_pretrained(model_name, **model_kwargs)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    
    model.eval()
    return model, tokenizer


def generate_text(
    model,
    tokenizer,
    prompts: list[str],
    max_new_tokens: int,
    logger: logging.Logger,
    device: str | None = None,
    generation_cfg: dict | None = None,
):
    inputs = tokenizer(prompts, return_tensors="pt")
    if device is not None:
        inputs = {k: v.to(device) for k, v in inputs.items()}
    else:
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": False,  # greedy like your vLLM example
    }

    # allow user to override sampling in YAML
    if generation_cfg:
        # only pick known safe keys
        for k in generation_cfg.keys():
            gen_kwargs[k] = generation_cfg[k]

    with torch.no_grad():
        output_ids = model.generate(**inputs, **gen_kwargs, return_dict_in_generate=True, output_scores=False)
    generated_ids = output_ids.sequences[:, inputs["input_ids"].shape[1]:]
    decoded = tokenizer.batch_decode(generated_ids, skip_special_tokens=False)
    if int(os.environ["RANK"]) == 0:
        for i, text in enumerate(decoded):
            logger.info("=" * 40)
            logger.info(f"Prompt {i}:")
            logger.info(f"Input:\n{prompts[i]}")
            logger.info(f"Generated:\n{text}")
            logging.info(f"Generated IDs: {generated_ids[i]}")
    return decoded


# ---------- reranker functions ----------

def format_instruction(instruction: str | None, query: str, doc: str, prefix: str, suffix: str) -> str:
    """Format query-document pair for reranker model."""
    if instruction is None:
        instruction = "Given a web search query, retrieve relevant passages that answer the query"
    output = f"{prefix}<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}{suffix}"
    return output


def process_inputs(
    tokenizer,
    pairs: list[str],
    prefix_tokens: list[int],
    suffix_tokens: list[int],
    max_length: int,
    model_device,
):
    """Process reranker inputs with prefix/suffix tokens."""
    inputs = tokenizer(
        pairs,
        padding=False,
        truncation='longest_first',
        return_attention_mask=False,
        max_length=max_length - len(prefix_tokens) - len(suffix_tokens)
    )
    
    # Add prefix and suffix tokens
    for i, ele in enumerate(inputs['input_ids']):
        inputs['input_ids'][i] = ele
    
    # Pad to max_length
    inputs = tokenizer.pad(inputs, return_tensors="pt", max_length=max_length)
    
    # Move to device
    for key in inputs:
        inputs[key] = inputs[key].to(model_device)
    
    return inputs


def run_reranker(
    model,
    tokenizer,
    queries: list[str],
    documents: list[str],
    instruction: str | None,
    prefix: str,
    suffix: str,
    max_length: int,
    logger: logging.Logger,
    device: str | None = None,
):
    """Run reranker model on query-document pairs."""
    # Encode prefix and suffix tokens
    prefix_tokens = tokenizer.encode(prefix, add_special_tokens=False)
    suffix_tokens = tokenizer.encode(suffix, add_special_tokens=False)
    
    # Format instruction pairs
    pairs = [
        format_instruction(instruction, query, doc, prefix, suffix)
        for query, doc in zip(queries, documents)
    ]
    
    # Process inputs
    model_device = device if device is not None else model.device
    inputs = process_inputs(tokenizer, pairs, prefix_tokens, suffix_tokens, max_length, model_device)
    
    # Get logits
    with torch.no_grad():
        logits = model(**inputs).logits.squeeze()
    
    # Apply sigmoid to get scores
    scores = logits.sigmoid()
    
    # Log results
    if int(os.environ.get("RANK", "0")) == 0:
        logger.info("=" * 40)
        logger.info("Reranker Results:")
        for i, (query, doc) in enumerate(zip(queries, documents)):
            logger.info(f"Pair {i}:")
            logger.info(f"Query: {query}")
            logger.info(f"Document: {doc}")
            if logits.dim() == 0:
                # Single pair case
                logger.info(f"Logit: {logits.item()}")
                logger.info(f"Score: {scores.item()}")
                break
            else:
                logger.info(f"Logit: {logits[i].item()}")
                logger.info(f"Score: {scores[i].item()}")
        logger.info("=" * 40)
    
    return logits, scores


# ---------- main ----------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="YAML config path")
    args = parser.parse_args()

    cfg = load_config(args.config)

    # logging
    output_path_from_cfg = cfg.get("output_path", None)
    final_log_path = resolve_output_path(output_path_from_cfg)
    logger = setup_logger(final_log_path)
    logger.info(f"Loaded config from {args.config}")
    logger.info(f"Logs will be saved to {final_log_path}")

    # env
    maybe_set_env(cfg.get("env", {}), logger)

    # ranks
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    # seed
    seed = cfg.get("seed", 1234)
    set_global_seed(seed)
    logger.info(f"Global seed set to {seed}")

    # mode: "generate" (default) or "reranker"
    mode = cfg.get("mode", "generate")
    logger.info(f"Running in mode: {mode}")

    # model
    model, tokenizer = load_hf_model_and_tokenizer(cfg["model"], logger, mode=mode)

    # device
    device = cfg["model"].get("device", None)


    # run model based on mode
    if mode == "reranker":
        # reranker mode: process query-document pairs
        reranker_cfg = cfg.get("reranker", {})
        queries = reranker_cfg.get("queries", None)
        documents = reranker_cfg.get("documents", None)
        if queries is None or documents is None:
            raise ValueError("reranker.queries and reranker.documents must be provided in YAML")
        if len(queries) != len(documents):
            raise ValueError("reranker.queries and reranker.documents must have the same length")
        
        instruction = reranker_cfg.get("instruction", None)
        prefix = reranker_cfg.get("prefix", '<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n')
        suffix = reranker_cfg.get("suffix", "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")
        max_length = reranker_cfg.get("max_length", 8192)
        
        _ = run_reranker(
            model,
            tokenizer,
            queries,
            documents,
            instruction,
            prefix,
            suffix,
            max_length,
            logger,
            device=device,
        )
    else:
        # generate mode: generate text
        prompt_cfg = cfg.get("prompt", {})
        prompt_texts = prompt_cfg.get("texts", None)
        if prompt_texts is None:
            raise ValueError("prompt.texts must be provided in YAML")
        generation_cfg = cfg.get("generation", {})
        max_new_tokens = generation_cfg.get("max_new_tokens", 128)
        
        _ = generate_text(
            model,
            tokenizer,
            prompt_texts,
            max_new_tokens=max_new_tokens,
            logger=logger,
            device=device,
            generation_cfg=generation_cfg,
        )

    logger.info("### HF run completed successfully.")
    
if __name__ == "__main__":
    main()
