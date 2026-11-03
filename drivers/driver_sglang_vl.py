#!/usr/bin/env python3
"""
Run SGLang from a YAML config with timestamped logging.
Supports one mode:
  - generate (default): standard text / token generation via sglang.Engine

Usage:
    python driver_sglang_v2.py --config config.yaml

Config schema (mirrors vLLM driver where possible):

top-level:
  output_path: (optional) str, log file or directory
  env: (optional) dict of env vars
  seed: (optional) int, random seed (default 42)
  mode: (optional) "generate" (anything else will error)

  model:
    name: or model_path: (HF name or local path)
    ... other Engine kwargs (tp, max_batch_size, impl, trust_remote_code, etc.)

  sampling:
    temperature: float (default 0.0)
    top_p: float (default 1.0)
    max_tokens: int -> mapped to max_new_tokens
    max_new_tokens: int
    stop: list[str] or other SGLang-compatible sampling fields

  prompt:
    mode: "text" or "tokens"
    texts: [str, ...]         # when mode == "text"
    token_ids: [[int, ...]]   # when mode == "tokens"
"""

import os
import sys
import argparse
import json
import yaml
import random
import numpy as np
import torch
import logging
from datetime import datetime

# Make the repository root importable (common/, activation_collector/, scripts/).
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
from common.config import load_config  # expands ${EKKA_ROOT} / ${EKKA_ACTIVATION_DIR} in YAML values
from typing import Any, Dict, List

import sglang as sgl


# -----------------------------
# Utilities
# -----------------------------


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False




def resolve_output_path(output_path_from_cfg: str | None) -> str:
    """Return a concrete log file path, always timestamped when directory-like."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    if output_path_from_cfg is None:
        # default: logs/sglang_run_<ts>.log
        base_dir = "logs"
        os.makedirs(base_dir, exist_ok=True)
        return os.path.join(base_dir, f"sglang_run_{ts}.log")

    # user gave something
    # 1) explicit file with extension -> use as-is
    # 2) directory-like -> drop timestamp file inside
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
        return os.path.join(output_path_from_cfg, f"sglang_run_{ts}.log")


def setup_logger(final_output_path: str) -> logging.Logger:
    """Set up an isolated logger that is not affected by root logger changes."""
    logger_name = "sglang_driver"   # any unique name is fine
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    logger.propagate = False  # <-- critical: don't send logs to root

    # Clear existing handlers if we re-call setup_logger
    if logger.handlers:
        logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    # File handler
    fh = logging.FileHandler(final_output_path, mode="w")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    # Console handler
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    return logger


def maybe_set_env(env_cfg: Dict[str, Any] | None, logger: logging.Logger):
    if not env_cfg:
        return
    for k, v in env_cfg.items():
        os.environ[str(k)] = str(v)
        logger.info(f"Set environment variable: {k}={v}")


# -----------------------------
# Engine construction
# -----------------------------


def build_engine(cfg: dict, logger: logging.Logger) -> sgl.Engine:
    """
    Build a SGLang offline Engine from cfg["model"].

    Supports both:
      model:
        model_path: meta-llama/...
      model:
        name: meta-llama/...
    """
    if "model" not in cfg:
        raise ValueError("Config must contain a 'model' block.")

    model_cfg = cfg["model"]
    model_path = model_cfg.get("model_path") or model_cfg.get("name")
    if model_path is None:
        raise ValueError("model.model_path or model.name must be provided in YAML")

    engine_kwargs: Dict[str, Any] = {}
    for key, val in model_cfg.items():
        if key in ("model_path", "name"):
            continue
        engine_kwargs[key] = val

    # You can set random_seed here as well for SGLang engine itself
    seed = cfg.get("seed", 42)
    engine_kwargs.setdefault("random_seed", seed)

    logger.info(
        f"Initializing SGLang Engine with model_path={model_path}, kwargs={engine_kwargs}"
    )
    llm = sgl.Engine(model_path=model_path, **engine_kwargs)
    return llm


def build_runtime(cfg: dict, logger: logging.Logger) -> "sgl.Runtime":
    """Build an SGLang Runtime (backend) for multimodal / chat-style prompts.

    This is required for image/video primitives (e.g., sgl.image()).
    """
    model_cfg = cfg.get("model", {}) or {}
    model_path = model_cfg.get("model_path") or model_cfg.get("name")
    if not model_path:
        raise ValueError("model.model_path or model.name must be provided in YAML")

    rt_kwargs: Dict[str, Any] = {}
    for key, val in model_cfg.items():
        if key in ("model_path", "name"):
            continue
        rt_kwargs[key] = val

    # Prefer tp_size naming (Runtime uses tp_size)
    if "tp" in rt_kwargs and "tp_size" not in rt_kwargs:
        rt_kwargs["tp_size"] = rt_kwargs.pop("tp")

    seed = cfg.get("seed", 42)
    rt_kwargs.setdefault("random_seed", seed)

    logger.info(f"Initializing SGLang Runtime with model_path={model_path}, kwargs={rt_kwargs}")
    runtime = sgl.Runtime(model_path=model_path, **rt_kwargs)

    # Optional chat template override (useful for VLMs, e.g., qwen2-vl / qwen2.5-vl)
    chat_template = model_cfg.get("chat_template") or model_cfg.get("model_chat_template")
    if chat_template:
        try:
            runtime.endpoint.chat_template = sgl.lang.chat_template.get_chat_template(chat_template)
            logger.info(f"Set runtime chat template to: {chat_template}")
        except Exception as e:
            logger.warning(f"Failed to set chat template '{chat_template}': {e}")

    sgl.set_default_backend(runtime)
    return runtime


# -----------------------------
# Sampling / prompts
# -----------------------------


def build_sampling(cfg: dict) -> Dict[str, Any]:
    """
    Build sampling params dict for SGLang Engine.generate.

    We map vLLM-style 'max_tokens' -> SGLang 'max_new_tokens' if needed,
    and otherwise just forward the dict.
    """
    samp_cfg = dict(cfg.get("sampling", {}) or {})

    # Default values
    samp_cfg.setdefault("temperature", 0.0)
    samp_cfg.setdefault("top_p", 1.0)

    # Map vLLM-style name to SGLang name if user reused same config
    if "max_tokens" in samp_cfg and "max_new_tokens" not in samp_cfg:
        samp_cfg["max_new_tokens"] = samp_cfg.pop("max_tokens")

    # If neither is provided, add a default
    samp_cfg.setdefault("max_new_tokens", 512)

    return samp_cfg


def _ensure_list(x: Any) -> List[Any]:
    """Ensure x is a list; if not, wrap it."""
    if isinstance(x, list):
        return x
    return [x]


# -----------------------------
# Generation helpers
# -----------------------------



def run_generate_messages(runtime: "sgl.Runtime", cfg: dict, logger: logging.Logger) -> None:
    """Generate using chat-style messages, supporting image/video primitives via sgl.image()."""
    prompt_cfg = cfg.get("prompt", {}) or {}
    messages = prompt_cfg.get("messages")
    if not messages:
        raise ValueError("prompt.mode='messages' but no prompt.messages provided")

    gen_cfg = cfg.get("generation", {}) or {}

    # HF-style key + SGLang-style key
    max_tokens = int(gen_cfg.get("max_new_tokens", gen_cfg.get("max_tokens", 256)))

    # Build kwargs for sgl.gen; don't pass None values
    gen_kwargs = {"max_tokens": max_tokens}
    for k in ("temperature", "top_p", "top_k", "repetition_penalty", "min_p"):
        if k in gen_cfg and gen_cfg[k] is not None:
            gen_kwargs[k] = gen_cfg[k]

    @sgl.function
    def chat_program(s, messages):
        for msg in messages:
            role = (msg.get("role") or "").lower()
            content = msg.get("content")

            # Build payload as concatenation of strings + multimodal primitives
            if isinstance(content, str):
                payload = content
            elif isinstance(content, list):
                parts = []
                for item in content:
                    t = (item.get("type") or "").lower()
                    if t == "text":
                        parts.append(item.get("text", ""))
                    elif t == "image":
                        img_path = item.get("image") or item.get("image_url") or item.get("path")
                        if not img_path:
                            raise ValueError("Image content item missing 'image' path")
                        parts.append(sgl.image(str(img_path)))
                    elif t == "video":
                        vid_path = item.get("video") or item.get("video_url") or item.get("path")
                        if not vid_path:
                            raise ValueError("Video content item missing 'video' path")
                        parts.append(sgl.video(str(vid_path)))
                    else:
                        raise ValueError(f"Unsupported content type in messages: {t!r}")

                payload = parts[0] if parts else ""
                for p in parts[1:]:
                    payload = payload + p
            else:
                raise ValueError("message.content must be a string or a list")

            if role == "system":
                s += sgl.system(payload)
            elif role == "user":
                s += sgl.user(payload)
            elif role == "assistant":
                s += sgl.assistant(payload)
            else:
                raise ValueError(f"Unsupported role in messages: {role!r}")

        # IMPORTANT: use max_tokens (not max_new_tokens)
        s += sgl.assistant(sgl.gen("answer", **gen_kwargs))

    state = chat_program.run(messages=messages, backend=runtime)

    # Depending on SGLang version, answer may be in different places
    answer = None
    if hasattr(state, "__getitem__"):
        try:
            answer = state["answer"]
        except Exception:
            pass
    if answer is None and hasattr(state, "get"):
        answer = state.get("answer")

    logger.info(f"Answer: {answer}")
    print(answer)


def run_generate(llm: sgl.Engine, cfg: dict, logger: logging.Logger):
    sampling_params = build_sampling(cfg)

    prompt_cfg = cfg.get("prompt", {})
    prompt_mode = prompt_cfg.get("mode", "text")

    if prompt_mode == "text":
        prompts = prompt_cfg.get("texts", [])
        if not prompts:
            raise ValueError("prompt.mode='text' but no prompt.texts provided")
        prompts = _ensure_list(prompts)
        logger.info(f"Generating for {len(prompts)} text prompt(s).")

        outputs = llm.generate(prompts, sampling_params)

        # For a list of prompts, we expect a list of dicts
        outputs = _ensure_list(outputs)

        for i, (prompt, out) in enumerate(zip(prompts, outputs)):
            # SGLang offline engine usually returns dicts with "text" and "output_ids"
            text = None
            output_ids = None
            if isinstance(out, dict):
                text = out.get("text")
                output_ids = out.get("output_ids")
            else:
                # Fallback: try attributes if some wrapper object
                text = getattr(out, "text", None)
                output_ids = getattr(out, "output_ids", None)

            logger.info("=" * 40)
            logger.info(f"Prompt {i}:")
            logger.info(f"Input text:\n{prompt}")
            logger.info(f"Generated text:\n{text}")
            logger.info(f"Generated token ids: {output_ids}")

    elif prompt_mode == "tokens":
        token_lists = prompt_cfg.get("token_ids", [])
        if not token_lists:
            raise ValueError("prompt.mode='tokens' but no prompt.token_ids provided")
        token_lists = _ensure_list(token_lists)
        logger.info(f"Generating for {len(token_lists)} token prompt(s).")

        # SGLang Engine supports passing input_ids explicitly
        outputs = llm.generate(input_ids=token_lists, sampling_params=sampling_params)

        outputs = _ensure_list(outputs)

        for i, (input_ids, out) in enumerate(zip(token_lists, outputs)):
            if isinstance(out, dict):
                text = out.get("text")
                output_ids = out.get("output_ids")
            else:
                text = getattr(out, "text", None)
                output_ids = getattr(out, "output_ids", None)

            logger.info("=" * 40)
            logger.info(f"Prompt {i}:")
            logger.info(f"Input token ids:\n{input_ids}")
            logger.info(f"Generated text:\n{text}")
            logger.info(f"Generated token ids: {output_ids}")
    else:
        raise ValueError(f"Unknown prompt.mode={prompt_mode}")


# -----------------------------
# Main
# -----------------------------


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

    # activation collector (optional, kept for symmetry with vLLM driver;
    # you can hook your own code off this env var if desired)
    ac_cfg = cfg.get("activation_collector", None)
    if ac_cfg is not None and ac_cfg.get("enabled", False):
        activation_path = ac_cfg.get("save_path", None)
        activation_prefix = ac_cfg.get("save_prefix", None)
        assert (
            activation_path is not None
        ), "activation_collector.save_path must be provided in YAML"
        os.makedirs(activation_path, exist_ok=True)
        os.environ["SGLANG_ACTIVATION_SAVE_PATH"] = activation_path
        if activation_prefix is not None:
            os.environ["SGLANG_ACTIVATION_SAVE_PREFIX"] = activation_prefix
        logger.info(
            f"Activation collector enabled. "
            f"SGLANG_ACTIVATION_SAVE_PATH={activation_path}"
        )

    # seed
    seed = cfg.get("seed", 42)
    set_seed(seed)
    logger.info(f"Random seed set to {seed}")

    # mode (only 'generate' is supported here)
    driver_mode = cfg.get("mode", "generate")
    logger.info(f"Driver mode: {driver_mode}")
    if driver_mode != "generate":
        raise ValueError(
            f"Only mode='generate' is supported in this SGLang driver, got {driver_mode}"
        )

    # model / engine
    
    prompt_cfg = cfg.get("prompt", {}) or {}
    prompt_mode = prompt_cfg.get("mode", "text")

    if prompt_mode == "messages":
        runtime = build_runtime(cfg, logger)
        try:
            run_generate_messages(runtime, cfg, logger)
            logger.info("### SGLang run completed successfully.")
        finally:
            try:
                runtime.shutdown()
                logger.info("SGLang Runtime shut down.")
            except Exception as e:
                logger.warning(f"Failed to shut down SGLang Runtime cleanly: {e}")
    else:
        llm = build_engine(cfg, logger)
        try:
            run_generate(llm, cfg, logger)
            logger.info("### SGLang run completed successfully.")
        finally:
            try:
                llm.shutdown()
                logger.info("SGLang Engine shut down.")
            except Exception as e:
                logger.warning(f"Failed to shut down SGLang Engine cleanly: {e}")


if __name__ == "__main__":
    main()
