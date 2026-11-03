# Baselines

General-purpose coding agents ([mini-swe-agent](https://github.com/SWE-agent/mini-swe-agent) and
[OpenCode](https://opencode.ai)) given the same bug report and a container with both engines installed,
used as a comparison point for Ekka.

## Layout

- `for_docker/` – copied into the baseline image at `/workspace`
  - `bug-cases/<engine>/<case>/` – issue description, driver configs, model tree, and the prompts
    (`prompt_top{1,3}.yaml` → `prompt_top{1,3}.md`)
  - `drivers/` – reproduction drivers and wrappers (`run_{hf,vllm,sglang}_driver.sh`)
  - `mini-swe-agent-config/` – mini-swe-agent config; `mini-swe-agent.env` holds placeholder values only,
    since the real model key is held by the LiteLLM proxy
- `build_prompt.py` – fills the prompt template from a `prompt_top*.yaml`
  (`scripts/generate_baseline_prompt/generate_baseline_prompt.sh` rebuilds all of them)
- `litellm_config.yaml`, `opencode.json` – model routing through a local LiteLLM proxy
- `run_mini_swe_agent.sh`, `run_opencode_agent.sh` – run every case/prompt in the container

## Running

1. Build the image (see `docker-scripts/README.md`):
   ```bash
   docker-scripts/docker-build.sh
   ```
2. Start the LiteLLM proxy on the host with Bedrock or Anthropic credentials in the environment:
   ```bash
   pip install 'litellm[proxy]'
   litellm --config baseline/litellm_config.yaml --port 4000
   ```
3. Run a baseline. Outputs are written to `$EKKA_DOCKER_MOUNT` (default `./docker_mount`), under `results/<case>/<timestamp>/`.
   ```bash
   # all cases, top-1 and top-3 prompts, 5 trials each
   baseline/run_mini_swe_agent.sh
   # a subset
   CASES="sglang-7936 vllm-26812" TOP_K=3 TRIALS=1 baseline/run_opencode_agent.sh
   ```
   To use the Anthropic API instead of Bedrock, set `model_name: "anthropic-claude-sonnet-4.5"` in
   `for_docker/mini-swe-agent-config/mini.yaml` (and rebuild the image), or
   `OPENCODE_MODEL=litellm/anthropic-claude-sonnet-4.5` for OpenCode.

Gated Hugging Face models need a token at runtime: `docker run -e HF_TOKEN=... ...`.
