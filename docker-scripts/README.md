# Docker

`Dockerfile` builds the image used by the baselines (`baseline/`) and for reproducing bug cases
with both engines side by side:

| Path in container | Contents |
| --- | --- |
| `/workspace/frameworks/vllm`, `/workspace/venvs/vllm` | vLLM fork with activation hooks (editable install) |
| `/workspace/frameworks/sglang`, `/workspace/venvs/sglang` | SGLang fork with activation hooks (editable install) |
| `/workspace/drivers/` | `run_hf_driver.sh`, `run_vllm_driver.sh`, `run_sglang_driver.sh` and their drivers |
| `/workspace/bug-cases/` | per-case configs, issue descriptions, and baseline prompts (from `baseline/for_docker/`) |
| `/workspace/frameworks/mini-swe-agent`, `/root/.opencode` | baseline agents |

## Build

```bash
docker-scripts/docker-build.sh
# or, from the repository root:
docker build -t ekka-baseline:latest -f docker-scripts/Dockerfile .
```

Override the engine sources with `VLLM_REPO`, `VLLM_BRANCH`, `SGLANG_REPO`, `SGLANG_BRANCH`.
SSH URLs (`git@github.com:...`) are supported; `docker-build.sh` forwards your ssh-agent.

## Run

```bash
HF_TOKEN=<token for gated models> docker-scripts/docker-run.sh
# inside the container
/workspace/drivers/run_vllm_driver.sh --config /workspace/bug-cases/vllm/vllm-26812/vllm_config.yaml
/workspace/drivers/run_hf_driver.sh   --config /workspace/bug-cases/vllm/vllm-26812/hf_config.yaml --tp 1
```

The Hugging Face token is read at runtime from `HF_TOKEN` and is never stored in the image.
