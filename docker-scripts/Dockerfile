# Baseline container for Ekka: vLLM and SGLang (activation-instrumented forks) built from source in
# separate virtual environments, plus the reproduction drivers and the baseline agents
# (mini-swe-agent, OpenCode). Build from the repository root:
#   docker build -t ekka-baseline:latest -f docker-scripts/Dockerfile .

FROM nvidia/cuda:12.9.0-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV WORKSPACE=/workspace

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    wget \
    build-essential \
    python3.10 \
    python3.10-dev \
    python3-pip \
    python3.10-venv \
    libssl-dev \
    libffi-dev \
    cmake \
    ninja-build \
    openssh-client \
    && rm -rf /var/lib/apt/lists/*

# Trust GitHub host key (prevents interactive prompt / verification failure)
RUN mkdir -p -m 0700 /root/.ssh && \
    ssh-keyscan github.com >> /root/.ssh/known_hosts

# Create symlink for python
RUN ln -sf /usr/bin/python3.10 /usr/bin/python && \
    ln -sf /usr/bin/python3.10 /usr/bin/python3

# Upgrade pip
RUN python3 -m pip install --upgrade pip setuptools wheel

# Create workspace directory
WORKDIR ${WORKSPACE}

# Engine forks with activation-collection hooks, and the mini-swe-agent release.
# SSH URLs (git@github.com:...) also work; docker-build.sh then forwards your ssh-agent.
ARG VLLM_REPO=https://github.com/IKACE/vLLM.git
ARG SGLANG_REPO=https://github.com/IKACE/SGLang.git
ARG VLLM_BRANCH=v0.11.0-baseline-docker
ARG SGLANG_BRANCH=v0.5.6-baseline-docker
ARG MINI_SWE_AGENT_REPO=https://github.com/SWE-agent/mini-swe-agent.git
ARG MINI_SWE_AGENT_BRANCH=v1.17.4

# log arguments
RUN echo "VLLM_REPO: ${VLLM_REPO}"
RUN echo "VLLM_BRANCH: ${VLLM_BRANCH}"
RUN echo "SGLANG_REPO: ${SGLANG_REPO}"
RUN echo "SGLANG_BRANCH: ${SGLANG_BRANCH}"

# Create directories for frameworks and virtual environments
RUN mkdir -p ${WORKSPACE}/frameworks && \
    mkdir -p ${WORKSPACE}/venvs

# Set up separate virtual environments for vLLM and SGLang
RUN python3 -m venv ${WORKSPACE}/venvs/vllm && \
    ${WORKSPACE}/venvs/vllm/bin/pip install --upgrade pip setuptools wheel && \
    python3 -m venv ${WORKSPACE}/venvs/sglang && \
    ${WORKSPACE}/venvs/sglang/bin/pip install --upgrade pip setuptools wheel

# Clone and install vLLM in its own virtual environment
RUN --mount=type=ssh \
    cd ${WORKSPACE}/frameworks && \
    if [ -n "${VLLM_BRANCH}" ]; then \
        git clone -b "${VLLM_BRANCH}" "${VLLM_REPO}" vllm; \
    else \
        git clone "${VLLM_REPO}" vllm; \
    fi && \
    cd vllm && \
    VLLM_USE_PRECOMPILED=1 ${WORKSPACE}/venvs/vllm/bin/pip install -e . && \
    ${WORKSPACE}/venvs/vllm/bin/pip install accelerate


# Clone and install SGLang in its own virtual environment
RUN --mount=type=ssh \
    cd ${WORKSPACE}/frameworks && \
    if [ -n "${SGLANG_BRANCH}" ]; then \
        git clone -b "${SGLANG_BRANCH}" "${SGLANG_REPO}" sglang; \
    else \
        git clone "${SGLANG_REPO}" sglang; \
    fi && \
    cd sglang && \
    ${WORKSPACE}/venvs/sglang/bin/pip install --upgrade pip && \
    ${WORKSPACE}/venvs/sglang/bin/pip install -e "python" && \
    ${WORKSPACE}/venvs/sglang/bin/pip install accelerate


# Hugging Face hub client. Gated models need a token at runtime:
#   docker run -e HF_TOKEN=<your token> ...
# (The token is deliberately not baked into the image.)
RUN ${WORKSPACE}/venvs/vllm/bin/pip install huggingface-hub && \
    ${WORKSPACE}/venvs/sglang/bin/pip install huggingface-hub

# install some libraries
RUN apt-get update && apt-get install -y numactl

# Install mini-swe-agent
RUN --mount=type=ssh \
    cd ${WORKSPACE}/frameworks && \
    if [ -n "${MINI_SWE_AGENT_BRANCH}" ]; then \
        git clone -b "${MINI_SWE_AGENT_BRANCH}" "${MINI_SWE_AGENT_REPO}" mini-swe-agent; \
    else \
        git clone "${MINI_SWE_AGENT_REPO}" mini-swe-agent; \
    fi && \
    cd mini-swe-agent && \
    pip install -e .


# Create wrapper scripts directory and copy scripts first
RUN mkdir -p ${WORKSPACE}/scripts

# recursively copy all files in baseline/for_docker to the workspace
COPY baseline/for_docker/ ${WORKSPACE}/
RUN chmod +x ${WORKSPACE}/drivers/*.sh

# Remove git history for vllm and sglang
RUN cd ${WORKSPACE}/frameworks/vllm && \
    rm -rf .git && \
    cd ${WORKSPACE}/frameworks/sglang && \
    rm -rf .git

# Setup env for mini-swe-agent
RUN mkdir -p /root/.config/mini-swe-agent

COPY baseline/for_docker/mini-swe-agent-config/mini-swe-agent.env /root/.config/mini-swe-agent/.env

# setup opencode
RUN curl -fsSL https://opencode.ai/install | bash
COPY baseline/opencode.json ${WORKSPACE}/opencode.json

# Default command
CMD ["/bin/bash"]

