#!/bin/bash
# Build the Ekka baseline image (vLLM + SGLang forks, drivers, baseline agents).
#
# Usage (from anywhere):
#   docker-scripts/docker-build.sh
#
# Optional environment variables:
#   IMAGE_NAME, TAG                     image name and tag (default ekka-baseline:latest)
#   VLLM_REPO, VLLM_BRANCH              vLLM fork to build
#   SGLANG_REPO, SGLANG_BRANCH          SGLang fork to build
# SSH repository URLs (git@github.com:...) are supported; your ssh-agent is forwarded to the build.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
IMAGE_NAME="${IMAGE_NAME:-ekka-baseline}"
TAG="${TAG:-latest}"

VLLM_REPO="${VLLM_REPO:-https://github.com/IKACE/vLLM.git}"
SGLANG_REPO="${SGLANG_REPO:-https://github.com/IKACE/SGLang.git}"
VLLM_BRANCH="${VLLM_BRANCH:-v0.11.0-baseline-docker}"
SGLANG_BRANCH="${SGLANG_BRANCH:-v0.5.6-baseline-docker}"

echo "Building Docker image: ${IMAGE_NAME}:${TAG}"
echo "This may take a while as it builds vLLM and SGLang from source..."
echo "  vLLM:   ${VLLM_REPO}${VLLM_BRANCH:+ (branch: ${VLLM_BRANCH})}"
echo "  SGLang: ${SGLANG_REPO}${SGLANG_BRANCH:+ (branch: ${SGLANG_BRANCH})}"
echo ""

export DOCKER_BUILDKIT=1
BUILD_ARGS=(
    --build-arg "VLLM_REPO=${VLLM_REPO}"
    --build-arg "SGLANG_REPO=${SGLANG_REPO}"
    --build-arg "VLLM_BRANCH=${VLLM_BRANCH}"
    --build-arg "SGLANG_BRANCH=${SGLANG_BRANCH}"
)

if [[ "${VLLM_REPO}" == git@* ]] || [[ "${SGLANG_REPO}" == git@* ]]; then
    echo "Detected SSH URLs: forwarding ssh-agent to the build (load your key with ssh-add first)."
    BUILD_ARGS+=(--ssh default)
fi

docker build \
    -t "${IMAGE_NAME}:${TAG}" \
    -f "${SCRIPT_DIR}/Dockerfile" \
    "${BUILD_ARGS[@]}" \
    "${PROJECT_ROOT}"

echo ""
echo "Build complete. Start a shell in the container with:"
echo "  docker-scripts/docker-run.sh"
