#!/bin/bash
# Start an interactive shell in the Ekka baseline container.
#
# Optional environment variables:
#   IMAGE_NAME, TAG       image to run (default ekka-baseline:latest)
#   CONTAINER_NAME        container name (default ekka-baseline)
#   EKKA_DOCKER_MOUNT     host directory mounted at /docker_mount (default ./docker_mount)
#   HF_TOKEN              passed through for gated Hugging Face models

set -e

IMAGE_NAME="${IMAGE_NAME:-ekka-baseline}"
TAG="${TAG:-latest}"
CONTAINER_NAME="${CONTAINER_NAME:-ekka-baseline}"
MOUNT_HOST="${EKKA_DOCKER_MOUNT:-$(pwd)/docker_mount}"
mkdir -p "${MOUNT_HOST}"

if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo "Container ${CONTAINER_NAME} already exists. Starting it..."
    docker start -i "${CONTAINER_NAME}"
else
    echo "Creating and starting new container: ${CONTAINER_NAME}"
    docker run \
        --gpus all \
        --network host \
        --shm-size=2g \
        --name "${CONTAINER_NAME}" \
        -e HF_TOKEN \
        -v "${MOUNT_HOST}:/docker_mount" \
        -it \
        "${IMAGE_NAME}:${TAG}"
fi
