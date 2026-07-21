#!/bin/bash

set -exo pipefail
DOCKERFILE=.buildkite/xpu-tests/docker/Dockerfile.xpu
CURRENT_HASH=$(md5sum "$DOCKERFILE" | awk '{print $1}')
NEED_BUILD=false
if ! docker image ls --format '{{.Repository}}:{{.Tag}}' | grep -Fxq "$IMAGE_NAME"; then
    NEED_BUILD=true
else
    EXISTING_HASH=$(docker inspect --format='{{index .Config.Labels "dockerfile.hash"}}' "$IMAGE_NAME" 2>/dev/null || echo "")
    if [[ "$CURRENT_HASH" != "$EXISTING_HASH" ]]; then
    echo "Dockerfile changed (old=$EXISTING_HASH, new=$CURRENT_HASH), rebuilding image..."
    NEED_BUILD=true
    fi
fi
if [[ "$NEED_BUILD" == "true" ]]; then
    docker build --build-arg USER_ID=$(id -u) --build-arg GROUP_ID=$(id -g) \
        --build-arg RENDER_GID=$(getent group render | cut -d: -f3 || echo 991) \
        --label "dockerfile.hash=$CURRENT_HASH" -t "$IMAGE_NAME" -f "$DOCKERFILE" .
fi
docker images "$IMAGE_NAME"