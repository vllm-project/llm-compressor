#!/bin/bash
set -exo pipefail

echo "Trigger test scripts..."
echo "NUMA_NODE is $NUMA_NODE"
echo "NUMA_CPUSET is $NUMA_CPUSET"
echo "ZE_AFFINITY_MASK is $ZE_AFFINITY_MASK"
echo "NODE_LABEL is $NODE_LABEL"

CONTAINER_NAME=vllm-test-container-$NODE_LABEL
REPO_PATH="${BUILDKITE_BUILD_CHECKOUT_PATH:-$PWD}"

# trap 'docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true' EXIT
if docker ps -a --format '{{.Names}}' | grep -Fxq "$CONTAINER_NAME"; then
    docker rm -f "$CONTAINER_NAME"
fi

docker run -tid --disable-content-trust --privileged --shm-size="2g" --name "$CONTAINER_NAME" -v "$REPO_PATH:/workspace" -w /workspace "$IMAGE_NAME"
echo "Show the container list after docker run ... "
docker ps -a | grep "$CONTAINER_NAME"

echo "--- Setup test environment..."
docker exec "$CONTAINER_NAME" \
              bash -c "uv pip install --upgrade pip setuptools && \
              uv pip install .[dev] --extra-index-url https://download.pytorch.org/whl/xpu --index-strategy unsafe-best-match && \
              echo 'Installing compressed-tensors (nightly)' && \
              git clone --quiet https://github.com/vllm-project/compressed-tensors.git && \
              uv pip uninstall compressed-tensors && \
              export GIT_CEILING_DIRECTORIES=/workspace && \
              cd compressed-tensors && \
              BUILD_TYPE=nightly uv pip install . --extra-index-url https://download.pytorch.org/whl/xpu --index-strategy unsafe-best-match && \
              uv pip list"

echo "--- Run tests inside the container..."
docker exec -e NUMA_NODE=${NUMA_NODE} -e NUMA_CPUSET=${NUMA_CPUSET} -e ZE_AFFINITY_MASK=${ZE_AFFINITY_MASK} "$CONTAINER_NAME" \
              bash -c "numactl --physcpubind=${NUMA_CPUSET:-84-111} --membind=${NUMA_NODE:-3} make test-xpu"
