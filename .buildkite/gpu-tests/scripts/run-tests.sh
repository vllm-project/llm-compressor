#!/usr/bin/env bash
set -euo pipefail

TEST_TYPE="${1:?Usage: run-tests.sh <base|transformers>}"

echo "~~~ System info"
cat /etc/issue

echo "--- Installing system packages"
git fetch --tags --unshallow 2>/dev/null || git fetch --tags
apt-get update -qq && apt-get install -y -qq curl ffmpeg g++ gcc make python3-dev
curl -LsSf https://github.com/astral-sh/uv/releases/download/0.8.15/uv-installer.sh | sh

export LD_LIBRARY_PATH=/usr/local/nvidia/lib64
export PATH="$HOME/.local/bin:/usr/local/nvidia/bin:$PATH"

echo "~~~ GPU info"
nvidia-smi

echo "--- Setting up Python environment"
export UV_NO_PROGRESS=1
export UV_CACHE_DIR="$PWD/.uv-cache"
uv venv testvenv --python "${PYTHON_VERSION}"
source testvenv/bin/activate

export CADENCE=commit
export HF_HOME=/model-cache

# install RELEASE dependencies for oder releases;
# add torch and torchvision etc specific install here
# to make sure uv picks up the compatible versions due
# to uv bug: https://github.com/astral-sh/uv/issues/22965
export BUILD_TYPE=release
export UV_TORCH_BACKEND=cu129
uv pip install ".[dev,autoround]" "torch==2.8.0" "torchvision==0.23.0" "torchcodec==0.7.0"

echo "--- Installed packages"
uv pip list

TEST_EXIT_CODE=0
if [ "${TEST_TYPE}" = "base" ]; then
  echo "+++ Running base tests"
  make test || TEST_EXIT_CODE=$?
elif [ "${TEST_TYPE}" = "pytorch" ]; then
  echo "+++ Running pytorch tests"
  pytest -v tests/llmcompressor/pytorch || TEST_EXIT_CODE=$?
elif [ "${TEST_TYPE}" = "transformers" ]; then
  echo "+++ Running transformers tests"
  for eachfolder in compression finetune gptq oneshot sparsification obcq tracing kv_cache
  do
    echo "=== Running: tests/llmcompressor/transformers/${eachfolder} ==="
    pytest -vra "tests/llmcompressor/transformers/${eachfolder}" || TEST_EXIT_CODE=$?
  done
else
  echo "Unknown test type: ${TEST_TYPE}"
  echo "Usage: run-tests.sh <base|transformers>"
  exit 1
fi

exit ${TEST_EXIT_CODE}
