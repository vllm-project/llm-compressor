#!/usr/bin/env bash
set -euo pipefail

TEST_TYPE="${1:?Usage: run-tests.sh <base|transformers>}"

echo "~~~ System info"
cat /etc/issue

echo "--- Installing system packages"
git fetch --tags --unshallow 2>/dev/null || git fetch --tags
apt-get update -qq && apt-get install -y -qq curl ffmpeg g++ gcc make python3-dev
curl -LsSf https://astral.sh/uv/install.sh | sh

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
export UV_TORCH_BACKEND=auto
export HF_HOME=/model-cache

# install RELEASE dependencies for oder releases
export BUILD_TYPE=release
uv pip install .[dev]

echo "--- Installed packages"
uv pip list

TEST_EXIT_CODE=0
if [ "${TEST_TYPE}" = "base" ]; then
  echo "+++ Running pytorch tests"
  pytest -v tests/llmcompressor/pytorch || TEST_EXIT_CODE=$?
elif [ "${TEST_TYPE}" = "transformers" ]; then
  echo "+++ Running transformers tests"
  for test_file in $(find tests/llmcompressor/transformers -name "test_*.py" | sort); do
    echo "=== Running: ${test_file} ==="
    pytest -v "${test_file}" || TEST_EXIT_CODE=$?
  done
else
  echo "Unknown test type: ${TEST_TYPE}"
  echo "Usage: run-tests.sh <base|transformers>"
  exit 1
fi

exit ${TEST_EXIT_CODE}
