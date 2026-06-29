#!/usr/bin/env bash
set -euo pipefail

TEST_TYPE="${1:?Usage: run-tests.sh <base|transformers>}"

cat /etc/issue

git fetch --tags --unshallow 2>/dev/null || git fetch --tags

apt-get update && apt-get install -y curl g++ gcc make python3-dev

curl -LsSf https://astral.sh/uv/install.sh | sh

export LD_LIBRARY_PATH=/usr/local/nvidia/lib64
export PATH="$HOME/.local/bin:/usr/local/nvidia/bin:$PATH"
nvidia-smi

uv venv testvenv --python "${PYTHON_VERSION}"
source testvenv/bin/activate

export CADENCE=commit
export UV_TORCH_BACKEND=cu130
export HF_HOME=/model-cache
uv pip install .[dev]

CODE_COVERAGE=$(buildkite-agent meta-data get "code-coverage" --default "false" 2>/dev/null || echo "false")
if [ "${CODE_COVERAGE}" = "true" ]; then
  export COVERAGE_FILE=".coverage.${TEST_TYPE}.${PYTHON_VERSION}"
  uv pip install coverage pytest-cov https://github.com/neuralmagic/pytest-nm-releng/archive/v0.4.0.tar.gz
  FLAGS_FILE="coverage_flags.sh"
  nmre-generate-coverage-flags --package "llmcompressor" --output-file "${FLAGS_FILE}"
  source "${FLAGS_FILE}"
  rm "${FLAGS_FILE}"
  export PYTEST_ADDOPTS
fi

git clone https://github.com/vllm-project/compressed-tensors.git
uv pip uninstall compressed-tensors
export GIT_CEILING_DIRECTORIES="$(pwd)"
cd compressed-tensors
BUILD_TYPE=nightly uv pip install .
cd ..
rm -rf compressed-tensors/

uv pip list

TEST_EXIT_CODE=0
if [ "${TEST_TYPE}" = "base" ]; then
  make test || TEST_EXIT_CODE=$?
elif [ "${TEST_TYPE}" = "transformers" ]; then
  COV_APPEND=""
  for test_file in $(find tests/llmcompressor/transformers -name "test_*.py" | sort); do
    echo "--- Running: ${test_file} ---"
    pytest -vra ${COV_APPEND} "${test_file}" || TEST_EXIT_CODE=$?
    if [ "${CODE_COVERAGE}" = "true" ]; then
      COV_APPEND="--cov-append"
    fi
  done
else
  echo "Unknown test type: ${TEST_TYPE}"
  echo "Usage: run-tests.sh <base|transformers>"
  exit 1
fi

if [ "${CODE_COVERAGE}" = "true" ]; then
  coverage report --data-file="${COVERAGE_FILE}" --skip-empty --format=markdown || true
  buildkite-agent artifact upload "${COVERAGE_FILE};coverage-html/**;coverage.json" || true
fi

exit ${TEST_EXIT_CODE}
