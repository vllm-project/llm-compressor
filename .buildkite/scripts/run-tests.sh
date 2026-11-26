#!/usr/bin/env bash
set -euo pipefail

TEST_TYPE="${1:?Usage: run-tests.sh <unit-integration|e2e|lmeval>}"

echo "~~~ System info"
cat /etc/issue
df -h

echo "--- Installing system packages"
git fetch --tags --unshallow 2>/dev/null || git fetch --tags
apt-get update -qq > /dev/null 2>&1 && apt-get install -y -qq curl g++ gcc make python3-dev ffmpeg
curl -LsSf https://astral.sh/uv/install.sh | sh

export LD_LIBRARY_PATH=/usr/local/nvidia/lib64
export PATH="$HOME/.local/bin:/usr/local/nvidia/bin:$PATH"
export TQDM_DISABLE=1
export HF_HUB_DISABLE_PROGRESS_BARS=1

echo "~~~ GPU info"
nvidia-smi

echo "--- Setting up Python environment"
export UV_NO_PROGRESS=1
export UV_CACHE_DIR="$PWD/.uv-cache"
export UV_TORCH_BACKEND=cu130
export HF_HOME=/model-cache

uv venv testvenv --python "${PYTHON_VERSION}"
source testvenv/bin/activate

uv pip install ".[dev]"

# Handle dependency bounds (lower/upper)
if [ -n "${DEP_BOUND:-}" ] && [ "${DEP_BOUND}" != "skip" ]; then
  echo "--- Applying ${DEP_BOUND} dependency bounds"
  required=$(sed -n '/install_requires\s*=\s*\[/,/]/p' setup.py \
    | grep -e '"*>=.*<=*' -e '"*==[0-9].*"' \
    | sed 's/[[:space:]]*//' \
    | sed 's/^(//; s/ if .*//' \
    | sed 's/else\ //g' \
    | grep -v 'compressed-tensors')

  if [ "${DEP_BOUND}" = "lower" ]; then
    final=$(echo "$required" | grep -oP '"[^"]*(>=|==)[^",]+' | tr -d '"' | sed 's/>=/==/g')
  elif [ "${DEP_BOUND}" = "upper" ]; then
    final=$(echo "$required" | tr -d '"' | sed 's/>=.*,<=/==/g')
  fi

  non_torch=$(echo "$final" | grep -v torch || true)
  if [ -n "$non_torch" ]; then
    echo "$non_torch" > requirements-bounds.txt
    echo "Pinned dependencies:"
    cat requirements-bounds.txt
    uv pip install -r requirements-bounds.txt
  fi

  torch_bound=$(echo "$final" | grep torch || true)
  if [ -n "$torch_bound" ]; then
    echo "$torch_bound" > requirements-torch.txt
    echo "Pinned torch:"
    cat requirements-torch.txt
    uv pip install -r requirements-torch.txt
  fi
fi

# Handle transformers version override
if [ -n "${TRANSFORMERS_VERSION:-}" ] && [ "${TRANSFORMERS_VERSION}" != "skip" ]; then
  echo "--- Installing transformers (${TRANSFORMERS_VERSION})"
  if [ "${TRANSFORMERS_VERSION}" = "latest" ]; then
    uv pip install --reinstall transformers
  elif [ "${TRANSFORMERS_VERSION}" = "main" ]; then
    uv pip install --reinstall "transformers @ git+https://github.com/huggingface/transformers.git@main"
  else
    uv pip install --reinstall "transformers${TRANSFORMERS_VERSION}"
  fi
fi

if [ "${CT_GITREF:-skip}" != "skip" ]; then
  echo "--- Installing compressed-tensors from source (${CT_GITREF})"
  git clone --quiet https://github.com/vllm-project/compressed-tensors.git
  uv pip uninstall compressed-tensors
  export GIT_CEILING_DIRECTORIES="$(pwd)"
  cd compressed-tensors
  git checkout "${CT_GITREF}"
  BUILD_TYPE=nightly uv pip install .
  cd ..
  rm -rf compressed-tensors/
fi

if [[ "${TEST_TYPE}" == "e2e" ]] || [[ "${TEST_TYPE}" == "lmeval" ]]; then
  echo "--- Setting up vLLM environment"
  uv venv vllm_venv --python "${PYTHON_VERSION}"
  VLLM_VENV_PYTHON="$PWD/vllm_venv/bin/python"

  if [[ "${PYTHON_VERSION}" =~ 3.1[01] ]]; then
    uv pip install --python "${VLLM_VENV_PYTHON}" setuptools
  fi

  if [ "${VLLM_VERSION:-}" = "latest" ]; then
    UV_TORCH_BACKEND=cu130 uv pip install --python "${VLLM_VENV_PYTHON}" vllm
  else
    UV_TORCH_BACKEND=cu130 uv pip install --python "${VLLM_VENV_PYTHON}" vllm \
      --extra-index-url https://wheels.vllm.ai/nightly/cu130
  fi

  if [[ "${TEST_TYPE}" == "lmeval" ]]; then
    echo "--- Installing lm-eval dependencies"
    uv pip install --python "${VLLM_VENV_PYTHON}" "lm-eval[api]>=0.4.11" "ray[cgraph,default]>=2.48.0"
  fi

  export VLLM_PYTHON="${VLLM_VENV_PYTHON}"

  echo "~~~ Main environment"
  uv pip freeze
  echo "~~~ vLLM environment"
  uv pip freeze --python "${VLLM_VENV_PYTHON}"
fi

echo "--- Installed packages"
uv pip list

TEST_EXIT_CODE=0

if [ "${TEST_TYPE}" = "unit-integration" ]; then
  echo "+++ Running non-transformers tests"
  pytest -ra tests \
    --ignore tests/llmcompressor/transformers \
    --ignore tests/lmeval \
    --ignore tests/tools \
    --ignore tests/examples \
    --ignore tests/e2e || TEST_EXIT_CODE=$?

  echo "+++ Running transformers tests"
  for test_file in $(find tests/llmcompressor/transformers -name "test_*.py" | sort); do
    echo "=== Running: ${test_file} ==="
    pytest -vra "${test_file}" || TEST_EXIT_CODE=$?
  done

elif [ "${TEST_TYPE}" = "e2e" ]; then
  echo "+++ Running E2E tests (${E2E_CONFIGS})"
  VLLM_BIN_DIR="$(dirname "${VLLM_PYTHON}")"
  export PATH="$PATH:${VLLM_BIN_DIR}"
  bash tests/e2e/run_tests_in_python.sh \
    -c "tests/e2e/${E2E_CONFIGS}" \
    -t tests/e2e/test_vllm.py || TEST_EXIT_CODE=$?

elif [ "${TEST_TYPE}" = "lmeval" ]; then
  echo "+++ Running LM Eval tests (${LMEVAL_CONFIGS})"
  VLLM_BIN_DIR="$(dirname "${VLLM_PYTHON}")"
  export PATH="$PATH:${VLLM_BIN_DIR}"
  bash tests/e2e/run_tests_in_python.sh \
    -c "tests/lmeval/${LMEVAL_CONFIGS}" \
    -t tests/lmeval/test_lmeval.py || TEST_EXIT_CODE=$?

else
  echo "Unknown test type: ${TEST_TYPE}"
  echo "Usage: run-tests.sh <unit-integration|e2e|lmeval>"
  exit 1
fi

exit ${TEST_EXIT_CODE}
