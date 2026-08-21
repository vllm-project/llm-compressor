#!/usr/bin/env bash
set -euo pipefail

echo "--- Installing system packages"
git fetch --tags --unshallow 2>/dev/null || git fetch --tags
apt-get update -qq && apt-get install -y -qq curl python3-dev
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

echo "--- Setting up Python environment"
export UV_NO_PROGRESS=1
uv venv covenv --python "3.12"
source covenv/bin/activate

export UV_TORCH_BACKEND=auto
uv pip install -U setuptools
uv pip install coverage setuptools-scm
python3 setup.py sdist bdist_wheel

echo "--- Downloading coverage artifacts"
buildkite-agent artifact download ".coverage.*" .
buildkite-agent artifact download "coverage-html/**" . || true
buildkite-agent artifact download "coverage.json" . || true

echo "+++ Combining coverage reports"
cat << 'EOF' > .coveragerc
[paths]
source =
    src/
    */site-packages/
EOF

coverage combine
coverage report --skip-empty --format=markdown
coverage html --directory coverage-html
coverage json -o coverage.json

echo "--- Uploading combined coverage"
buildkite-agent artifact upload ".coverage;coverage-html/**;coverage.json"
