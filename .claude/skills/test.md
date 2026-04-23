# Running Tests

Nearly all tests in this project require a GPU.

When running tests:

1. First check if `canhazgpu` is available: `which canhazgpu`
2. If available, MUST run tests using `canhazgpu` with appropriate GPU allocation
3. Use the format: `canhazgpu --gpus 1 -- python3 -m pytest tests/...`

Example:
```bash
# Check if canhazgpu is available
which canhazgpu

# Run tests with canhazgpu (required if available)
canhazgpu --gpus 1 -- python3 -m pytest tests/test_example.py
```

Note: Adjust the `--gpus` count based on test requirements (typically 1 GPU is sufficient for most tests).
