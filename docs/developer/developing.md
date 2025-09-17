---
title: Development Guide
weight: -6
---

# Developing LLM Compressor

LLM Compressor is developed and tested using Python 3.8-3.11.
To develop LLM Compressor, you will also need the development dependencies and to follow the styling guidelines.

Here are some details to get started.

## Basic Commands

**Development Installation**

```bash
git clone https://github.com/vllm-project/llm-compressor
cd llm-compressor
python3 -m pip install -e "./[dev]"
```

This will clone the LLM Compressor repo, install it, and install the development dependencies.

**Code Styling and Formatting checks**

```bash
make style
make quality
```

This will run automatic code styling using `ruff` to test that the repository's code matches its standards.

**EXAMPLE: test changes locally**

```bash
make test
```

This will run the targeted LLM Compressor unit tests for the frameworks specified.

File any error found before changes as an Issue and fix any errors found after making changes before submitting a Pull Request.

## GitHub Workflow

1. Fork the `llm-compressor` repository into your GitHub account: https://github.com/vllm-project/llm-compressor.

2. Clone your fork of the GitHub repository, replacing `<username>` with your GitHub username.

   Use ssh (recommended):

   ```bash
   git clone git@github.com:<username>/llm-compressor.git
   ```

   Or https:

   ```bash
   git clone https://github.com/<username>/llm-compressor.git
   ```

3. Add a remote to keep up with upstream changes.

   ```bash
   git remote add upstream https://github.com/vllm-project/llm-compressor.git
   ```

   If you already have a copy, fetch upstream changes.

   ```bash
   git fetch upstream
   ```

4. Create a feature branch to work in.

   ```bash
   git checkout -b feature-xxx upstream/main
   ```

5. Work in your feature branch.

   ```bash
   git commit -a
   ```

6. Periodically rebase your changes

   ```bash
   git pull --rebase
   ```

7. When done, combine ("squash") related commits into a single one

   ```bash
   git rebase -i upstream/main
   ```

   This will open your editor and allow you to re-order commits and merge them:
   - Re-order the lines to change commit order (to the extent possible without creating conflicts)
   - Prefix commits using `s` (squash) or `f` (fixup) to merge extraneous commits.

8. Submit a pull-request

   ```bash
   git push origin feature-xxx
   ```

   Go to your fork main page

   ```bash
   https://github.com/<username>/llm-compressor
   ```

   If you recently pushed your changes GitHub will automatically pop up a `Compare & pull request` button for any branches you recently pushed to. If you click that button it will automatically offer you to submit your pull-request to the `vllm-project/llm-compressor` repository.

   - Give your pull-request a meaningful title.
     You'll know your title is properly formatted once the `Semantic Pull Request` GitHub check
     transitions from a status of "pending" to "passed".
   - In the description, explain your changes and the problem they are solving.

9. Addressing code review comments

   Repeat steps 5. through 7. to address any code review comments and rebase your changes if necessary.

   Push your updated changes to update the pull request

   ```bash
   git push origin [--force] feature-xxx
   ```

   `--force` may be necessary to overwrite your existing pull request in case your
  commit history was changed when performing the rebase.

   Note: Be careful when using `--force` since you may lose data if you are not careful.

   ```bash
   git push origin --force feature-xxx
   ```
