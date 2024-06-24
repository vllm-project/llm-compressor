<!--
Copyright (c) 2021 - present / Neuralmagic, Inc. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing,
software distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

TODO: update for upstream push

# Developing SparseML

SparseML is developed and tested using Python 3.8-3.11.
To develop SparseML, you will also need the development dependencies and to follow the styling guidelines.

Here are some details to get started.

## Basic Commands

**Development Installation**

```bash
git https://github.com/vllm-project/llm-compressor
cd llm-compressor
python3 -m pip install -e "./[dev]"
```

This will clone the SparseML repo, install it, and install the development dependencies.

To develop framework specific features, you will also need the relevant framework packages.
Those can be installed by adding the framework name to the install extras. Frameworks include
`torch`, `keras`, and `tensorflow_v1`. For example:
```bash
python3 -m pip install -e "./[dev,torch]"
```

Note: Running all pytorch tests using `make test TARGETS=torch`, also requires `torchvision`
and `onnxruntime` install all these dependencies using `python3 -m pip install -e "./[dev, torch, torchvision, onnxruntime]"`

**Code Styling and Formatting checks**

```bash
make style
make quality
```

This will run automatic code styling using `black` and `isort` and test that the
repository's code matches its standards.

**EXAMPLE: test changes locally**

```bash
make test TARGETS=<CSV of frameworks to run>
```

This will run the targeted SparseML unit tests for the frameworks specified.
The targets should be specified, because not all framework dependencies can be installed to run all tests.

To run just PyTorch tests, run
```bash
make test TARGETS=pytorch
```

File any error found before changes as an Issue and fix any errors found after making changes before submitting a Pull Request.

## GitHub Workflow

1. Fork the `llmcompressor` repository into your GitHub account: https://github.com/vllm-project/llm-compressor.

2. Clone your fork of the GitHub repository, replacing `<username>` with your GitHub username.

   Use ssh (recommended):

   ```bash
   git clone git@github.com:<username>/llmcompressor.git
   ```

   Or https:

   ```bash
   git clone https://github.com/<username>/llmcompressor.git
   ```

3. Add a remote to keep up with upstream changes.

   ```bash
   git remote add upstream https://github.com/neuralmagic/sparseml.git
   ```

   If you already have a copy, fetch upstream changes.

   ```bash
   git fetch upstream
   ```

4. Create a feature branch to work in.

   ```bash
   git checkout -b feature-xxx remotes/upstream/main
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
   https://github.com/<username>/llmcompressor
   ```

   If you recently pushed your changes GitHub will automatically pop up a `Compare & pull request` button for any branches you recently pushed to. If you click that button it will automatically offer you to submit your pull-request to the `TODO/llmcompressor` repository.

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
