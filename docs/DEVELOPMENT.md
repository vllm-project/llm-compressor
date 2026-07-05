# Getting started with LLM Compressor docs

```bash
cd docs
```

- Install the dependencies:

```bash
make install
```

- Clean the previous build (optional but recommended):

```bash
make clean
```

- Generate docs content (files, API references, and navigation):

```bash
make gen
```

- Serve the docs locally (runs `gen` automatically):

```bash
make serve
```

This will start a local server. You can now open your browser and view the documentation.

- Build the static site (runs `gen` automatically):

```bash
make build
```

- List all available targets:

```bash
make help
```