# Minimal makefile for Sphinx documentation

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = source
BUILDDIR      = build

EXAMPLES_SRC  = ../examples
EXAMPLES_DST  = $(SOURCEDIR)/doc

.PHONY: help Makefile clean update-examples html

# Default target
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

# Clean build and copied example files
clean:
	@$(SPHINXBUILD) -M clean "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
	rm -rf $(EXAMPLES_DST)

# Copies .md files from ../examples/ to source/doc/, and updates docs/examples.md
update-examples:
	rm -rf $(EXAMPLES_DST)
	mkdir -p $(EXAMPLES_DST)
	DOCS_ROOT=$$(pwd); \
	cd $(EXAMPLES_SRC) && find . -name '*.md' -type f -exec sh -c ' \
		DOCS_ROOT="'"$$DOCS_ROOT"'"; \
		EXAMPLES_DST="'"$(EXAMPLES_DST)"'"; \
		for f; do \
			mkdir -p "$$DOCS_ROOT/$$EXAMPLES_DST/$$(dirname "$$f")"; \
			cp "$$f" "$$DOCS_ROOT/$$EXAMPLES_DST/$$f"; \
		done' sh {} +
	@echo "Generating source/examples.md..."
	@./scripts/copy-example-docs.sh

# Build docs and optionally serve locally
html: clean update-examples
	@$(SPHINXBUILD) -M html "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
	@if [ -z "$$CI" ]; then \
		echo "Starting local HTTP server at http://localhost:8000"; \
		python3 -m http.server --directory $(BUILDDIR)/html; \
	fi

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
