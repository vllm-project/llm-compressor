# Minimal mkdocs makefile

PYTHON      := python3
MKDOCS_CMD  := mkdocs
MKDOCS_CONF := ../mkdocs.yml

.PHONY: help install serve build clean

help:
	@echo "Available targets:"
	@echo "  install  Install dependencies globally"
	@echo "  serve    Serve docs locally"
	@echo "  build    Build static site"
	@echo "  clean    Remove build artifacts"

install:
	pip install -e "../[dev]"

serve:
	$(MKDOCS_CMD) serve --livereload -f $(MKDOCS_CONF)

build:
	$(MKDOCS_CMD) build -f $(MKDOCS_CONF)

clean:
	rm -rf site/ .cache/ 