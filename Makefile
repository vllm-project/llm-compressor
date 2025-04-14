BUILDDIR := $(PWD)
CHECKDIRS := src tests examples setup.py
DOCDIR := docs

BUILD_ARGS :=  # set nightly to build nightly release

# refer to setup.py for allowed values for BUILD_TYPE
BUILD_TYPE?=dev
export BUILD_TYPE

TARGETS := ""  # targets for running pytests: deepsparse,keras,onnx,pytorch,pytorch_models,export,pytorch_datasets,tensorflow_v1,tensorflow_v1_models,tensorflow_v1_datasets
PYTEST_ARGS ?= ""
ifneq ($(findstring transformers,$(TARGETS)),transformers)
    PYTEST_ARGS := $(PYTEST_ARGS) --ignore tests/llmcompressor/transformers
endif
ifneq ($(findstring pytorch,$(TARGETS)),pytorch)
    PYTEST_ARGS := $(PYTEST_ARGS) --ignore tests/llmcompressor/pytorch
endif
ifneq ($(findstring examples,$(TARGETS)),examples)
    PYTEST_ARGS := $(PYTEST_ARGS) --ignore tests/examples
endif

# run checks on all files for the repo
# leaving out mypy src for now
quality:
	@echo "Running python quality checks";
	ruff check $(CHECKDIRS);
	isort --check-only $(CHECKDIRS);
	flake8 $(CHECKDIRS) --max-line-length 88 --extend-ignore E203;

# style the code according to accepted standards for the repo
style:
	@echo "Running python styling";
	ruff format $(CHECKDIRS);
	isort $(CHECKDIRS);
	flake8 $(CHECKDIRS) --max-line-length 88 --extend-ignore E203;

# run tests for the repo
test:
	@echo "Running python tests";
	pytest tests $(PYTEST_ARGS)

# creates wheel file
.PHONY: build
build:
	python3 setup.py sdist bdist_wheel $(BUILD_ARGS)

# clean package
clean:
	rm -fr .pytest_cache;
	rm -fr docs/_build docs/build;
	find $(CHECKDIRS) | grep -E "(__pycache__|\.pyc|\.pyo)" | xargs rm -fr;
