.PHONY: build docs test

BUILD_ARGS := dev # set nightly to build nightly release
PYCHECKDIRS := src tests
PYCHECKGLOBS := 'src/**/*.py' 'tests/**/*.py' 'utils/**/*.py' 'examples/**/*.py' setup.py
# run checks on all files for the repo
quality:
	@echo "Running copyright checks";
	python utils/copyright.py quality $(PYCHECKGLOBS)
	@echo "Running python quality checks";
	black --check $(PYCHECKDIRS);
	isort --check-only $(PYCHECKDIRS);
	flake8 $(PYCHECKDIRS);

# style the code according to accepted standards for the repo
style:
	@echo "Running copyright style";
	python utils/copyright.py style $(PYCHECKGLOBS)
	@echo "Running python styling";
	black $(PYCHECKDIRS);
	isort $(PYCHECKDIRS);

# run tests for the repo
test:
	@echo "Running python tests";
	pytest tests;

# creates wheel file
build:
	@echo "Building the wheel for the repository";
	BUILD_TYPE=$(BUILD_ARGS) python3 setup.py sdist bdist_wheel;

# clean package
clean:
	@echo "Cleaning up";
	rm -rf .pytest_cache;
	find $(PYCHECKDIRS) | grep -E "(__pycache__|\.pyc|\.pyo)" | xargs rm -rf;
