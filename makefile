PYCHECKDIRS := tests src
# run checks on all files for the repo
quality:
	@echo "Running python quality checks";
	black --check $(PYCHECKDIRS);
	isort --check-only $(PYCHECKDIRS);
	flake8 $(PYCHECKDIRS);

# style the code according to accepted standards for the repo
style:
	@echo "Running python styling";
	black $(PYCHECKDIRS);
	isort $(PYCHECKDIRS);

# run tests for the repo
test:
	@echo "Running python tests";
	pytest tests;

# clean package
clean:
	@echo "Cleaning up";
	rm -rf .pytest_cache;
	rm -rf src/sparsezoo.egg-info;
	find $(PYCHECKDIRS) | grep -E "(__pycache__|\.pyc|\.pyo)" | xargs rm -rf;