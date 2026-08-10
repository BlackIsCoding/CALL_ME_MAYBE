# Don't forget .gitgnore

DIR = $(CURDIR)
FUNCTIONS_DEFINITION := $(CURDIR)/data/input/functions_definition.json
INPUT := $(CURDIR)/data/input/function_calling_tests.json
OUTPUT := $(CURDIR)/data/output/function_calling_results.json

install:
	@pip install flake8
	@pip install mypy
	@pip install uv
	@uv sync

run:
	@uv run python3 -m src \
		--functions_definition $(FUNCTIONS_DEFINITION) \
		--input $(INPUT) \
		--output $(OUTPUT)

lint:
	@-uv run flake8 src
	@uv run mypy src \
		--warn-return-any \
		--warn-unused-ignores \
		--ignore-missing-imports \
		--disallow-untyped-defs \
		--check-untyped-defs

.PHONY: clean
clean:
	@find . -type d \( -name "__pycache__" -o -name ".mypy_cache" \) -exec rm -rf {} +

debug:
	@uv run python3 -m pdb -m src \
		--functions_definition $(FUNCTIONS_DEFINITION) \
		--input $(INPUT) \
		--output $(OUTPUT)
