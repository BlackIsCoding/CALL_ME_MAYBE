install:
	@pip install flake8
	@pip install mypy
	@pip install uv
	@uv sync

run:
	@uv run python3 -m src \
	--functions_definition /goinfre/akoudri/project_1/data/input/functions_definition.json \
	--input data/input/function_calling_tests.json \
	--output /goinfre/akoudri/project_1/data/output/output.json

lint:
	@- uv run flake8 src 
	@ uv run mypy src --warn-return-any \
	--warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs \
	--check-untyped-defs

.PHONY: clean

clean:
	@find . -type d \( -name "__pycache__" -o -name ".mypy_cache" \) -exec rm -rf {} +

debug:
	uv run python3 -m pdb -m src \
	--functions_definition /home/akoudri/goinfre/project_1/data/input/functions_definition.json \
	--input /home/akoudri/goinfre/project_1/data/input/function_calling_tests.json \
	--output /home/akoudri/goinfre/project_1/data/input/output.json
