install:
	pip install flake8
	pip install mypy
	pip install uv

run:
	uv run python3 -m src \
	--functions_definition /goinfre/akoudri/project_1/data/input/functions_definition.json \
	--input data/input/function_calling_tests.json \
	--output /goinfre/akoudri/project_1/data/output/output.json

lint:
	- flake8 src 
	 mypy . --warn-return-any \
	--warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs \
	--check-untyped-defs

clean :
	rm -rf __pycache__ .mypy_cache */__pycache__ */.mypy_cache

debug:
	uv run python3 -m pdb src \
	--functions_definition /home/akoudri/goinfre/project_1/data/input/functions_definition.json \
	--input /home/akoudri/goinfre/project_1/data/input/function_calling_tests.json \
	--output /home/akoudri/goinfre/project_1/data/input/output.json
