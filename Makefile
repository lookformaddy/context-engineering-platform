.PHONY: install dev test demo run docker-build docker-run

install:
	pip install -e .

dev:
	pip install -e ".[dev]"

test:
	PYTHONPATH=src pytest -q

demo:
	PYTHONPATH=src python examples/demo.py

run:
	PYTHONPATH=src uvicorn context_platform.api:app --host 0.0.0.0 --port 8000

docker-build:
	docker build -t context-engineering-platform .

docker-run:
	docker run --rm -p 8000:8000 context-engineering-platform
