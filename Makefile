PY ?= .venv/bin/python
FC ?= .venv/bin/forecastmcp

.PHONY: help install lint test bench serve build docker-test clean

help:
	@echo "install      pip install -e '.[dev]'"
	@echo "lint         ruff check"
	@echo "test         pytest with coverage"
	@echo "bench        run the full benchmark"
	@echo "serve        run the MCP server over stdio"
	@echo "build        docker build"
	@echo "docker-test  pytest inside the built image"

install:
	$(PY) -m pip install -e ".[dev]"

lint:
	$(PY) -m ruff check forecastmcp tests

test:
	$(PY) -m pytest --cov=forecastmcp.eval --cov=forecastmcp.mcp --cov=forecastmcp.agent \
		--cov-report=term-missing --cov-fail-under=80

bench:
	$(FC) bench --series 50 --windows 6 --resamples 1000
	$(FC) eval plots
	$(FC) eval summary

serve:
	$(FC) serve

build:
	docker build -t forecastmcp:latest .

docker-test:
	docker run --rm forecastmcp:latest version

clean:
	rm -rf .pytest_cache .ruff_cache htmlcov .coverage
