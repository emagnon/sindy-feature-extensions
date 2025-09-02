.PHONY: fmt lint test

fmt:
	black src tests
	isort src tests

lint:
	flake8 src tests

test:
	pytest -q
