.PHONY: start
start:
	uvicorn backend.main:app --reload --port 8000

.PHONY: format
format:
	black .
	isort .