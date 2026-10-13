.PHONY: install lint test seed refresh api mcp up down logs

install:
	pip install -e ".[dev]"

lint:
	ruff check .

test:
	pytest -q

seed:
	imms-seed --markets 40 --weeks 104

refresh:
	imms-refresh --profile quick

api:
	imms-api

mcp:
	MCP_TRANSPORT=streamable-http imms-mcp

up:
	docker compose up --build -d

down:
	docker compose down -v

logs:
	docker compose logs -f refresh api mcp
