.PHONY: install train run test lint

install:
	python -m pip install -r requirements-dev.txt

train:
	python -m ml.train

run:
	uvicorn app.main:app --reload

test:
	pytest -q

lint:
	ruff check .

