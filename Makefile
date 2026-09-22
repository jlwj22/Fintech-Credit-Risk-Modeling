.PHONY: install data train train-lc test lint app api docker

install:
	pip install -r requirements-dev.txt

data:
	python -m src.download_data

train:
	python -m src.train

train-lc:
	python -m src.lc_data
	python -m src.lc_train

test:
	pytest

lint:
	ruff check .

app:
	streamlit run app.py

api:
	uvicorn src.api:app --reload

docker:
	docker build -t credit-risk-api .
	docker run --rm -p 8000:8000 credit-risk-api
