.PHONY: install install-max demo api dashboard test
install:
	python -m pip install -r requirements.txt && python -m pip install -e .
install-max:
	python -m pip install -r requirements-max.txt && python -m pip install -e .
demo:
	python scripts/demo.py
api:
	uvicorn aegis.api:app --reload --port 8000
dashboard:
	streamlit run aegis/dashboard.py
test:
	python -m pytest -q
