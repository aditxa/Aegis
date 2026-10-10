@echo off
setlocal
if not exist .venv (
  py -3.11 -m venv .venv 2>nul || py -m venv .venv
)
call .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
uvicorn aegis.api:app --reload --port 8000
