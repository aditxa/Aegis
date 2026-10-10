@echo off
setlocal
if not exist .venv (
  py -3.11 -m venv .venv 2>nul || py -m venv .venv
)
call .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
python scripts\demo.py
pause
