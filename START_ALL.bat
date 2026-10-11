@echo off
cd /d "%~dp0"
echo Launching AegisSignal-X Omega v0.5 Research Desk...
start "Aegis API" cmd /k python -m uvicorn aegis.api:app --host 127.0.0.1 --port 8000
start "Aegis Risk Desk" cmd /k python -m streamlit run aegis\dashboard.py
start "" http://localhost:8501
start "" http://127.0.0.1:8000/docs
