@echo off
echo Starting BTC MLOps System...

:: Start the FastAPI Backend in a new minimized window
start /min cmd /c "call venv\Scripts\activate && python api/main.py"

:: Wait a few seconds for the API to connect to ClearML
timeout /t 5

:: Start the Streamlit UI
call venv\Scripts\activate && streamlit run app/ui.py

pause