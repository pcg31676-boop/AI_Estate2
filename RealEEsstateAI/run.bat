@echo off
title AI Estate - Property Valuation Server
echo ========================================================
echo   Launching AI Estate | Real Estate Valuation Portal
echo ========================================================
echo.

cd /d "%~dp0"

if exist "C:\Users\hp\anaconda3\Scripts\streamlit.exe" (
    echo Starting Streamlit using Anaconda on http://localhost:8501 ...
    "C:\Users\hp\anaconda3\Scripts\streamlit.exe" run app.py
) else if exist ".venv\Scripts\streamlit.exe" (
    echo Starting Streamlit using .venv on http://localhost:8501 ...
    ".venv\Scripts\streamlit.exe" run app.py
) else (
    echo Starting Streamlit using system PATH on http://localhost:8501 ...
    streamlit run app.py
)
pause
