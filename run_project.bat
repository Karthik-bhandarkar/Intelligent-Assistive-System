@echo off
echo ========================================================
echo   Smart Assistive Vision System - Launcher
echo ========================================================
echo.
echo Activating Virtual Environment...
call venv\Scripts\activate

echo.
echo Starting Application...
streamlit run streamlit_app.py

pause
