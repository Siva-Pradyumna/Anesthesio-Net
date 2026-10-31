@echo off
echo ===================================================================
echo        AnesthesioNet-AI: Master System Evaluation
echo ===================================================================
"%~dp0.venv\Scripts\python.exe" "%~dp0evaluate.py" %*
echo.
pause
