@echo off
cd /d "%~dp0"
"C:\ProgramData\Anaconda3\python.exe" "%~dp0scripts\label_tool.py"
if errorlevel 1 (
    echo.
    echo [Error] exit code: %errorlevel%
    pause
)
