@echo off
rem Double-click to start AOMe. The engine starts automatically and the app opens.
rem Close this window to quit AOMe.
cd /d "%~dp0"
where python >nul 2>nul || (echo Python was not found. Install Python 3.10-3.12 from python.org, or use AOMe.exe instead. & pause & exit /b 1)
python run.py
if errorlevel 1 (echo. & echo AOMe stopped with an error. See the log in %%LOCALAPPDATA%%\AOMe\logs\aome.log & pause)
