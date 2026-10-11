@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  py -3 -m venv .venv
  if errorlevel 1 goto :failed
)
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto :failed
".venv\Scripts\python.exe" prepare_assets.py
if errorlevel 1 goto :failed
".venv\Scripts\python.exe" cli.py doctor --verify
if errorlevel 1 goto :failed
echo Setup complete. Double-click Launch.cmd. Future predictions work offline.
pause
exit /b 0
:failed
echo Setup did not finish. Read the error above. Internet is needed for initial setup.
pause
exit /b 1
