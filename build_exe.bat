@echo off
setlocal
cd /d "%~dp0"
rem ============================================================================
rem  Builds dist\AOMe\AOMe.exe  (Windows, 64-bit Python 3.10-3.12; found automatically via the "py" launcher)
rem  Takes 10-30 minutes and several GB of disk. NVIDIA GPU build (CUDA 12.4).
rem  For a smaller CPU-only build, change "cu124" to "cpu" in the TORCH line.
rem ============================================================================
set "TORCH=https://download.pytorch.org/whl/cu124"

set "PY="
for %%V in (3.12 3.11 3.10) do (
  if not defined PY (
    py -%%V -c "import struct,sys; sys.exit(0 if struct.calcsize('P')==8 else 1)" >nul 2>nul && set "PY=py -%%V"
  )
)
if not defined PY (
  python -c "import sys,struct; sys.exit(0 if (3,10)<=sys.version_info[:2]<=(3,12) and struct.calcsize('P')==8 else 1)" >nul 2>nul && set "PY=python"
)
if not defined PY (
  echo [X] No suitable Python found. AOMe needs 64-bit Python 3.12 ^(3.10-3.12 work; 3.13 and 3.14 do NOT work yet^).
  echo     Install Python 3.12 from https://www.python.org/downloads/release/python-3129/ ^(Windows installer 64-bit^).
  echo     You can keep your newer Python installed too - this script finds 3.12 by itself. Then run this file again.
  goto :fail
)
echo Using: %PY%
%PY% --version

echo [1/5] Creating a clean build environment...
if exist .venv-build rmdir /s /q .venv-build
%PY% -m venv .venv-build || goto :fail
call .venv-build\Scripts\activate || goto :fail
python -m pip install --upgrade pip || goto :fail

echo [2/5] Installing PyTorch first (peft/accelerate would otherwise pull the slow CPU build)...
pip install torch --index-url %TORCH% || goto :fail

echo [3/5] Installing AOMe requirements + pywebview + PyInstaller...
pip install -r requirements.txt pywebview pyinstaller || goto :fail
python -c "import torch; print('torch', torch.__version__, '| CUDA build:', torch.version.cuda)" || goto :fail

echo [4/5] Building the exe (be patient)...
pyinstaller --noconfirm --clean AOMe.spec || goto :fail
if not exist dist\AOMe\AOMe.exe (echo [X] The build finished but AOMe.exe is missing. & goto :fail)

echo [5/5] Self-test of the built exe...
start /wait "" dist\AOMe\AOMe.exe --selftest --strict
set "RC=%ERRORLEVEL%"
if exist "%LOCALAPPDATA%\AOMe\logs\selftest.txt" type "%LOCALAPPDATA%\AOMe\logs\selftest.txt"
if not "%RC%"=="0" (echo. & echo [X] Self-test found problems - see the FAIL lines above. & goto :fail)

echo.
echo ===== Done: dist\AOMe\AOMe.exe  (self-test passed) =====
pause
exit /b 0

:fail
echo.
echo ===== BUILD FAILED - copy the red/[X] lines above if you need help =====
pause
exit /b 1
