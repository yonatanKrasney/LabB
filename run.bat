@echo off
rem ===================================================================
rem  Lab B fitting tool -- double-click this file to start.
rem
rem  First run sets up a private Python environment and installs what
rem  it needs. That takes a couple of minutes and happens only once.
rem  Every run after that starts in a few seconds.
rem ===================================================================

cd /d "%~dp0"
title Lab B fitting

rem The environment lives outside this folder on purpose: putting thousands
rem of package files inside OneDrive would make it sync them all forever.
set "VENVDIR=%LOCALAPPDATA%\labB-fitting\venv"
set "PY=%VENVDIR%\Scripts\python.exe"

if exist "%PY%" goto :have_env

echo.
echo  First run: setting up. A couple of minutes, only this once.
echo.

set "BASEPY="
py -3 --version >nul 2>&1
if %errorlevel%==0 set "BASEPY=py -3"
if defined BASEPY goto :make_env

python --version >nul 2>&1
if %errorlevel%==0 set "BASEPY=python"
if defined BASEPY goto :make_env

echo  Python was not found on this computer.
echo.
echo  Install it from  https://www.python.org/downloads/windows/
echo  and tick "Add python.exe to PATH" during the install.
echo  Then run this file again.
echo.
pause
exit /b 1

:make_env
%BASEPY% -m venv "%VENVDIR%"
if not exist "%PY%" (
  echo.
  echo  Could not create the Python environment in:
  echo  %VENVDIR%
  echo.
  pause
  exit /b 1
)
"%PY%" -m pip install --upgrade pip --quiet
echo  Installing packages...
"%PY%" -m pip install -r requirements.txt
if errorlevel 1 (
  echo.
  echo  The install failed. Check that you are online, then run this again.
  echo.
  pause
  exit /b 1
)

:have_env
rem If a package went missing or requirements changed, top it up quietly.
"%PY%" -c "import streamlit, scipy, plotly, pandas, matplotlib, openpyxl" >nul 2>&1
if errorlevel 1 (
  echo  Updating packages...
  "%PY%" -m pip install -r requirements.txt
)

if not exist "data" mkdir "data"

echo.
echo  Starting. Your browser will open at http://localhost:8501
echo  Put your CSV or Excel files in the  data  folder next to this file.
echo.
echo  Keep this window open while you work. Close it to stop the app.
echo.

"%PY%" -m streamlit run app.py

echo.
echo  The app has stopped.
pause
