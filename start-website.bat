@echo off
cd /d "%~dp0"
echo Installing/checking required Python packages...
py -m pip install -r requirements.txt
if errorlevel 1 goto failed
echo.
echo Starting DigitLab. This window must stay open.
py app.py
if errorlevel 1 goto failed
goto done
:failed
echo.
echo DigitLab did not start. Read the error above.
pause
:done
