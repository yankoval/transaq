rem run sentimentIndicator
@echo on
echo Starting....
SET mypath=%~dp0
echo %mypath:~0,-1%

call C:\Users\ivan\PycharmProjects\market\venv\Scripts\activate.bat
REM cd C:\Users\ivan\AppData\Roaming\JetBrains\PyCharm2024.1\scratches
cd %mypath%

START "=== sentiment_indicator === " /MIN python.exe sentiment_indicator.py -c update


echo exiting...
timeout 3 > NULL
rem pause
