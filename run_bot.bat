@echo off
cd /d "%~dp0"
echo [%date% %time%] Waiting for v2rayN local proxy on 127.0.0.1:10808 >> "bot_autostart.log"
powershell.exe -NoProfile -ExecutionPolicy Bypass -Command "$deadline=(Get-Date).AddSeconds(90); do { $ok=Test-NetConnection -ComputerName 127.0.0.1 -Port 10808 -InformationLevel Quiet -WarningAction SilentlyContinue; if ($ok) { exit 0 }; Start-Sleep -Seconds 2 } while ((Get-Date) -lt $deadline); exit 1"
if errorlevel 1 echo [%date% %time%] v2rayN proxy was not detected within 90 seconds; starting bot anyway >> "bot_autostart.log"
if not errorlevel 1 echo [%date% %time%] v2rayN local proxy is ready >> "bot_autostart.log"
echo [%date% %time%] Starting Telegram bot >> "bot_autostart.log"
".venv\Scripts\python.exe" "bot.py" >> "bot_autostart.log" 2>&1
echo [%date% %time%] Bot process stopped with code %errorlevel% >> "bot_autostart.log"
