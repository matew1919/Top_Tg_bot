@echo off
cd /d "%~dp0"
echo [%date% %time%] Starting Telegram bot >> "bot_autostart.log"
".venv\Scripts\python.exe" "bot.py" >> "bot_autostart.log" 2>&1
echo [%date% %time%] Bot process stopped with code %errorlevel% >> "bot_autostart.log"
