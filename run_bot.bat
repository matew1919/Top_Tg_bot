@echo off
cd /d "%~dp0"
if exist "C:\Program Files\FlyFrogLLC\Happ\Happ.exe" (
    echo [%date% %time%] Waiting for HAPP VPN tunnel >> "bot_autostart.log"
    powershell.exe -NoProfile -ExecutionPolicy Bypass -Command "$deadline=(Get-Date).AddSeconds(180); do { $adapter=Get-NetAdapter -Name 'happ-tun' -ErrorAction SilentlyContinue; if ($adapter -and $adapter.Status -eq 'Up') { exit 0 }; Start-Sleep -Seconds 2 } while ((Get-Date) -lt $deadline); exit 1"
    if errorlevel 1 echo [%date% %time%] VPN tunnel did not become ready within 180 seconds >> "bot_autostart.log"
    if not errorlevel 1 echo [%date% %time%] HAPP VPN tunnel is ready >> "bot_autostart.log"
)
echo [%date% %time%] Starting Telegram bot >> "bot_autostart.log"
".venv\Scripts\python.exe" "bot.py" >> "bot_autostart.log" 2>&1
echo [%date% %time%] Bot process stopped with code %errorlevel% >> "bot_autostart.log"
