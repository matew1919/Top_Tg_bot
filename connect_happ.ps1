$ErrorActionPreference = 'Stop'
$logPath = Join-Path $PSScriptRoot 'happ_autostart.log'

function Write-Log([string]$message) {
    Add-Content -LiteralPath $logPath -Value ("[{0}] {1}" -f (Get-Date), $message)
}

try {
    Add-Type -AssemblyName UIAutomationClient, UIAutomationTypes
    $deadline = (Get-Date).AddSeconds(60)
    $process = $null
    while ((Get-Date) -lt $deadline) {
        $process = Get-Process -Name Happ -ErrorAction SilentlyContinue | Select-Object -First 1
        if ($process -and $process.MainWindowHandle -ne 0) { break }
        Start-Sleep -Seconds 1
    }
    if (-not $process -or $process.MainWindowHandle -eq 0) {
        Write-Log 'HAPP window was not available; VPN was not toggled'
        exit 1
    }

    $root = [System.Windows.Automation.AutomationElement]::FromHandle($process.MainWindowHandle)
    $condition = New-Object System.Windows.Automation.PropertyCondition(
        [System.Windows.Automation.AutomationElement]::NameProperty, 'TUN'
    )
    $tun = $root.FindFirst([System.Windows.Automation.TreeScope]::Descendants, $condition)
    if (-not $tun) {
        Write-Log 'HAPP TUN switch was not found; VPN was not toggled'
        exit 1
    }

    $pattern = $tun.GetCurrentPattern([System.Windows.Automation.TogglePattern]::Pattern)
    if ($pattern.Current.ToggleState -eq [System.Windows.Automation.ToggleState]::Off) {
        $pattern.Toggle()
        Write-Log 'HAPP TUN switch enabled'
    } else {
        Write-Log 'HAPP TUN switch was already enabled'
    }
} catch {
    Write-Log ('HAPP auto-connect failed: ' + $_.Exception.Message)
    exit 1
}
