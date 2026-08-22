Option Explicit

Dim shell, fso, projectDir, launcherFile, happExe, connectScript, logFile, logStream
Set shell = CreateObject("WScript.Shell")
Set fso = CreateObject("Scripting.FileSystemObject")

projectDir = fso.GetParentFolderName(WScript.ScriptFullName)
launcherFile = projectDir & "\run_bot.bat"
happExe = "C:\Program Files\FlyFrogLLC\Happ\Happ.exe"
connectScript = projectDir & "\connect_happ.ps1"
logFile = projectDir & "\bot_autostart.log"

shell.CurrentDirectory = projectDir

If fso.FileExists(happExe) Then
    Set logStream = fso.OpenTextFile(logFile, 8, True)
    logStream.WriteLine "[" & Now & "] Starting HAPP VPN"
    logStream.Close
    shell.Run """" & happExe & """", 1, False
    If fso.FileExists(connectScript) Then
        shell.Run "powershell.exe -NoProfile -ExecutionPolicy Bypass -File """ & connectScript & """", 0, True
    Else
        WScript.Sleep 10000
    End If
Else
    Set logStream = fso.OpenTextFile(logFile, 8, True)
    logStream.WriteLine "[" & Now & "] HAPP not found: " & happExe
    logStream.Close
End If

shell.Run """" & launcherFile & """", 0, False
