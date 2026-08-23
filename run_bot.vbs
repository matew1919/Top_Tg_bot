Option Explicit

Dim shell, fso, projectDir, launcherFile, logFile, logStream
Set shell = CreateObject("WScript.Shell")
Set fso = CreateObject("Scripting.FileSystemObject")

projectDir = fso.GetParentFolderName(WScript.ScriptFullName)
launcherFile = projectDir & "\run_bot.bat"
logFile = projectDir & "\bot_autostart.log"

shell.CurrentDirectory = projectDir

Set logStream = fso.OpenTextFile(logFile, 8, True)
logStream.WriteLine "[" & Now & "] Starting bot launcher; VPN is managed by v2rayN"
logStream.Close

shell.Run """" & launcherFile & """", 0, False
