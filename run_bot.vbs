Option Explicit

Dim shell, fso, projectDir, launcherFile
Set shell = CreateObject("WScript.Shell")
Set fso = CreateObject("Scripting.FileSystemObject")

projectDir = fso.GetParentFolderName(WScript.ScriptFullName)
launcherFile = projectDir & "\run_bot.bat"

shell.CurrentDirectory = projectDir
shell.Run """" & launcherFile & """", 0, False
