Option Explicit

Dim shell, fso, projectDir, pythonExe, botFile
Set shell = CreateObject("WScript.Shell")
Set fso = CreateObject("Scripting.FileSystemObject")

projectDir = fso.GetParentFolderName(WScript.ScriptFullName)
pythonExe = projectDir & "\.venv\Scripts\pythonw.exe"
If Not fso.FileExists(pythonExe) Then
    pythonExe = projectDir & "\.venv\Scripts\python.exe"
End If
botFile = projectDir & "\bot.py"

shell.CurrentDirectory = projectDir
shell.Run """" & pythonExe & """ """ & botFile & """", 0, False
