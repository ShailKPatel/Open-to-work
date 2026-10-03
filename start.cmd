@echo off
rem Windows launcher: runs start.ps1 without changing PowerShell's script policy.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0start.ps1"
if errorlevel 1 pause
