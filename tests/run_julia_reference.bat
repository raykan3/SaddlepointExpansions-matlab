@echo off
cd /d C:\work\SPA\SaddlepointExpansions-matlab\tests
"%LOCALAPPDATA%\Microsoft\WindowsApps\julia.exe" julia_reference.jl > julia_reference.log 2>&1
echo FINISHED >> julia_reference.log
