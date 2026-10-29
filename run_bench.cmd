@echo off
rem Bench wrapper: bypass PowerShell execution policy.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0run_bench.ps1" %*
