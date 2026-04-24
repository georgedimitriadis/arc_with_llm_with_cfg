@echo off
REM ============================================================
REM  Start llama.cpp server with Qwen3-Coder-30B-A3B
REM  Keeps the window open so you can read logs / errors
REM
REM  Usage:
REM    "Llama server.bat"                      (no grammar)
REM    "Llama server.bat" "C:\path\to\file.gbnf"
REM ============================================================

title Qwen3.6 llama-server

REM --- Edit this if you extracted llama.cpp somewhere else ---
set LLAMA_DIR=G:\Software_Tools\llama.cpp
set MODEL_PATH=G:\Code\ML_Large_Models\Qwen\Qwen3.6-35B-A3B-UD-Q4_K_XL\Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf

cd /d "%LLAMA_DIR%"

echo Starting llama-server...
echo Model: %MODEL_PATH%
echo Endpoint: http://127.0.0.1:8080
echo Press Ctrl+C to stop.
echo.

llama-server.exe ^
  --model "%MODEL_PATH%" ^
  --n-gpu-layers 999 ^
  --ctx-size 100000 ^
  --host 127.0.0.1 ^
  --port 8080 ^
  --jinja


echo.
echo ============================================================
echo llama-server has exited. Check messages above for errors.
echo ============================================================
pause