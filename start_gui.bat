@echo off
title PBL7 - GUI Client
echo ============================================
echo   PBL7 - GUI Client
echo ============================================
echo.

cd /d "%~dp0"

echo [1/2] Khoi dong FastAPI backend...
start "PBL7-Backend" cmd /k "cd /d %~dp0 && python -m uvicorn gui_backend.main:app --host 127.0.0.1 --port 8001 --reload"

echo [2/2] Khoi dong Vite frontend...
start "PBL7-Frontend" cmd /k "cd /d %~dp0gui && npm run dev"

timeout /t 4 /nobreak >nul
start http://localhost:5173

echo.
echo Da khoi dong. Backend: http://localhost:8001 ^| Frontend: http://localhost:5173
echo.
pause
