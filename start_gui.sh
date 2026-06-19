#!/bin/bash
echo "============================================"
echo "  PBL7 - GUI Client"
echo "============================================"
echo ""

cd "$(dirname "$0")"

# Kích hoạt virtual environment
source .venv/bin/activate

echo "[1/2] Starting FastAPI backend..."
python -m uvicorn gui_backend.main:app --host 0.0.0.0 --port 8001 &
BACKEND_PID=$!

echo "[2/2] Starting Vite frontend..."
cd gui
npm run dev -- --host 0.0.0.0 &
FRONTEND_PID=$!

echo ""
echo "  Backend:  http://192.168.1.31:8001"
echo "  Frontend: http://192.168.1.31:5173"
echo ""
echo "Press Ctrl+C to stop both servers"

# Bắt Ctrl+C -> kill cả 2 process
trap "kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; exit 0" INT TERM

# Đợi
wait
