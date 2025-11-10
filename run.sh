#!/bin/bash
# Simple runner for FastAPI + Vite dev servers

# Start backend in background
echo "Starting FastAPI backend..."
uvicorn backend.main:app --reload --port 8000 &

# Save its PID so we can clean up later
BACK_PID=$!

# Start frontend (Vite)
echo "Starting React frontend..."
cd frontend
npm run dev

# When frontend stops, kill backend
kill $BACK_PID
