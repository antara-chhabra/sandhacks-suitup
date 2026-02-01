#!/bin/bash
# Start all services for SuitUp
# 1. Python backend (port 8000)
# 2. Ollama (port 11434) - run separately: ollama serve & ollama run llama3
# 3. Next.js frontend (port 3000)

cd "$(dirname "$0")/.."

echo "Starting SuitUp services..."

# Start backend
echo "Starting Python backend on port 8000..."
cd suit-up-web/app/backend
uvicorn main:app --reload --port 8000 &
BACKEND_PID=$!
cd ../../..

# Wait for backend to be ready
sleep 3

# Start frontend
echo "Starting Next.js frontend on port 3000..."
cd suit-up-web
npm run dev &
FRONTEND_PID=$!
cd ..

echo ""
echo "Services started!"
echo "  - Backend:  http://localhost:8000"
echo "  - Frontend: http://localhost:3000"
echo ""
echo "Make sure Ollama is running: ollama serve && ollama run llama3"
echo "Press Ctrl+C to stop all services"

wait
