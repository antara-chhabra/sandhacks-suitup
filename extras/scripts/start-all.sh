#!/bin/bash
# Start all services for SuitUp
# 1. Python backend (port 8000)
# 2. Ollama (port 11434) - run separately: ollama serve & ollama run llama3
# 3. Next.js frontend (port 3000)

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

echo "Starting SuitUp services..."

# Start backend (repo root/backend)
echo "Starting Python backend on port 8000..."
cd "$REPO_ROOT/backend"
uvicorn main:app --reload --port 8000 &
BACKEND_PID=$!
cd "$REPO_ROOT"

# Wait for backend to be ready
sleep 3

# Start frontend (suit-up-web)
echo "Starting Next.js frontend on port 3000..."
cd "$REPO_ROOT/suit-up-web"
npm run dev &
FRONTEND_PID=$!
cd "$REPO_ROOT"

echo ""
echo "Services started!"
echo "  - Backend:  http://localhost:8000"
echo "  - Frontend: http://localhost:3000"
echo ""
echo "Make sure Ollama is running: ollama serve && ollama run llama3"
echo "Press Ctrl+C to stop all services"

wait
