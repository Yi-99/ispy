#!/bin/bash

# iSpy Startup Script
echo "🚀 Starting iSpy Application..."
echo ""

# Check if Node.js is available
if ! command -v node &> /dev/null; then
    echo "❌ Node.js is not installed. Please install Node.js first."
    exit 1
fi

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3 first."
    exit 1
fi

# Check if dependencies are installed
if [ ! -d "frontend/node_modules" ] || [ ! -f "backend/.env" ]; then
    echo "📦 Installing dependencies and setting up environment..."
    npm run install:all
    
    if [ ! -f "backend/.env" ]; then
        echo "⚠️  Creating .env file from template..."
        cp backend/.env.example backend/.env
        echo "✏️  Please edit backend/.env and add your Gemini API key!"
    fi
fi

echo "🎉 Starting both frontend and backend servers..."
echo "Frontend will be available at: http://localhost:5173"
echo "Backend will be available at: http://localhost:8000"
echo ""
echo "Press Ctrl+C to stop both servers"
echo ""

# Start both servers
npm run dev