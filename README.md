# iSpy

A full-stack web application with React+Vite frontend (TypeScript + TailwindCSS) and FastAPI backend with Google Gemini AI integration.

## Project Structure

```
ispy/
├── frontend/          # React + Vite + TypeScript + TailwindCSS
├── backend/           # FastAPI + Python + Gemini AI
├── package.json       # Root package.json with scripts
└── README.md
```

## Features

- **Frontend**: Modern React application with Vite, TypeScript, and TailwindCSS
- **Backend**: FastAPI server with Google Gemini AI integration
- **AI Chat**: Interactive chat interface to communicate with Gemini AI
- **Real-time Communication**: Frontend-backend communication via REST API
- **Responsive Design**: Mobile-friendly UI with TailwindCSS

## Prerequisites

- Node.js (v20 or higher)
- Python 3.12 or higher
- pip3
- Google Gemini API key

## Setup Instructions

### 1. Clone the repository

```bash
git clone <repository-url>
cd ispy
```

### 2. Install all dependencies

```bash
npm run install:all
```

### 3. Configure Gemini API

1. Get your Gemini API key from [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Copy the environment file:
   ```bash
   cp backend/.env.example backend/.env
   ```
3. Edit `backend/.env` and add your API key:
   ```
   GEMINI_API_KEY=your_actual_api_key_here
   ```

### 4. Run the application

#### Development mode (runs both frontend and backend):
```bash
npm run dev
```

#### Or run separately:

**Frontend only:**
```bash
npm run dev:frontend
# Opens at http://localhost:5173
```

**Backend only:**
```bash
npm run dev:backend
# Runs at http://localhost:8000
```

### 5. Build for production

```bash
npm run build
```

## API Endpoints

- `GET /` - Root endpoint
- `GET /health` - Health check
- `POST /api/chat` - Chat with Gemini AI
  ```json
  {
    "message": "Your message here"
  }
  ```

## Development

### Frontend Development
- Located in `frontend/` directory
- Built with Vite + React + TypeScript
- Styled with TailwindCSS
- Hot reload enabled in development

### Backend Development
- Located in `backend/` directory
- Built with FastAPI + Python
- Google Gemini AI integration
- CORS enabled for frontend communication

### Available Scripts

- `npm run dev` - Start both frontend and backend
- `npm run dev:frontend` - Start only frontend
- `npm run dev:backend` - Start only backend
- `npm run build` - Build frontend for production
- `npm run lint` - Lint frontend code
- `npm run install:all` - Install all dependencies

## Environment Variables

### Backend (.env)
- `GEMINI_API_KEY` - Your Google Gemini API key

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test your changes
5. Submit a pull request

## License

MIT License