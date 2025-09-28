const dotenv = require('dotenv');

require('dotenv').config({ path: '.env.local' });

const express = require('express');
const multer = require('multer');
const path = require('path');
const fs = require('fs');

const { detectImage, detectImageFromUrl, analyzeForFraud } = require('./service/aiImageDetect');

const app = express();
const PORT = process.env.PORT || 8000;

// Configure multer for file uploads
const storage = multer.diskStorage({
  destination: (req, file, cb) => {
    const uploadDir = 'uploads/';
    if (!fs.existsSync(uploadDir)) {
      fs.mkdirSync(uploadDir, { recursive: true });
    }
    cb(null, uploadDir);
  },
  filename: (req, file, cb) => {
    const uniqueSuffix = Date.now() + '-' + Math.round(Math.random() * 1E9);
    cb(null, file.fieldname + '-' + uniqueSuffix + path.extname(file.originalname));
  }
});

const upload = multer({ 
  storage: storage,
  limits: {
    fileSize: 10 * 1024 * 1024 // 10MB limit
  },
  fileFilter: (req, file, cb) => {
    if (file.mimetype.startsWith('image/')) {
      cb(null, true);
    } else {
      cb(new Error('Only image files are allowed!'), false);
    }
  }
});

// Middleware
app.use(express.json());
app.use(express.urlencoded({ extended: true }));

// CORS middleware
app.use((req, res, next) => {
  res.header('Access-Control-Allow-Origin', '*');
  res.header('Access-Control-Allow-Methods', 'GET, POST, PUT, DELETE, OPTIONS');
  res.header('Access-Control-Allow-Headers', 'Origin, X-Requested-With, Content-Type, Accept, Authorization');
  if (req.method === 'OPTIONS') {
    res.sendStatus(200);
  } else {
    next();
  }
});

// Health check endpoint
app.get('/health', (req, res) => {
  res.status(200).json({
    status: 'OK',
    message: 'Server is running',
    timestamp: new Date().toISOString(),
    uptime: process.uptime()
  });
});

// Enhanced fraud detection endpoint
app.post('/analyze_fraud', async (req, res) => {
  try {
    const { image_url } = req.body;
    if (!image_url) {
      return res.status(400).json({ error: 'image_url is required' });
    }
    
    const result = await analyzeForFraud(image_url);
    res.json({ 
      success: true,
      data: result
    });
  } catch (error) {
    console.error('Error in /analyze_fraud:', error);
    res.status(500).json({ 
      success: false,
      error: error.message 
    });
  }
});


// Basic root endpoint
app.get('/', (req, res) => {
  res.json({
    message: 'iSpy Backend Server',
    version: '1.0.0',
    health: '/health',
    endpoints: {
      'POST /detect_image': 'Detect objects in image from file path',
      'POST /detect_image_url': 'Detect objects in image from URL',
      'POST /upload_and_detect': 'Upload image file and detect objects',
      'POST /analyze_fraud': 'Analyze image URL for fraud detection',
      'POST /upload_and_analyze': 'Upload image file and analyze for fraud'
    }
  });
});

// Start server
app.listen(PORT, () => {
  console.log(`🚀 Server is running on port ${PORT}`);
  console.log(`📊 Health check: http://localhost:${PORT}/health`);
});

module.exports = app;
