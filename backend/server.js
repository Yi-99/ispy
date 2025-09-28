const dotenv = require('dotenv');

require('dotenv').config({ path: '.env' });

const express = require('express');
const multer = require('multer');
const path = require('path');
const fs = require('fs');

const { detectImage, detectImageFromUrl, analyzeForFraud } = require('./service/aiImageDetect');
const { parseClaimPdf } = require('./service/documentScan');
const { spawn } = require('child_process');

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
app.use(express.json({ limit: '50mb' }));
app.use(express.urlencoded({ extended: true, limit: '50mb' }));

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

// Document scan routes
app.post('/api/parse-claim-pdf', parseClaimPdf);

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

// Fraud prediction endpoint using Python scorer
app.post('/predict', async (req, res) => {
  try {
    const payload = req.body;
    if (!payload) {
      return res.status(400).json({ error: 'No payload provided' });
    }

    // Call Python scorer
    const py = spawn('python3', ['ml/scorer.py'], {
      env: {
        ...process.env,
        MODEL_BUNDLE_PATH: 'ml/models/fraud_ensemble_bundle.joblib',
        THRESHOLD: '0.45'
      }
    });

    let stdout = '';
    let stderr = '';

    py.stdout.on('data', (data) => {
      stdout += data.toString();
    });

    py.stderr.on('data', (data) => {
      stderr += data.toString();
    });

    py.on('error', (err) => {
      console.error('Python spawn error:', err);
      return res.status(500).json({ error: `Python spawn error: ${err.message}` });
    });

    py.on('close', (code) => {
      if (code !== 0) {
        console.error(`Python scorer exited with code ${code}, stderr: ${stderr}`);
        return res.status(500).json({ 
          error: `Python scorer failed with code ${code}`, 
          stderr: stderr 
        });
      }

      try {
        const result = JSON.parse(stdout);
        res.json({
          success: true,
          data: result
        });
      } catch (parseError) {
        console.error('JSON parse error:', parseError);
        res.status(500).json({ 
          error: 'Failed to parse Python output', 
          stdout: stdout,
          parseError: parseError.message 
        });
      }
    });

    // Send payload to Python script
    py.stdin.write(JSON.stringify(payload));
    py.stdin.end();

  } catch (error) {
    console.error('Error in /predict:', error);
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
      'POST /upload_and_analyze': 'Upload image file and analyze for fraud',
      'POST /api/parse-claim-pdf': 'Parse insurance claim PDF and extract structured data',
      'POST /predict': 'Predict fraud probability using ML model'
    }
  });
});

// Start server
app.listen(PORT, () => {
  console.log(`🚀 Server is running on port ${PORT}`);
  console.log(`📊 Health check: http://localhost:${PORT}/health`);
});

module.exports = app;
