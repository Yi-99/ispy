const dotenv = require('dotenv');

require('dotenv').config({ path: '.env' });

const express = require('express');
const multer = require('multer');
const path = require('path');
const fs = require('fs');

const { detectImage, detectImageFromUrl, analyzeForFraud } = require('./service/aiImageDetect');
const { parseClaimPdf } = require('./service/documentScan');
const { aggregateRisk, analyzeRisk } = require('./service/aggregateRisk');
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

// 통합 분석 엔드포인트
app.post('/api/analyze-comprehensive', async (req, res) => {
  try {
    const { 
      imageAnalysis = [], 
      documentAnalysis = [],
      weights = { wDoc: 0.2, wImg: 0.2, wGen: 0.6, thrLow: 0.2, thrHigh: 0.4 }
    } = req.body;

    if (!imageAnalysis.length && !documentAnalysis.length) {
      return res.status(400).json({ 
        success: false, 
        error: 'No analysis data provided' 
      });
    }

    // aggregateRisk 입력 데이터 준비
    const docProbas = documentAnalysis.map(doc => doc.fraudPrediction?.proba || 0);
    const imgProbas = imageAnalysis.map(img => img.analysis?.fraudScore || 0);
    const imgUncerts = imageAnalysis.map(img => img.analysis?.uncertainty || 0);
    const aiScores = imageAnalysis.map(img => img.analysis?.aiScore || 0);

    const imgProb = imgProbas.length ? Math.max(...imgProbas) : 0;
    const imgUnc = imgUncerts.length ? Math.min(1, Math.max(0, imgUncerts.reduce((a, b) => a + b, 0) / imgUncerts.length)) : 0;
    const aiScore = aiScores.length ? Math.max(...aiScores) : 0;

    // aggregateRisk 계산
    const aggResult = aggregateRisk(
      { docProbas, imgProb, imgUnc, aiScore },
      weights
    );

    // analyzeRisk 페이로드 구성
    const docItemsForSignal = documentAnalysis.map(doc => ({
      proba: doc.fraudPrediction?.proba || 0,
      decision: doc.fraudPrediction?.decision || 0
    }));

    let topIdx = -1;
    let topVal = -1;
    docProbas.forEach((v, i) => {
      if (v > topVal) {
        topVal = v;
        topIdx = i;
      }
    });

    const payload = {
      overall: { 
        score: aggResult.overall, 
        level: aggResult.level, 
        thresholds: aggResult.thresholds 
      },
      signals: {
        document: {
          type: 'tabular_fraud',
          items: docItemsForSignal,
          top_suspicious: topIdx >= 0 ? { index: topIdx, proba: topVal } : null,
          model: 'fraud_rf_v1',
        },
        image_fraud: {
          type: 'visual_fraud',
          proba: imgProbas.length ? imgProb : null,
          risk: imgProb >= 0.7 ? 'High' : imgProb >= 0.4 ? 'Medium' : (imgProb > 0 ? 'Low' : null),
          uncertainty: imgUncerts.length ? imgUnc : null,
          model: imgProbas.length ? 'img_model_v1' : null,
        },
        ai_generated: {
          type: 'gen_detector',
          score: aiScores.length ? aiScore : null,
          risk: aiScore >= 0.7 ? 'High' : aiScore >= 0.4 ? 'Medium' : (aiScores.length ? 'Low' : null),
        },
      },
      consistency: {
        media_consistency: (docProbas.length && imgProbas.length)
          ? ((imgProb >= 0.7) === ((Math.max(0, ...docProbas)) >= 0.7) ? 'consistent' : 'inconsistent')
          : 'unknown',
        notes: [],
      },
    };

    // Gemini AI 분석 (getGenAI 함수 필요)
    let aiAnalysis = null;
    try {
      // GoogleGenerativeAI 동적 import
      const { GoogleGenerativeAI } = await import('@google/generative-ai');
      
      // getGenAI 함수 생성
      const getGenAI = async () => {
        if (!process.env.GOOGLE_API_KEY) {
          throw new Error('GOOGLE_API_KEY not found');
        }
        return new GoogleGenerativeAI(process.env.GOOGLE_API_KEY);
      };
      
      aiAnalysis = await analyzeRisk(payload, getGenAI);
    } catch (error) {
      console.warn('Gemini AI analysis failed:', error.message);
      // AI 분석 실패해도 aggregateRisk 결과는 반환
    }

    res.json({
      success: true,
      data: {
        aggregateRisk: aggResult,
        aiAnalysis: aiAnalysis,
        payload: payload,
        summary: {
          totalFiles: imageAnalysis.length + documentAnalysis.length,
          imageCount: imageAnalysis.length,
          documentCount: documentAnalysis.length,
          overallRisk: aggResult.level,
          overallScore: aggResult.overall
        }
      }
    });

  } catch (error) {
    console.error('Error in comprehensive analysis:', error);
    res.status(500).json({ 
      success: false, 
      error: error.message 
    });
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
      'POST /predict': 'Predict fraud probability using ML model',
      'POST /api/analyze-comprehensive': 'Comprehensive analysis combining images and documents with AI summary'
    }
  });
});

// Start server
app.listen(PORT, () => {
  console.log(`🚀 Server is running on port ${PORT}`);
  console.log(`📊 Health check: http://localhost:${PORT}/health`);
});

module.exports = app;
