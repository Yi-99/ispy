const express = require('express');
const app = express();
const PORT = process.env.PORT || 3000;
const { spawn } = require('child_process');
const PYTHON_BIN = process.env.PYTHON_BIN || 'python3';
const MODEL_BUNDLE_PATH = process.env.MODEL_BUNDLE_PATH || 'service/fraud_ensemble_bundle.joblib';
const THRESHOLD = process.env.THRESHOLD || '0.45';

// Middleware
app.use(express.json());

// Health check endpoint
app.get('/health', (req, res) => {
  res.status(200).json({
    status: 'OK',
    message: 'Server is running',
    timestamp: new Date().toISOString(),
    uptime: process.uptime()
  });
});

// Detect image endpoint
app.post('/detect_image', (req, res) => {
  const image_path = req.body.image_path;
  const result = detect_image(image_path);
  res.json({ result });
});

//Fraud prediction endpoint
app.post('/predict', async (req, res) => {
  try {
    // 유연 입력: 단건 객체 or 배열
    const payload = req.body;

    // Python scorer 호출
    const py = spawn(PYTHON_BIN, ['service/scorer.py'], {
      env: {
        ...process.env,
        MODEL_BUNDLE_PATH,
        THRESHOLD
      }
    });

    let stdout = '';
    let stderr = '';

    py.stdout.on('data', (d) => (stdout += d.toString()));
    py.stderr.on('data', (d) => (stderr += d.toString()));

    py.on('error', (err) => {
      console.error('Python spawn error:', err);
    });

    py.on('close', (code) => {
      if (code !== 0) {
        console.error('Python scorer exited with code', code, 'stderr:', stderr);
        return res.status(500).json({ error: 'Scoring failed', detail: stderr });
      }
      try {
        const out = JSON.parse(stdout);
        return res.json(out);
      } catch (e) {
        console.error('JSON parse error:', e, stdout);
        return res.status(500).json({ error: 'Invalid scorer output', detail: stdout });
      }
    });

    // stdin에 payload 전달
    py.stdin.write(JSON.stringify(payload));
    py.stdin.end();
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: 'Unexpected server error' });
  }
});


// Basic root endpoint
app.get('/', (req, res) => {
  res.json({
    message: 'iSpy Backend Server',
    version: '1.0.0',
    health: '/health'
  });
});

// Start server
app.listen(PORT, () => {
  console.log(`🚀 Server is running on port ${PORT}`);
  console.log(`📊 Health check: http://localhost:${PORT}/health`);
});

module.exports = app;
