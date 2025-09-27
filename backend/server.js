const express = require('express');
const app = express();
const PORT = process.env.PORT || 3000;

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
