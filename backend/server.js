const express = require('express');
const app = express();
const PORT = process.env.PORT || 3000;
const { processImageFromLink } = require('./service/image_processor');

// Middleware
app.use(express.json());

// Process image endpoint - backend handles everything
app.post('/process_incident', async (req, res) => {
  try {
    const { incidentId } = req.body;
    
    // Validate required field
    if (!incidentId) {
      return res.status(400).json({
        error: 'Missing required field: incidentId is required'
      });
    }
    
    // Process the image using incident ID
    const result = await processImageByIncidentId(incidentId);
    
    res.json({
      success: true,
      data: result
    });
    
  } catch (error) {
    console.error('Error in /process_incident endpoint:', error);
    res.status(500).json({
      success: false,
      error: 'Internal server error',
      message: error.message
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
