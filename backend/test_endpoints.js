const axios = require('axios');

const BASE_URL = 'http://localhost:8000';

async function testEndpoints() {
  console.log('🧪 Testing iSpy Backend Endpoints...\n');

  try {
    // Test health endpoint
    console.log('1. Testing health endpoint...');
    const healthResponse = await axios.get(`${BASE_URL}/health`);
    console.log('✅ Health check:', healthResponse.data);
    console.log('');

    // Test root endpoint
    console.log('2. Testing root endpoint...');
    const rootResponse = await axios.get(`${BASE_URL}/`);
    console.log('✅ Root endpoint:', rootResponse.data);
    console.log('');

    // Test fraud analysis with a sample image URL
    console.log('3. Testing fraud analysis endpoint...');
    const fraudResponse = await axios.post(`${BASE_URL}/analyze_fraud`, {
      image_url: 'https://images.unsplash.com/photo-1449824913935-59a10b8d2000?w=400&h=300&fit=crop'
    });
    console.log('✅ Fraud analysis result:', JSON.stringify(fraudResponse.data, null, 2));
    console.log('');

    console.log('🎉 All tests completed successfully!');

  } catch (error) {
    console.error('❌ Test failed:', error.response?.data || error.message);
  }
}

// Run tests if this file is executed directly
if (require.main === module) {
  testEndpoints();
}

module.exports = { testEndpoints };
