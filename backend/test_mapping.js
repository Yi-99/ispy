const fs = require('fs');
const path = require('path');
const { spawn } = require('child_process');

// Import mapping functions
const claimMapping = require('./service/claim_mapping.js');

// Test files directory
const testDir = './test';

// Get all test files
const testFiles = fs.readdirSync(testDir).filter(file => file.endsWith('.json'));

console.log('🧪 Testing claim mapping with scorer.py...\n');

async function testMapping(testFile) {
  const filePath = path.join(testDir, testFile);
  const carrierName = testFile.replace('_payload.json', '');
  
  console.log(`📋 Testing ${carrierName.toUpperCase()}...`);
  
  try {
    // Read test data
    const rawData = fs.readFileSync(filePath, 'utf8');
    const testData = JSON.parse(rawData);
    
    // Map the data using appropriate mapping function
    let mappedData;
    switch (carrierName) {
      case 'allstate':
        mappedData = claimMapping.mapAllstate(testData);
        break;
      case 'geico':
        mappedData = claimMapping.mapGeico(testData);
        break;
      case 'progressive':
        mappedData = claimMapping.mapProgressive(testData);
        break;
      case 'statefarm':
        mappedData = claimMapping.mapStateFarm(testData);
        break;
      default:
        console.log(`❌ Unknown carrier: ${carrierName}`);
        return;
    }
    
    console.log(`📤 Mapped data:`, JSON.stringify(mappedData, null, 2));
    
    // Send to scorer.py
    const result = await runScorer(mappedData);
    
    console.log(`📊 ${carrierName.toUpperCase()} Result:`, result);
    console.log('─'.repeat(50));
    
  } catch (error) {
    console.error(`❌ Error testing ${carrierName}:`, error.message);
  }
}

function runScorer(data) {
  return new Promise((resolve, reject) => {
    const py = spawn('python3', ['ml/scorer.py'], {
      env: {
        ...process.env,
        MODEL_BUNDLE_PATH: 'ml/models/fraud_ensemble_bundle.joblib'
      }
    });

    let stdout = '';
    let stderr = '';

    py.stdout.on('data', (d) => (stdout += d.toString()));
    py.stderr.on('data', (d) => (stderr += d.toString()));

    py.on('error', (err) => {
      reject(new Error(`Python spawn error: ${err.message}`));
    });

    py.on('close', (code) => {
      if (code !== 0) {
        reject(new Error(`Python scorer exited with code ${code}, stderr: ${stderr}`));
      } else {
        try {
          const result = JSON.parse(stdout);
          resolve(result);
        } catch (e) {
          reject(new Error(`JSON parse error: ${e.message}, stdout: ${stdout}`));
        }
      }
    });

    // Send data to Python script
    py.stdin.write(JSON.stringify(data));
    py.stdin.end();
  });
}

// Run tests for all carriers
async function runAllTests() {
  for (const testFile of testFiles) {
    await testMapping(testFile);
    console.log(''); // Add spacing between tests
  }
  
  console.log('✅ All mapping tests completed!');
}

// Run the tests
runAllTests().catch(console.error);
