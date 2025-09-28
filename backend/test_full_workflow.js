#!/usr/bin/env node

const fs = require('fs');

async function testFullWorkflow() {
  console.log('🧪 Testing Full Workflow: PDF Scan → Fraud Prediction\n');
  
  try {
    // Step 1: PDF 스캔
    console.log('📄 Step 1: Scanning PDF...');
    const pdfPath = './test/bundle_acord_1.pdf';
    const pdfBuffer = fs.readFileSync(pdfPath);
    const pdfBase64 = pdfBuffer.toString('base64');
    
    const scanResponse = await fetch('http://localhost:8000/api/parse-claim-pdf', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        pdf: pdfBase64,
        extraPrompt: "Focus on extracting all available claim information accurately."
      })
    });
    
    if (!scanResponse.ok) {
      throw new Error(`PDF scan failed: ${scanResponse.status}`);
    }
    
    const scanResult = await scanResponse.json();
    console.log('✅ PDF scan completed');
    console.log('📊 Extracted claim data:', JSON.stringify(scanResult.canonical, null, 2));
    
    // Step 2: 사기 탐지
    console.log('\n🔍 Step 2: Running fraud prediction...');
    const predictResponse = await fetch('http://localhost:8000/predict', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(scanResult.canonical)
    });
    
    if (!predictResponse.ok) {
      throw new Error(`Fraud prediction failed: ${predictResponse.status}`);
    }
    
    const predictResult = await predictResponse.json();
    console.log('✅ Fraud prediction completed');
    console.log('📊 Prediction result:', JSON.stringify(predictResult, null, 2));
    
    // Step 3: 결과 요약
    console.log('\n📋 Workflow Summary:');
    console.log(`- Claim Amount: $${scanResult.canonical.ClaimAmount}`);
    console.log(`- Vehicle: ${scanResult.canonical.Make} ${scanResult.canonical.Year}`);
    console.log(`- Fraud Probability: ${(predictResult.data[0].proba * 100).toFixed(2)}%`);
    console.log(`- Fraud Decision: ${predictResult.data[0].decision === 1 ? 'FRAUD' : 'LEGITIMATE'}`);
    console.log(`- Risk Level: ${predictResult.data[0].proba > 0.7 ? 'HIGH' : predictResult.data[0].proba > 0.4 ? 'MEDIUM' : 'LOW'}`);
    
  } catch (error) {
    console.error('❌ Workflow test failed:', error.message);
    console.error('Stack trace:', error.stack);
  }
}

testFullWorkflow();
