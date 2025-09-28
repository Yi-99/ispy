#!/usr/bin/env node

const fs = require('fs');
const path = require('path');

async function testDocumentScanAPI() {
  console.log('🧪 Testing Document Scan API with bundle_acord_1.pdf...\n');
  
  try {
    // PDF 파일을 base64로 인코딩
    const pdfPath = './test/bundle_acord_1.pdf';
    if (!fs.existsSync(pdfPath)) {
      throw new Error(`PDF file not found: ${pdfPath}`);
    }
    
    const pdfBuffer = fs.readFileSync(pdfPath);
    const pdfBase64 = pdfBuffer.toString('base64');
    
    console.log(`📄 PDF loaded: ${pdfPath} (${pdfBuffer.length} bytes)`);
    console.log(`📄 Base64 length: ${pdfBase64.length} characters\n`);
    
    // API 호출
    const response = await fetch('http://localhost:8000/api/parse-claim-pdf', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        pdf: pdfBase64,
        extraPrompt: "Focus on extracting all available claim information accurately."
      })
    });
    
    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`HTTP ${response.status}: ${errorText}`);
    }
    
    const result = await response.json();
    
    console.log('✅ Test passed!');
    console.log('📊 Extracted Data:\n', JSON.stringify(result, null, 2));
    
  } catch (error) {
    console.error('❌ Test failed:', error.message);
    console.error('Stack trace:', error.stack);
  }
}

testDocumentScanAPI();
