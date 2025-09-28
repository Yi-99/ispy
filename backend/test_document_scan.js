#!/usr/bin/env node

const { extractCanonicalJsonFromPdf } = require('./service/documentScan.js');

async function testDocumentScan() {
  console.log('🧪 Testing Document Scan with bundle_acord_1.pdf...\n');
  
  try {
    const pdfPath = './test/bundle_acord_1.pdf';
    console.log(`📄 Scanning PDF: ${pdfPath}`);
    
    const startTime = Date.now();
    const result = await extractCanonicalJsonFromPdf(pdfPath);
    const endTime = Date.now();
    
    console.log(`⏱️  Processing time: ${endTime - startTime}ms`);
    console.log('\n📊 Extracted Data:');
    console.log(JSON.stringify(result, null, 2));
    
    // Validate the result
    const requiredFields = [
      "Month", "WeekOfMonth", "DayOfWeek", "Make", "AccidentArea", "Sex", "MaritalStatus",
      "Age", "Fault", "PolicyType", "VehicleCategory", "VehiclePrice", "PolicyNumber",
      "RepNumber", "Deductible", "DriverRating", "Days:Policy-Accident", "Days:Policy-Claim",
      "PastNumberOfClaims", "AgeOfVehicle", "AgeOfPolicyHolder", "PoliceReportFiled",
      "WitnessPresent", "AgentType", "NumberOfSuppliments", "AddressChange-Claim",
      "NumberOfCars", "Year", "BasePolicy", "ClaimAmount"
    ];
    
    console.log('\n🔍 Validation:');
    let missingFields = [];
    let validFields = 0;
    
    for (const field of requiredFields) {
      if (result[field] !== undefined && result[field] !== null) {
        validFields++;
        console.log(`✅ ${field}: ${result[field]}`);
      } else {
        missingFields.push(field);
        console.log(`❌ ${field}: Missing`);
      }
    }
    
    console.log(`\n📈 Summary:`);
    console.log(`- Valid fields: ${validFields}/${requiredFields.length}`);
    console.log(`- Missing fields: ${missingFields.length}`);
    console.log(`- Success rate: ${((validFields / requiredFields.length) * 100).toFixed(1)}%`);
    
    if (missingFields.length > 0) {
      console.log(`- Missing: ${missingFields.join(', ')}`);
    }
    
  } catch (error) {
    console.error('❌ Test failed:', error.message);
    console.error('Stack trace:', error.stack);
  }
}

// Run the test
testDocumentScan().catch(console.error);
