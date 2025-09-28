const express = require('express');
const router = express.Router();

// @google/generative-ai 는 ESM이므로 동적 import 사용 (CommonJS 호환)
let _genAICache = null;
async function getGenAI() {
  if (_genAICache) return _genAICache;
  const { GoogleGenerativeAI } = await import('@google/generative-ai');
  if (!process.env.GOOGLE_API_KEY) {
    throw new Error('Missing GOOGLE_API_KEY');
  }
  _genAICache = new GoogleGenerativeAI(process.env.GOOGLE_API_KEY || 'AIzaSyByIXKwtmYA7YtKv-ewU1HI8C0FK08lXAM');
  return _genAICache;
}

// 캐노니컬 스키마 지시문 (우리 모델 입력형 JSON으로 정규화)
const CANONICAL_PROMPT = `
You will receive an insurance claim PDF.
Extract fields and return STRICT JSON only (no prose, no code fences).
If unknown, use sensible defaults: strings="unknown", numbers=0.
Output a single JSON object with EXACTLY these keys:

["Month","WeekOfMonth","DayOfWeek","Make","AccidentArea","Sex","MaritalStatus",
 "Age","Fault","PolicyType","VehicleCategory","VehiclePrice","PolicyNumber",
 "RepNumber","Deductible","DriverRating","Days:Policy-Accident","Days:Policy-Claim",
 "PastNumberOfClaims","AgeOfVehicle","AgeOfPolicyHolder","PoliceReportFiled",
 "WitnessPresent","AgentType","NumberOfSuppliments","AddressChange-Claim",
 "NumberOfCars","Year","BasePolicy","ClaimAmount"]

Normalization:
- AccidentArea: "Urban" | "Rural"
- Fault: "Policy Holder" | "Third Party"
- PoliceReportFiled, WitnessPresent: "Yes" | "No"
- VehiclePrice: one of
  less_than_20000, 11000_to_20000, 20000_to_29000,
  30000_to_39000, 39000_to_52000, 52000_to_69000, more_than_69000
- Numbers must be numeric (no units or commas).
Return ONLY the JSON object.
`.trim();

// 누락 키 채우기 (우리 스키마 강제)
const REQUIRED_KEYS = [
  "Month","WeekOfMonth","DayOfWeek","Make","AccidentArea","Sex","MaritalStatus",
  "Age","Fault","PolicyType","VehicleCategory","VehiclePrice","PolicyNumber",
  "RepNumber","Deductible","DriverRating","Days:Policy-Accident","Days:Policy-Claim",
  "PastNumberOfClaims","AgeOfVehicle","AgeOfPolicyHolder","PoliceReportFiled",
  "WitnessPresent","AgentType","NumberOfSuppliments","AddressChange-Claim",
  "NumberOfCars","Year","BasePolicy","ClaimAmount"
];
const NUMERIC_KEYS = new Set([
  "Age","PolicyNumber","RepNumber","Deductible","DriverRating",
  "Days:Policy-Accident","Days:Policy-Claim","PastNumberOfClaims",
  "AgeOfVehicle","AgeOfPolicyHolder","NumberOfSuppliments",
  "NumberOfCars","Year","ClaimAmount","WeekOfMonth"
]);

function fillMissingKeys(obj) {
  const out = { ...obj };
  for (const k of REQUIRED_KEYS) {
    if (!(k in out)) out[k] = NUMERIC_KEYS.has(k) ? 0 : "unknown";
  }
  return out;
}

// POST /parse-claim-pdf
// body: { pdf: <base64 of PDF>, extraPrompt?: <string> }
router.post('/parse-claim-pdf', async (req, res) => {
  try {
    const { pdf, extraPrompt = "" } = req.body || {};
    if (!pdf) {
      return res.status(400).json({ error: "No PDF base64 provided in 'pdf'." });
    }

    const genAI = await getGenAI();
    const model = genAI.getGenerativeModel({ model: 'gemini-2.5-flash' });

    const result = await model.generateContent({
      contents: [
        { role: 'user', parts: [{ text: CANONICAL_PROMPT + (extraPrompt ? `\n\n${extraPrompt}` : "") }] },
        {
          role: 'user',
          parts: [
            {
              inlineData: {
                mimeType: 'application/pdf', // base64 PDF
                data: pdf,                   // data: prefix 없이 순수 base64
              }
            }
          ]
        }
      ],
      generationConfig: {
        responseMimeType: 'application/json', // JSON 강제
      }
    });

    // JSON 텍스트 수신 (이미 JSON으로 올 확률 큼)
    let text = result.response.text();

    // 혹시 코드펜스가 들어오면 제거
    const fence = text.match(/```json\s*([\s\S]*?)```/i);
    if (fence) text = fence[1];

    let obj = {};
    try {
      obj = JSON.parse(text);
      if (Array.isArray(obj)) obj = obj[0] || {};
    } catch (e) {
      // 모델이 가끔 포맷 틀릴 때 대비
      return res.status(502).json({ error: 'Model returned non-JSON response', raw: text });
    }

    // 스키마 보강(누락키 기본값)
    const canonical = fillMissingKeys(obj);

    return res.status(200).json({
      canonical,        // 우리 모델 입력 스키마의 단일 객체
      for_predict: [canonical], // /predict 배열 입력용
      model: 'gemini-2.5-flash'
    });
  } catch (err) {
    console.error('parse-claim-pdf error:', err);
    return res.status(500).json({ error: err?.message || 'Failed to parse PDF' });
  }
});

module.exports = router;
