// 필요: getGenAI()는 네 코드의 기존 구현 사용 (GoogleGenerativeAI 캐시)
// const genAI = await getGenAI(); 형태로 호출

function aggregateRisk(
    { docProbas = [], imgProb, imgUnc, aiScore },
    { wDoc = 0.3, wImg = 0.1, wGen = 0.6, thrLow = 0.3, thrHigh = 0.6 } = {}
  ) {
    // Handle both single value and array for docProbas
    const docProbasArray = Array.isArray(docProbas) ? docProbas : [docProbas];
    const s_doc = docProbasArray.length > 0 ? Math.max(0, ...docProbasArray) : 0;
    const s_img = Number(imgProb || 0);
    const u_img = Math.min(Math.max(Number(imgUnc || 0), 0), 1);
    const s_gen = Number(aiScore || 0);
  
    const w_img_eff = wImg * (1 - u_img);
    const terms = [1 - wDoc * s_doc, 1 - w_img_eff * s_img, 1 - wGen * s_gen];
    const overall = 1 - terms.reduce((a, b) => a * b, 1);
  
    const level =
      overall >= thrHigh ? "HIGH" : overall >= thrLow ? "MEDIUM" : "LOW";
    return { overall: Number(overall.toFixed(3)), level, thresholds: { low: thrLow, high: thrHigh } };
  }
  
  // ---------- NEW: 프롬프트와 파서 ----------
  
  const SCHEMA_PROMPT = `
  You are a risk summarizer. You will receive a JSON payload that already includes computed risk scores.
  Return STRICT JSON ONLY matching this schema. DO NOT add code fences or extra text.
  Never change provided numeric values in the payload (scores, probabilities). You may add short textual "drivers" and "recommendations".
  
  SCHEMA:
  {
    "overall": {
      "score": number,                  // echo payload.overall.score exactly
      "level": "LOW"|"MEDIUM"|"HIGH",   // echo payload.overall.level exactly
      "thresholds": { "low": number, "high": number }, // echo from payload if present, else {0.3, 0.6}
      "drivers": string[]               // 2-4 concise reasons (text only)
    },
    "signals": {
      "document": {
        "type": "tabular_fraud",
        "items": [ { "proba": number, "decision": 0|1 } ], // echo provided items if present; else []
        "top_suspicious": { "index": number, "proba": number } | null,
        "model": string
      },
      "image_fraud": {
        "type": "visual_fraud",
        "proba": number|null,
        "risk": "Low"|"Medium"|"High"|null,
        "uncertainty": number|null,
        "model": string|null
      },
      "ai_generated": {
        "type": "gen_detector",
        "score": number|null,
        "risk": "Low"|"Medium"|"High"|null
      }
    },
    "consistency": {
      "media_consistency": "consistent"|"inconsistent"|"unknown",
      "notes": string[]
    },
    "recommendations": string[]         // 2-4 short action phrases
  }
  
  RULES:
  - Use the payload values as the source of truth for all numbers and model names.
  - If a field is missing in the payload, set it to a sensible default (null, [], or literal type in schema).
  - Keep drivers/recommendations short (max ~10 words each).
  - No speculation beyond the payload context.
  `.trim();
  
  function toCompactJSON(obj) {
    // 숫자 소수점 과다 방지 (3자리), 안정적 직렬화
    return JSON.stringify(obj, (k, v) => {
      if (typeof v === "number") return Number(v.toFixed ? v.toFixed(6) : v);
      return v;
    });
  }
  
  function parseJSONStrict(text) {
    // 코드펜스가 실수로 들어오면 제거
    const fence = text.match(/```json\s*([\s\S]*?)```/i);
    const clean = fence ? fence[1] : text;
    return JSON.parse(clean);
  }
  
  function fillDefaults(api) {
    // 기본 키 보강
    api.overall = api.overall || {};
    if (typeof api.overall.score !== "number") api.overall.score = 0;
    if (!api.overall.level) api.overall.level = "LOW";
    if (!api.overall.thresholds) api.overall.thresholds = { low: 0.3, high: 0.6 };
    if (!Array.isArray(api.overall.drivers)) api.overall.drivers = [];
  
    api.signals = api.signals || {};
    api.signals.document = api.signals.document || { type: "tabular_fraud", items: [], top_suspicious: null, model: "ensemble_v1" };
    if (!Array.isArray(api.signals.document.items)) api.signals.document.items = [];
    if (api.signals.document.top_suspicious === undefined) api.signals.document.top_suspicious = null;
    if (!api.signals.document.type) api.signals.document.type = "tabular_fraud";
    if (!api.signals.document.model) api.signals.document.model = "ensemble_v1";
  
    api.signals.image_fraud = api.signals.image_fraud || { type: "visual_fraud", proba: null, risk: null, uncertainty: null, model: null };
    if (!api.signals.image_fraud.type) api.signals.image_fraud.type = "visual_fraud";
  
    api.signals.ai_generated = api.signals.ai_generated || { type: "gen_detector", score: null, risk: null };
    if (!api.signals.ai_generated.type) api.signals.ai_generated.type = "gen_detector";
  
    api.consistency = api.consistency || { media_consistency: "unknown", notes: [] };
    if (!Array.isArray(api.consistency.notes)) api.consistency.notes = [];
  
    if (!Array.isArray(api.recommendations)) api.recommendations = [];
  
    return api;
  }
  
  async function analyzeRisk(payload, getGenAI) {
    // payload는 네가 만든 aggregateRisk 출력 + signals를 포함한 객체여야 함
    // 예: {
    //   overall: { score, level, thresholds? },
    //   signals: { document: {...}, image_fraud: {...}, ai_generated: {...} },
    //   consistency: {...}
    // }
    const genAI = await getGenAI();
    const model = genAI.getGenerativeModel({ model: "gemini-2.5-flash" });
  
    const compact = toCompactJSON(payload);
    const result = await model.generateContent({
      contents: [
        { role: "user", parts: [{ text: SCHEMA_PROMPT }] },
        { role: "user", parts: [{ text: `PAYLOAD:\n${compact}` }] }
      ],
      generationConfig: {
        responseMimeType: "application/json",
        temperature: 0.2,
      }
    });
  
    let txt = result?.response?.text?.() || "";
    let obj = parseJSONStrict(txt);
    obj = fillDefaults(obj);
    return obj;
  }
  
  module.exports = {
    aggregateRisk,
    analyzeRisk,
  };
  