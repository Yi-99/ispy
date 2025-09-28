#!/usr/bin/env node
const { aggregateRisk, analyzeRisk } = require('./service/aggregateRisk');

async function getGenAI() {
  const { GoogleGenerativeAI } = await import('@google/generative-ai');
  const apiKey = process.env.GOOGLE_API_KEY;
  if (!apiKey) throw new Error('Missing GOOGLE_API_KEY');
  return new GoogleGenerativeAI(apiKey);
}

async function main() {
  console.log('🧪 Testing aggregateRisk and analyzeRisk');

  const docProbas = 0.54;
  const imgProb = 0.35;
  const imgUnc = 0.15;
  const aiScore = 0.2;

  const agg = aggregateRisk({ docProbas, imgProb, imgUnc, aiScore }, { wDoc: 0.5, wImg: 0.3, wGen: 0.2, thrLow: 0.3, thrHigh: 0.6 });
  console.log('✅ aggregateRisk:', agg);

  const payload = {
    overall: { score: agg.overall, level: agg.level, thresholds: agg.thresholds },
    signals: {
      document: { type: 'tabular_fraud', items: [{ proba: docProbas, decision: docProbas >= 0.45 ? 1 : 0 }], top_suspicious: { index: 0, proba: docProbas }, model: 'ensemble_v1' },
      image_fraud: { type: 'visual_fraud', proba: imgProb, risk: imgProb > 0.7 ? 'High' : imgProb > 0.4 ? 'Medium' : 'Low', uncertainty: imgUnc, model: 'haywoodsloan/ai-image-detector-deploy' },
      ai_generated: { type: 'gen_detector', score: aiScore, risk: aiScore > 0.7 ? 'High' : aiScore > 0.4 ? 'Medium' : 'Low' }
    },
    consistency: { media_consistency: 'consistent', notes: ['doc and image signals aligned'] },
    recommendations: []
  };

  try {
    const result = await analyzeRisk(payload, getGenAI);
    console.log('✅ analyzeRisk result:\n', JSON.stringify(result, null, 2));
  } catch (e) {
    console.error('❌ analyzeRisk failed:', e.message);
    process.exitCode = 1;
  }
}

main();
