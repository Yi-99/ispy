const { InferenceClient } = require('@huggingface/inference');
const fs = require('fs');
const axios = require('axios');

// For Node.js 18+, use built-in Blob
let Blob;
try {
    // Try Node.js built-in Blob first (Node 18+)
    Blob = globalThis.Blob;
    if (!Blob) {
        // Fallback to buffer Blob polyfill
        const { Blob: BufferBlob } = require('buffer');
        Blob = BufferBlob;
    }
} catch (error) {
    console.error('❌ Failed to load Blob:', error.message);
    // Create a simple Blob-like constructor as last resort
    Blob = class Blob {
        constructor(parts, options = {}) {
            this.parts = parts;
            this.type = options.type || '';
            this.size = parts.reduce((total, part) => total + (part.length || 0), 0);
        }
    };
    console.log('⚠️ Using fallback Blob implementation');
}

console.log('HF_API_KEY:', process.env.HF_API_KEY);

// Initialize Hugging Face client
const client_hf = new InferenceClient(process.env.HF_API_KEY);

/**
 * Detect objects in an image using Hugging Face model
 * 
 * @param {string} imagePath - Path to the image file
 * @returns {Promise<string>} Formatted detection results
 */
async function detectImage(imagePath) {
    try {
        // Check if file exists
        if (!fs.existsSync(imagePath)) {
            throw new Error(`Image file not found: ${imagePath}`);
        }

        // Read the image file
        const imageBuffer = fs.readFileSync(imagePath);
        
        // Create Blob with proper MIME type
        const imageBlob = new Blob([imageBuffer], { 
            type: 'image/jpeg' // Default, could be detected from file extension
        });
        
        console.log('✅ Blob created for local file:');
        console.log('  - Size:', imageBlob.size, 'bytes');
        console.log('  - Type:', imageBlob.type);
        
        // Perform image classification using Hugging Face model
        const output = await client_hf.imageClassification({
            model: process.env.HF_AI_DETECTOR_MODEL_NAME,
            data: imageBlob
        });

        // Format the output to a string
        const formattedParts = output.map(item => 
            `${item.label} score = ${item.score}`
        );
        const finalString = formattedParts.join(", ");

        console.log(finalString);
        return finalString;
    } catch (error) {
        console.error(`Error in detectImage: ${error.message}`);
        return `Error: ${error.message}`;
    }
}

/**
 * Detect objects from image URL using Hugging Face model
 * 
 * @param {string} imageUrl - URL of the image
 * @returns {Promise<string>} Formatted detection results
 */
async function detectImageFromUrl(imageUrl) {
    try {
        // Load image with content type info
        const { buffer: imageBuffer, contentType } = await loadImageFromUrl(imageUrl);
        
        // Create Blob with proper MIME type
        const imageBlob = new Blob([imageBuffer], { 
            type: contentType 
        });
        
        console.log('✅ Blob created for URL image:');
        console.log('  - Size:', imageBlob.size, 'bytes');
        console.log('  - Type:', imageBlob.type);
        
        // Perform image classification using Hugging Face model
        const output = await client_hf.imageClassification({
            model: process.env.HF_AI_DETECTOR_MODEL_NAME,
            data: imageBlob
        });

        // Format the output to a string
        const formattedParts = output.map(item => 
            `${item.label} score = ${item.score}`
        );
        const finalString = formattedParts.join(", ");

        console.log(finalString);
        return finalString;
    } catch (error) {
        console.error(`Error in detectImageFromUrl: ${error.message}`);
        return `Error: ${error.message}`;
    }
}

/**
 * Load image from URL and return as buffer with content type info
 * 
 * @param {string} imageUrl - URL of the image
 * @returns {Promise<{buffer: Buffer, contentType: string}>} Image buffer and content type
 */
async function loadImageFromUrl(imageUrl) {
    try {
        console.log('📥 Fetching image from:', imageUrl);
        const response = await axios.get(imageUrl, {
            responseType: 'arraybuffer',
            timeout: 30000,
            headers: {
                'User-Agent': 'Mozilla/5.0 (compatible; iSpy-Bot/1.0)'
            }
        });
        
        const imageBuffer = Buffer.from(response.data);
        const contentType = response.headers['content-type'] || 'image/jpeg';
        
        console.log('✅ Image loaded successfully:');
        console.log('  - Size:', imageBuffer.length, 'bytes');
        console.log('  - Content-Type:', contentType);
        
        // Validate it's actually an image
        if (!contentType.startsWith('image/')) {
            throw new Error(`Invalid content type: ${contentType}`);
        }
        
        return { buffer: imageBuffer, contentType };
    } catch (error) {
        console.error(`❌ Error loading image:`, error.message);
        throw new Error(`Failed to load image from URL: ${error.message}`);
    }
}

/**
 * Enhanced fraud detection analysis
 * 
 * @param {string} imageUrl - URL of the image to analyze
 * @returns {Promise<Object>} Analysis results with fraud detection
 */
async function analyzeForFraud(imageUrl) {
    console.log('analyzing for fraud: ', imageUrl);

    try {
        if (!client_hf) {
            throw new Error('Hugging Face client not initialized');
        }
        
        // Load image with content type info
        const { buffer: imageBuffer, contentType } = await loadImageFromUrl(imageUrl);
        console.log('✅ Image loaded, creating Blob...');
        console.log('  - Buffer size:', imageBuffer.length, 'bytes');
        console.log('  - Content type:', contentType);
        console.log('  - Blob constructor available:', typeof Blob);
        
        // Create Blob with proper MIME type
        let imageBlob;
        try {
            imageBlob = new Blob([imageBuffer], { 
                type: contentType 
            });
            console.log('✅ Blob created successfully:');
            console.log('  - Size:', imageBlob.size, 'bytes');
            console.log('  - Type:', imageBlob.type);
            console.log('  - Constructor:', imageBlob.constructor.name);
        } catch (blobError) {
            console.error('❌ Failed to create Blob:', blobError.message);
            console.log('🔄 Falling back to raw buffer...');
            // Fallback to raw buffer if Blob creation fails
            imageBlob = imageBuffer;
        }
        
        // Send the Blob to Hugging Face
        const output = await client_hf.imageClassification({
            model: process.env.HF_AI_DETECTOR_MODEL_NAME,
            provider: 'hf-inference',
            data: imageBlob
        });

        console.log('output from client_hf:', output);



        // Analyze results for fraud indicators
        const fraudIndicators = [];
        let fraudScore = 0;
        let isFraudulent = false;

        // Check for specific fraud indicators in the classification results
        output.forEach(item => {
            const label = item.label.toLowerCase();
            const score = item.score;

            // Look for fraud-related keywords
            if (label.includes('artificial')) {
                fraudIndicators.push(`${label} (${(score * 100).toFixed(1)}%)`);
                fraudScore += score * 0.8;
            }
            
            if (label.includes('real')) {
                fraudIndicators.push(`${label} (${(score * 100).toFixed(1)}%)`);
                fraudScore += score * 0.6;
            }
        });

        // Determine if fraudulent based on score
        isFraudulent = fraudScore > 0.5;
        
        // Determine risk level
        let riskLevel = 'LOW';
        if (fraudScore > 0.7) {
            riskLevel = 'HIGH';
        } else if (fraudScore > 0.4) {
            riskLevel = 'MEDIUM';
        }

        // Generate analysis text
        let analysis = '';
        if (isFraudulent) {
            analysis = `Multiple indicators of potential fraud detected. Fraud score: ${(fraudScore * 100).toFixed(1)}%. ${fraudIndicators.length > 0 ? 'Detected issues: ' + fraudIndicators.join(', ') : ''}`;
        } else {
            analysis = `No significant fraud indicators detected. Fraud score: ${(fraudScore * 100).toFixed(1)}%. Analysis appears consistent with legitimate damage.`;
        }

        console.log('fraud analysis finished!', analysis);

        return {
            isFraudulent,
            confidence: Math.max(0.1, Math.min(0.99, fraudScore + 0.2)), // Ensure confidence is between 0.1 and 0.99
            riskLevel,
            analysis,
            detectedIssues: fraudIndicators,
            fraudScore: fraudScore
        };
    } catch (error) {
        console.error(`Error in analyzeForFraud: ${error.message}`);
        return {
            isFraudulent: false,
            confidence: 0.1,
            riskLevel: 'LOW',
            analysis: `Analysis failed: ${error.message}`,
            detectedIssues: [],
            rawClassification: '',
            fraudScore: 0
        };
    }
}

module.exports = {
    detectImage,
    detectImageFromUrl,
    analyzeForFraud
};
