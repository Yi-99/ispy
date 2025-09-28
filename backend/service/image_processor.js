const axios = require('axios');
const { createClient } = require('@supabase/supabase-js');

// Initialize Supabase client
const supabaseUrl = process.env.SUPABASE_URL;
const supabaseKey = process.env.SUPABASE_ANON_KEY;
const supabase = createClient(supabaseUrl, supabaseKey);

/**
 * Process image from Supabase link
 * @param {string} imageLink - The Supabase storage link to the image
 * @param {string} incidentId - Unique ID for the car accident incident
 * @returns {Object} - Processing result with incident ID
 */
async function processImageFromLink(imageLink, incidentId) {
  try {
    // Extract the file path from the Supabase URL
    const url = new URL(imageLink);
    const filePath = url.pathname.split('/storage/v1/object/public/')[1];
    
    // Download image from Supabase storage
    const { data: imageData, error: downloadError } = await supabase.storage
      .from(VITE_STORAGE_NAME)
      .download(filePath);
    
    if (downloadError) {
      throw new Error(`Failed to download image: ${downloadError.message}`);
    }
    
    // Convert blob to buffer for processing
    const arrayBuffer = await imageData.arrayBuffer();
    const buffer = Buffer.from(arrayBuffer);
    
    // Here you can add your image processing logic
    // For example, call your existing AI detection service
    // const detectionResult = await detectImage(buffer);
    
    // Store the result with incident ID
    const result = {
      incidentId: incidentId,
      imageLink: imageLink,
      processedAt: new Date().toISOString(),
      status: 'processed',
      // detectionResult: detectionResult // Add this when you integrate AI detection
    };
    
    return result;
    
  } catch (error) {
    console.error('Error processing image:', error);
    return {
      incidentId: incidentId,
      imageLink: imageLink,
      processedAt: new Date().toISOString(),
      status: 'error',
      error: error.message
    };
  }
}

module.exports = {
  processImageFromLink
};