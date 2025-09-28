import { supabase } from '../lib/supabase';

export interface ImageAnalysis {
  id?: string;
  analysis_name: string;
  filename: string;
  file_size: number;
  file_url: string;
  fraud_score: number;
  ai_score: number;
  is_fraudulent: boolean;
  risk_level: 'LOW' | 'MEDIUM' | 'HIGH';
  ai_analysis: string;
  fraud_analysis: string;
  detected_issues: string;
  cost: number;
  created_at?: string;
}

export interface AnalysisMetadata {
  id?: string;
  analysis_name: string;
  total_files: number;
  completed_files: number;
  fraud_detected_count: number;
  total_claim_amount: number;
  total_cost: number;
  created_at?: string;
  file_urls: string[];
}

export const fetchAnalysisNames = async () => {
  try {
    const { data, error } = await supabase
      .from('analysis_metadata')
      .select('analysis_name')
      .order('created_at', { ascending: false });

    if (error) {
      console.error('Error fetching analysis names:', error);
      return { success: false, error: error.message };
    }

    return { success: true, data };
  } catch (error) {
    console.error('Error fetching analysis names:', error);
    return { success: false, error: error instanceof Error ? error.message : 'Failed to fetch analysis names' };
  }
};

/**
 * Save analysis metadata to Supabase database
 */
export const saveAnalysisMetadata = async (metadata: AnalysisMetadata) => {
  try {
    const { data, error } = await supabase
      .from('analysis_metadata')
      .insert([metadata])
      .select()
      .single();

    if (error) {
      console.error('Error saving analysis metadata:', error);
      return { success: false, error: error.message };
    }

    return { success: true, data };
  } catch (error) {
    console.error('Error saving analysis metadata:', error);
    return { 
      success: false, 
      error: error instanceof Error ? error.message : 'Failed to save metadata' 
    };
  }
};

export const updateAnalysisMetadata = async (metadata: AnalysisMetadata) => {
  try {
    const { data, error } = await supabase
      .from('analysis_metadata')
      .update(metadata)
      .eq('analysis_name', metadata.analysis_name);

    if (error) {
      console.error('Error updating analysis metadata:', error);
      return { success: false, error: error.message };
    }

    return { success: true, data };
  } catch (error) {
    console.error('Error updating analysis metadata:', error);
    return { 
      success: false, 
      error: error instanceof Error ? error.message : 'Failed to update analysis metadata' 
    };
  }
};

/**
 * Save batch analysis summary to Supabase database
 */
export const saveImageAnalysis = async (batchData: ImageAnalysis) => {
  try {
    const { data, error } = await supabase
      .from('image_metadata')
      .insert([batchData])
      .select()
      .single();

    if (error) {
      console.error('Error saving batch analysis:', error);
      return { success: false, error: error.message };
    }

    return { success: true, data };
  } catch (error) {
    console.error('Error saving batch analysis:', error);
    return { 
      success: false,
      error: error instanceof Error ? error.message : 'Failed to save batch analysis' 
    };
  }
};

/**
 * Fetch all analysis metadata from Supabase database
 */
export const fetchAnalysisMetadata = async () => {
  try {
    const { data, error } = await supabase
      .from('analysis_metadata')
      .select('*')
      .order('created_at', { ascending: false });

    console.log('analysis metadata:', data);  

    if (error) {
      console.error('Error fetching analysis metadata:', error);
      return { success: false, error: error.message }; 
    }

    return { success: true, data };
  } catch (error) {
    console.error('Error fetching analysis metadata:', error);
    return { 
      success: false, 
      error: error instanceof Error ? error.message : 'Failed to fetch analysis metadata' 
    };
  }
};

/**
 * Fetch all image analyses from Supabase database
 */
export const fetchImageAnalyses = async () => {
  try {
    const { data, error } = await supabase
      .from('image_metadata')
      .select('*')
      .order('created_at', { ascending: false });

    console.log('image analyses:', data);  

    if (error) {
      console.error('Error fetching image analyses:', error);
      return { success: false, error: error.message };
    }

    return { success: true, data };
  } catch (error) {
    console.error('Error fetching image analyses:', error);
    return { 
      success: false, 
      error: error instanceof Error ? error.message : 'Failed to fetch image analyses' 
    };
  }
};
