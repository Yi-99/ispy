import { supabase } from '../lib/supabase';

export interface AnalysisMetadata {
  id?: string;
  filename: string;
  file_size: number;
  file_url: string;
  fraud_score: number;
  ai_score: number;
  is_fraudulent: boolean;
  risk_level: 'LOW' | 'MEDIUM' | 'HIGH';
  ai_analysis: string;
  fraud_analysis: string;
  detected_issues: string[];
  created_at?: string;
}

export interface BatchAnalysis {
  id?: string;
  total_files: number;
  completed_files: number;
  fraud_detected_count: number;
  total_claim_amount: number;
  created_at?: string;
  file_urls: string[];
}

/**
 * Save analysis metadata to Supabase database
 */
export const saveAnalysisMetadata = async (metadata: AnalysisMetadata) => {
  try {
    const { data, error } = await supabase
      .from('analysis_results')
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

/**
 * Save batch analysis summary to Supabase database
 */
export const saveBatchAnalysis = async (batchData: BatchAnalysis) => {
  try {
    const { data, error } = await supabase
      .from('batch_analyses')
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
 * Get all analysis results
 */
export const getAnalysisResults = async () => {
  try {
    const { data, error } = await supabase
      .from('analysis_results')
      .select('*')
      .order('created_at', { ascending: false });

    if (error) {
      console.error('Error fetching analysis results:', error);
      return { success: false, error: error.message };
    }

    return { success: true, data };
  } catch (error) {
    console.error('Error fetching analysis results:', error);
    return { 
      success: false, 
      error: error instanceof Error ? error.message : 'Failed to fetch results' 
    };
  }
};
