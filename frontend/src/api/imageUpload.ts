import { 
	supabase, 
	STORAGE_BUCKET, 
	UPLOAD_FOLDER, 
	FRAUD_FOLDER, 
	NON_FRAUD_FOLDER 
} from '../lib/supabase';
import axios from 'axios';

export interface UploadResult {
	success: boolean;
	url?: string;
	error?: string;
	filename?: string;
}

export interface AnalysisResult {
	success: boolean;
	data?: {
		isFraudulent: boolean;
		aiScore: string;
		fraudScore: string;
		aiAnalysis: string;
		fraudAnalysis: string;
		riskLevel: 'LOW' | 'MEDIUM' | 'HIGH';
		detectedIssues: string[];
		totalCost: number;
	};
	error?: string;
}

/**
 * Upload an image file to Supabase Storage
 */
export const uploadImage = async (file: File): Promise<UploadResult> => {
	try {
		// Generate unique filename with ISO timestamp and random component
		const now = new Date();
		const timestamp = now.toISOString().replace(/[:.]/g, '-');
		const randomSuffix = Math.floor(Math.random() * 1000);
		const fileExtension = file.name.split('.').pop();
		const filename = `image-${timestamp}-${randomSuffix}.${fileExtension}`;
		
		// Upload file to Supabase Storage
		const { data, error } = await supabase.storage
			.from(STORAGE_BUCKET)
			.upload(`${UPLOAD_FOLDER}/${filename}`, file, {
				cacheControl: '3600',
				upsert: false
			});

		if (error) {
			console.error('Upload error:', error);
			return {
				success: false,
				error: error.message
			};
		}

		// Get public URL for the uploaded file
		const { data: urlData } = supabase.storage
			.from(STORAGE_BUCKET)
			.getPublicUrl(`${UPLOAD_FOLDER}/${filename}`);

		return {
			success: true,
			url: urlData.publicUrl,
			filename: filename
		};
	} catch (error) {
		console.error('Upload error:', error);
		return {
			success: false,
			error: error instanceof Error ? error.message : 'Unknown error occurred'
		};
	}
};

/**
 * Analyze an uploaded image for fraud detection
 * This is a mock implementation - in a real app, this would call your ML API
 */
export const analyzeImage = async (imageUrl: string): Promise<AnalysisResult> => {
	try {
		const res = await axios.post(`${import.meta.env.VITE_API_URL}/analyze_fraud`, {
			image_url: imageUrl
		});

		if (res.status === 200) {
			console.log('fraud analysis results: ', res.data);
			return {
				success: res.data.success,
				data: res.data.data
			};
		}

		return {
			success: false,
			error: res.data.error
		};
	} catch (error) {
		console.error('Analysis error:', error);
		return {
			success: false,
			error: error instanceof Error ? error.message : 'Analysis failed'
		};
	}
};

/**
 * Get all uploaded images for the current session
 */
export const getUploadedImages = async () => {
	try {
		const { data, error } = await supabase.storage
			.from(STORAGE_BUCKET)
			.list('', {
				limit: 100,
				offset: 0,
				sortBy: { column: 'created_at', order: 'desc' }
			});

		if (error) {
			console.error('Error fetching images:', error);
			return { success: false, error: error.message };
		}

		return { success: true, data };
	} catch (error) {
		console.error('Error fetching images:', error);
		return { 
			success: false, 
			error: error instanceof Error ? error.message : 'Failed to fetch images' 
		};
	}
};

/**
 * Delete an uploaded image
 */
export const deleteImage = async (filename: string) => {
	try {
		const { error } = await supabase.storage
			.from(STORAGE_BUCKET)
			.remove([filename]);

		if (error) {
			console.error('Delete error:', error);
			return { success: false, error: error.message };
		}

		return { success: true };
	} catch (error) {
		console.error('Delete error:', error);
		return { 
			success: false, 
			error: error instanceof Error ? error.message : 'Failed to delete image' 
		};
	}
};
